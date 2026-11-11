import argparse
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace


REVISION = 'cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b'
UPSTREAM = {
    'main.py': '439c4d8de398c8876f72806d7ba7e6436b5127a4751f3491505a3f51412a4eeb',
    'cnets.py': '71295b56d8bd5648a1c630af0995f6a23a13671c8e60fe92636fffd9ee786a9c',
    'configs.py': '14ae62877cd9ab12c9076fc70fe940714b9ca980865b00ba974de340df2d1ac5',
    'modeling_llama_kv.py': '18adff2c21e659755eb41d6804fa6dd4b4d486b70d3c749693fbb44cff3855b5',
    'config.json': '5d7879bc4b6abc55169b2fef99b20d36dbaebc30465a45353d98efc5f3e92d8f',
    'ds_config.json': '02b432995b51d8fa1cce7685b7430de7ce14de0419a6729cd9ff0c324c1bec0b',
}
SPECIAL = ('<|begin_of_text|>', '<|end_of_text|>', '<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>')


class NativeTrainingError(ValueError):
    pass


class BudgetComplete(Exception):
    pass


class RecoveryComplete(Exception):
    pass


def recovery_module():
    path = Path(__file__).with_name('native_recovery.py')
    spec = importlib.util.spec_from_file_location('frozen_native_recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('xb') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def records(path):
    path = Path(path)
    if path.suffix not in ('.json', '.jsonl'):
        raise NativeTrainingError('Native training accepts text JSON conversations, never prototype .pt features')
    if path.suffix == '.json':
        values = json.loads(path.read_bytes())
        if not isinstance(values, list):
            raise NativeTrainingError('Native JSON data must be an array of conversations')
        yield from values
    else:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def system_prompt(main_file):
    tree = ast.parse(Path(main_file).read_bytes())
    values = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith('You are a helpful, respectful and honest assistant.')}
    if len(values) != 1:
        raise NativeTrainingError('The inspected native system prompt is not unique')
    return next(iter(values))


def native_row(row, *, prompt=None, canonical_messages=False, split=None):
    if not isinstance(row, dict) or (split is not None and row.get('split', split) != split):
        raise NativeTrainingError('Conversation split differs from the declared input split')
    identifier = row.get('example_id', row.get('id'))
    if not isinstance(identifier, (str, int)) or isinstance(identifier, bool) or not str(identifier):
        raise NativeTrainingError('Every native example requires an identified conversation')
    messages = row.get('messages') if canonical_messages else row.get('conversations')
    if not isinstance(messages, list) or not messages:
        raise NativeTrainingError('Expected canonical messages or native conversations; tensor feature caches are incompatible')
    if any(not isinstance(message, dict) for message in messages):
        raise NativeTrainingError('Conversation turns must be mappings')
    if canonical_messages and messages[0].get('role') == 'system':
        if prompt is None or messages[0].get('content') != prompt:
            raise NativeTrainingError('A custom system prompt would change the native recipe')
        messages = messages[1:]
    if not messages or len(messages) % 2:
        raise NativeTrainingError('Native data must contain complete alternating user/assistant pairs')
    converted = []
    for index, message in enumerate(messages):
        role = message.get('role') if canonical_messages else message.get('from')
        expected = ('user', 'assistant')[index % 2] if canonical_messages else ('human', 'gpt')[index % 2]
        text = message.get('content') if canonical_messages else message.get('value')
        if role != expected or not isinstance(text, str) or not text.strip() or any(token in text for token in SPECIAL):
            raise NativeTrainingError('Native conversations require nonempty alternating text without embedded template control tokens')
        converted.append({'from': ('human', 'gpt')[index % 2], 'value': text})
    return {'id': str(identifier), 'conversations': converted}


def convert(input_path, output_path, *, split, prompt=None):
    if split not in ('train', 'dev'):
        raise NativeTrainingError('Only train/dev conversations may enter native training preparation')
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise NativeTrainingError('Existing datasets are never overwritten')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + '.' + str(os.getpid()) + '.tmp')
    seen = set()
    regimes = set()
    try:
        with temporary.open('xb') as handle:
            for row in records(input_path):
                value = native_row(row, prompt=prompt, canonical_messages=True, split=split)
                regimes.add(row.get('response_regime', 'unspecified'))
                if value['id'] in seen:
                    raise NativeTrainingError('Duplicate native example ID')
                seen.add(value['id'])
                handle.write(canonical(value) + b'\n')
            if not seen:
                raise NativeTrainingError('Native datasets must be nonempty')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {'schema_version': 1, 'input_path': str(Path(input_path).resolve()), 'input_sha256': digest(input_path), 'output_path': str(output_path.resolve()), 'output_sha256': digest(output_path), 'rows': len(seen), 'split': split, 'format': 'official_sharegpt_id_conversations', 'regeneration_performed_by_converter': False, 'input_response_regimes': sorted(regimes), 'text_preserved': True, 'not_a_prototype_feature_conversion': True}
    atomic_json(output_path.with_name(output_path.name + '.manifest.json'), receipt)
    return receipt


def dataset_inventory(path):
    ids, prompts = set(), set()
    count = 0
    for row in records(path):
        value = native_row(row)
        if value['id'] in ids:
            raise NativeTrainingError('Duplicate native example ID: ' + value['id'])
        ids.add(value['id'])
        prompts.add(hashlib.sha256(canonical([item['value'] for item in value['conversations'] if item['from'] == 'human'])).hexdigest())
        count += 1
    if not count:
        raise NativeTrainingError('Native datasets must be nonempty')
    return {'path': str(Path(path).resolve()), 'sha256': digest(path), 'rows_before_official_length_filter': count, 'ids': sorted(ids), 'prompt_identities': sorted(prompts)}


def training_view(value):
    expected = {'bs', 'num_epochs', 'num_workers', 'max_len', 'config_path', 'gradient_checkpoint'}
    if not isinstance(value, dict) or set(value) != expected or value['gradient_checkpoint'] is not True or value['max_len'] != 2048:
        raise NativeTrainingError('The pinned native training configuration contract changed')
    return SimpleNamespace(gradient_checkpointing=value['gradient_checkpoint'], max_len=value['max_len'])


def validate_config(value):
    if value.get('schema_version') != 1 or value.get('source_revision') != REVISION or value.get('implementation') != 'OFFICIAL_EAGLE3_EXTERNAL_COMPATIBILITY_ADAPTER':
        raise NativeTrainingError('Unsupported native training configuration identity')
    if value.get('scope') not in ('engineering', 'native_training') or type(value.get('world_size')) is not int or value['world_size'] < 1:
        raise NativeTrainingError('Invalid native scope or GPU world size')
    budget = value.get('budget', {})
    if set(budget) != {'optimizer_steps'} or (budget['optimizer_steps'] is not None and (type(budget['optimizer_steps']) is not int or budget['optimizer_steps'] < 1)):
        raise NativeTrainingError('Only an enforced optimizer-step ceiling is supported; token budgets are not implemented')
    if value['scope'] == 'engineering' and (budget['optimizer_steps'] not in (1, 2) or not value.get('initialize_from_released_draft')):
        raise NativeTrainingError('Engineering mode requires released initialization and a one/two optimizer-step ceiling')
    if value['scope'] == 'native_training' and value.get('initialize_from_released_draft'):
        raise NativeTrainingError('Released warm starts are currently engineering-only, not scratch native reproduction')
    if value['scope'] == 'native_training':
        gate = value.get('research_gate')
        old = {'reproduction_report', 'reproduction_config_sha256', 'full_model_temperature_one_report'}
        new = {'policy', 'pipeline_certificate', 'independent_review'}
        if not isinstance(gate, dict) or set(gate) not in (old, new) or set(gate) == new and gate['policy'] != 'pipeline-v2':
            raise NativeTrainingError('Native research_gate requires pinned source-bound validation and independent review evidence')
    if value.get('checkpoint_roundtrip') and budget['optimizer_steps'] is None:
        raise NativeTrainingError('The external checkpoint roundtrip requires a bounded optimizer-step run; disable it for the unchanged forty-epoch native command')
    if value.get('recovery') is not None:
        recovery_module().validate_policy(value['recovery'])
        if value.get('checkpoint_roundtrip') or (budget['optimizer_steps'] is not None and budget['optimizer_steps'] % value['recovery']['checkpoint_interval']):
            raise NativeTrainingError('Process recovery requires a checkpoint-aligned budget without same-engine roundtrip')
    return value



def gate_document(root, descriptor, label):
    if not isinstance(descriptor, dict) or set(descriptor) != {'path', 'sha256'} or not isinstance(descriptor['path'], str) or not re.fullmatch('[0-9a-f]{64}', descriptor.get('sha256', '')):
        raise NativeTrainingError('Research gate requires an actual path and SHA256 for ' + label)
    path = Path(descriptor['path'])
    path = (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(Path(root).resolve()) or digest(path) != descriptor['sha256']:
        raise NativeTrainingError('Research gate artifact path or SHA256 differs: ' + label)
    return json.loads(path.read_bytes())


def research_gate(root, config):
    if config['scope'] == 'engineering':
        return {'status': 'synthetic_engineering_only', 'scientific_training_authorized': False}
    evidence = config['research_gate']
    if evidence.get('policy') == 'pipeline-v2':
        report = gate_document(root, evidence['pipeline_certificate'], 'pipeline_certificate')
        gate_document(root, evidence['independent_review'], 'independent_review')
        if report.get('native_training_sha256') != digest(__file__):
            raise NativeTrainingError('Pipeline review covers another native training implementation')
        helper = Path(__file__).with_name('admission_v2.py')
        name = 'frozen_pipeline_' + digest(helper)
        spec = importlib.util.spec_from_file_location(name, helper, submodule_search_locations=[str(helper.parent)])
        verifier = importlib.util.module_from_spec(spec)
        sys.modules[name] = verifier
        spec.loader.exec_module(verifier)
        try:
            verifier.verify_training_scope(root, report, config)
            return verifier.verify_certificate(root, report, evidence['independent_review'])
        except ValueError as error:
            raise NativeTrainingError('Pipeline validation is closed: ' + str(error)) from error
    registry_path = Path(root) / 'configs/reproduction.json'
    if digest(registry_path) != evidence['reproduction_config_sha256']:
        raise NativeTrainingError('Research reproduction configuration SHA256 differs')
    registry = json.loads(registry_path.read_bytes())
    report = gate_document(root, evidence['reproduction_report'], 'reproduction_report')
    gate = report.get('reproduction_gate', {})
    cells = [cell for cell in report.get('cells', []) if cell.get('reference') is not None]
    key = lambda cell: (cell['method'], cell['benchmark'], cell['temperature'])
    expected = {key(cell): cell for cell in registry.get('reference_cells', [])}
    actual = {key(cell): cell for cell in cells}
    if not expected or len(actual) != len(cells) or set(actual) != set(expected) or {cell['method'] for cell in cells} != {'eagle3', 'eagle2', 'hass'} or gate.get('passed') is not True or gate.get('status') != 'passed' or gate.get('reference_cells') != len(expected) or gate.get('passed_reference_cells') != len(expected):
        raise NativeTrainingError('All three complete configured official reproduction matrices must pass')
    for identity, reference in expected.items():
        cell = actual[identity]
        if cell.get('reproduction_gate', {}).get('passed') is not True or any(cell['reference'].get(name) != reference.get(name) for name in ('published_tau', 'source_url', 'expected_questions', 'expected_turns', 'timing_repeats')):
            raise NativeTrainingError('A numerical reproduction reference or passed cell differs')
    full_model = gate_document(root, evidence['full_model_temperature_one_report'], 'full_model_temperature_one_report')
    if full_model.get('status') != 'passed' or full_model.get('models_loaded') is not True or full_model.get('full_model_temperature1_validation_still_required') is True or full_model.get('automatic_pipeline_gate_pass') is not True:
        raise NativeTrainingError('Finite-vocabulary checks or incomplete receipts do not establish the required full-model temperature-one audit')
    helper = Path(__file__).with_name('admission.py')
    if not helper.is_file():
        raise NativeTrainingError('Native research remains blocked: the source-bound admission verifier is missing; claimed passed metadata cannot authorize training')
    spec = importlib.util.spec_from_file_location('native_full_model_admission', helper)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    try:
        return verifier.verify_certificate(root, full_model)
    except verifier.AdmissionError as error:
        raise NativeTrainingError(str(error)) from error


def engineering_row(split, index):
    return {'example_id': 'native-engineering-' + split + '-' + str(index), 'split': split, 'response_regime': 'synthetic_engineering_reference', 'messages': [{'role': 'user', 'content': 'For the synthetic ' + split + ' fixture item ' + str(index) + ', write a short greeting.'}, {'role': 'assistant', 'content': 'Hello. This is synthetic engineering ' + split + ' fixture item ' + str(index) + '.'}]}


def engineering_data(path, split, minimum_rows):
    count = 0
    for count, row in enumerate(records(path), start=1):
        expected = native_row(engineering_row(split, count - 1), canonical_messages=True, split=split)
        if native_row(row) != expected:
            raise NativeTrainingError('The engineering exemption permits only the exact authored synthetic fixture; real data require the native research gate')
    if count < minimum_rows:
        raise NativeTrainingError('Engineering native data must cover all data-parallel ranks')


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise NativeTrainingError('Preparation and execution require an existing Slurm allocation')


def model_record(receipt, identifier, *, verify_content):
    model = receipt['models'][identifier]
    if receipt.get('status') != 'complete' or model.get('status') != 'verified':
        raise NativeTrainingError('A complete verified asset receipt is required')
    path = Path(model['path']).resolve()
    inventory = {}
    for name, item in sorted(model['files'].items()):
        file = path / name
        if not file.is_file() or file.stat().st_size != item['size_bytes'] or (verify_content and digest(file) != item['sha256']):
            raise NativeTrainingError('Prepared model file changed: ' + str(file))
        inventory[name] = {'size_bytes': item['size_bytes'], 'sha256': item['sha256']}
    identity = hashlib.sha256(canonical(inventory)).hexdigest()
    if identity != model['model_identity_sha256']:
        raise NativeTrainingError('Prepared model identity differs')
    return {'path': str(path), 'model_identity_sha256': identity, 'files': inventory, 'source': model['source']}



def trained_paths(root, run_id):
    root = Path(root).resolve()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id):
        raise NativeTrainingError('Invalid native trained-model run identifier')
    storage = (root / 'models').resolve()
    resolved = storage / 'trained/native-eagle3' / run_id
    logical = root / 'models/trained/native-eagle3' / run_id
    if resolved.resolve() != resolved or resolved.exists() or resolved.is_symlink():
        raise NativeTrainingError('Trained model output must be fresh without child symlink aliases')
    return {'logical_path': str(logical), 'resolved_path': str(resolved), 'resolved_storage_root': str(storage)}


def prepare(root, config_path, train_path, validation_path, run_dir):
    require_slurm()
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    config_path = Path(config_path).resolve()
    config = validate_config(json.loads(config_path.read_bytes()))
    admission = research_gate(root, config)
    if not run_dir.is_relative_to(root / 'outputs/native-training') or run_dir.exists():
        raise NativeTrainingError('Choose a fresh run under outputs/native-training')
    sources_path, assets_path = root / 'outputs/preparation/sources.json', root / 'outputs/preparation/assets.json'
    sources, assets = json.loads(sources_path.read_bytes()), json.loads(assets_path.read_bytes())
    source = sources['sources']['EAGLE']
    if sources.get('status') != 'complete' or source.get('revision') != REVISION:
        raise NativeTrainingError('Exact verified official EAGLE source is required')
    checkout = Path(source['path']).resolve()
    upstream = checkout / 'eagle/traineagle3'
    if subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip() != REVISION or subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
        raise NativeTrainingError('Official checkout revision or immutable worktree differs')
    if any(digest(upstream / name) != value for name, value in UPSTREAM.items()):
        raise NativeTrainingError('Pinned official training source bytes differ')
    train, validation = dataset_inventory(train_path), dataset_inventory(validation_path)
    if set(train['ids']) & set(validation['ids']) or set(train['prompt_identities']) & set(validation['prompt_identities']):
        raise NativeTrainingError('Native train and validation IDs/prompts overlap')
    if config['scope'] == 'engineering':
        engineering_data(train_path, 'train', config['world_size'])
        engineering_data(validation_path, 'dev', config['world_size'])
    target = model_record(assets, config['target_id'], verify_content=True)
    draft = model_record(assets, config['draft_id'], verify_content=True) if config['initialize_from_released_draft'] else None
    target_config = json.loads((Path(target['path']) / 'config.json').read_bytes())
    draft_config = json.loads((upstream / 'config.json').read_bytes())
    if target_config.get('architectures') != ['LlamaForCausalLM'] or any(target_config.get(key) != draft_config[key] for key in ('hidden_size', 'vocab_size')):
        raise NativeTrainingError('The pinned native recipe supports its Llama target architecture only')
    if draft and digest(Path(draft['path']) / 'config.json') != UPSTREAM['config.json']:
        raise NativeTrainingError('Released drafter configuration differs from the native architecture')
    trained = trained_paths(root, run_dir.name)
    Path(trained['resolved_path']).mkdir(parents=True, exist_ok=False)
    if Path(trained['logical_path']).resolve() != Path(trained['resolved_path']):
        raise NativeTrainingError('Model storage alias changed during preparation')
    inputs, work = run_dir / 'inputs', run_dir / 'work'
    inputs.mkdir(parents=True)
    work.mkdir()
    for name in UPSTREAM:
        shutil.copyfile(upstream / name, work / name)
    for source_file, name in ((config_path, 'config.json'), (sources_path, 'sources.json'), (assets_path, 'assets.json'), (Path(train_path), 'train' + Path(train_path).suffix), (Path(validation_path), 'validation' + Path(validation_path).suffix), (Path(__file__), 'native_training.py'), (root / 'src/taps_rebuild/pool.py', 'pool.py')):
        shutil.copyfile(source_file, inputs / name)
    if config['scope'] == 'native_training':
        shutil.copyfile(Path(__file__).with_name('admission.py'), inputs / 'admission.py')
        if config['research_gate'].get('policy') == 'pipeline-v2':
            for name in ('admission_v2.py', 'numerical_validation.py'):
                shutil.copyfile(Path(__file__).with_name(name), inputs / name)
    if config.get('recovery') is not None:
        shutil.copyfile(Path(__file__).with_name('native_recovery.py'), inputs / 'native_recovery.py')
    for label, data in (('train', train), ('validation', validation)):
        data['frozen_path'] = str(inputs / (label + Path(data['path']).suffix))
    python = root / config['runtime_python']
    if not python.is_relative_to(root / 'envs/native-eagle3'):
        raise NativeTrainingError('Native training requires its separate envs/native-eagle3 runtime')
    command = [str(python), '-B', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=' + str(config['world_size']), str(inputs / 'native_training.py'), 'worker', '--plan', str(run_dir / 'plan.json')]
    plan = {'schema_version': 1, 'status': 'prepared_not_executed', 'scope': config['scope'], 'implementation': config['implementation'], 'source_revision': REVISION, 'source_checkout': str(checkout), 'project_root': str(root), 'source_receipt_sha256': digest(sources_path), 'asset_receipt_sha256': digest(assets_path), 'config': config, 'training_admission': admission, 'config_sha256': digest(config_path), 'target': target, 'draft': draft, 'train': train, 'validation': validation, 'workdir': str(work), 'run_dir': str(run_dir), 'trained_models': trained, 'command': command, 'frozen_files': {str(path.relative_to(run_dir)): digest(path) for path in [*inputs.iterdir(), *work.iterdir()]}, 'upstream_arguments': ['--basepath', target['path'], '--trainpath', train['frozen_path'], '--testpath', validation['frozen_path'], '--savedir', trained['resolved_path'], '--deepspeed_config', str(work / 'ds_config.json')], 'compatibility_changes': ['dict training configuration exposed as gradient_checkpointing/max_len attributes', 'run-local serialized vocabulary cache', 'offline tracking', 'dedicated native Ninja and verified CUDA executable paths', 'spawn-safe main registration with exact upstream collator AST exposed in workers; upstream source bytes unchanged'], 'engineering_changes': ['released draft trainables and vocabulary initialization', 'bounded optimizer-step checkpoint stop'] if draft else [], 'native_recipe': {'seed': 0, 'epochs': 40, 'max_length': 2048, 'training_time_test_length': 7, 'length_policy': 'official tokenizer filter, no truncation', 'loss': 'official seven-position target-distribution cross entropy weighted 0.8**position', 'features': 'online native target low/mid/high hidden states; no .pt feature cache input', 'ds_config_sha256': UPSTREAM['ds_config.json']}, 'limitations': ['Not a reproduced result until GPU startup and benchmark gates pass', 'Upstream dependency list omits deepspeed/datasets versions', 'No matched-token budget enforcement', 'No claim that published training data/order/checkpoint are reproduced', 'Upstream per-batch model.zero_grad and native optimizer schedule are preserved', 'Native cache vocabulary requires enough observed supervised token types'], 'scientific_results': False, 'created_at': datetime.now(timezone.utc).isoformat()}
    if config.get('recovery') is not None:
        plan['compatibility_changes'].append('Opt-in pinned AST loop hooks for committed process recovery; upstream bytes and loss unchanged')
        plan['native_recipe']['process_recovery'] = config['recovery']
    plan['plan_identity_sha256'] = hashlib.sha256(canonical(plan)).hexdigest()
    atomic_json(run_dir / 'plan.json', plan)
    atomic_json(Path(trained['resolved_path']) / 'native-plan.json', {'plan_identity_sha256': plan['plan_identity_sha256'], 'plan_path': str(run_dir / 'plan.json')})
    return plan


def load_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_bytes())
    identity = plan.pop('plan_identity_sha256')
    if hashlib.sha256(canonical(plan)).hexdigest() != identity:
        raise NativeTrainingError('Native training plan identity differs')
    plan['plan_identity_sha256'] = identity
    directory = Path(plan['run_dir'])
    if path != directory / 'plan.json':
        raise NativeTrainingError('Native plan was relocated')
    for relative, expected in plan['frozen_files'].items():
        if digest(directory / relative) != expected:
            raise NativeTrainingError('Frozen native input differs: ' + relative)
    for model in (plan['target'], plan['draft']):
        if model and any(not (Path(model['path']) / name).is_file() or (Path(model['path']) / name).stat().st_size != item['size_bytes'] for name, item in model['files'].items()):
            raise NativeTrainingError('Prepared model inventory changed after plan freeze')
    trained = plan['trained_models']
    model_directory = Path(trained['resolved_path'])
    if Path(trained['logical_path']).resolve() != model_directory or model_directory.resolve() != model_directory or model_directory.parent.parent.parent != Path(trained['resolved_storage_root']):
        raise NativeTrainingError('Native trained-model storage alias changed')
    marker = json.loads((model_directory / 'native-plan.json').read_bytes())
    if marker.get('plan_identity_sha256') != identity:
        raise NativeTrainingError('Native trained-model run identity differs')
    validate_config(plan['config'])
    research_gate(plan['project_root'], plan['config'])
    if plan['config']['scope'] == 'engineering':
        engineering_data(plan['train']['frozen_path'], 'train', plan['config']['world_size'])
        engineering_data(plan['validation']['frozen_path'], 'dev', plan['config']['world_size'])
    return plan



def validate_vocabulary(torch, d2t, t2d, vocabulary, draft_vocabulary):
    if tuple(d2t.shape) != (draft_vocabulary,) or tuple(t2d.shape) != (vocabulary,) or d2t.dtype != torch.long or t2d.dtype != torch.bool:
        raise NativeTrainingError('Native vocabulary mapping shapes/dtypes differ; scratch data may contain fewer than the required draft-vocabulary token types')
    mapping = d2t + torch.arange(draft_vocabulary, device=d2t.device)
    if not torch.equal(mapping, torch.nonzero(t2d, as_tuple=True)[0]) or bool((mapping < 0).any()) or bool((mapping >= vocabulary).any()):
        raise NativeTrainingError('Native draft vocabulary mapping is inconsistent')


def install_model_adapter(module, plan, torch):
    original = module.Model

    class Model(original):
        def __init__(self, config, ds_config, training_config, load_head=False, load_emb=True, path=None):
            super().__init__(config, ds_config, training_view(training_config), load_head=load_head, load_emb=load_emb, path=path)

        def scandata(self, datapath, tokenizerpath):
            lock_path = Path(plan['workdir']) / 'vocabulary.lock'
            with lock_path.open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if plan['draft'] is None:
                    result = super().scandata(datapath, tokenizerpath)
                    validate_vocabulary(torch, self.d2t, self.t2d, self.vocab_size, self.draft_vocab_size)
                    return result
                state = torch.load(Path(plan['draft']['path']) / 'pytorch_model.bin', map_location='cpu', weights_only=True)
                required = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
                if not required <= set(state) or not {'d2t', 't2d'} <= set(state) or any(name.startswith('target_model.') for name in state):
                    raise NativeTrainingError('Released draft does not contain the complete native trainable state and vocabulary')
                validate_vocabulary(torch, state['d2t'], state['t2d'], self.vocab_size, self.draft_vocab_size)
                self.register_buffer('d2t', state['d2t'])
                self.register_buffer('t2d', state['t2d'])
                result = self.load_state_dict(state, strict=False)
                if result.unexpected_keys or any(not (name.startswith('target_model.') or name == 'embed_tokens.weight') for name in result.missing_keys):
                    raise NativeTrainingError('Released draft state is incompatible with the native training architecture')
                self.l1smooth = torch.nn.SmoothL1Loss(reduction='none')

    module.Model = Model
    return original



def tensor_state_identity(value, torch):
    result = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            result.update(canonical({'dtype': str(tensor.dtype), 'shape': list(tensor.shape)}))
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, torch.dtype):
            result.update(canonical(['torch.dtype', str(item)]))
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                result.update(canonical([type(key).__name__, key]))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(canonical([type(item).__name__, len(item)]))
            for element in item:
                visit(element)
        elif type(item).__module__.startswith('deepspeed.runtime.fp16.loss_scaler'):
            result.update(canonical([type(item).__module__, type(item).__qualname__]))
            visit(vars(item))
        elif (type(item).__module__, type(item).__qualname__) == ('deepspeed.utils.tensor_fragment', 'fragment_address'):
            fields = vars(item)
            if set(fields) != {'numel', 'start'} or any(type(value) is not int or value < 0 for value in fields.values()):
                raise NativeTrainingError('Unsupported DeepSpeed fragment-address state')
            result.update(canonical([type(item).__module__, type(item).__qualname__]))
            visit(fields)
        else:
            result.update(canonical([type(item).__name__, item]))
    visit(value)
    return result.hexdigest()


def recovery_targets(engine, torch):
    trainable = {name: value for name, value in engine.module.named_parameters() if value.requires_grad}
    candidates = [(name, value) for name, value in sorted(trainable.items()) if value.is_floating_point() and value.numel() and value.is_contiguous()]
    if not candidates:
        raise NativeTrainingError('Native recovery audit requires an owned trainable floating parameter')
    name, parameter = candidates[0]
    names = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(names, name)
    if any(item != name for item in names):
        raise NativeTrainingError('Native recovery selected different replicated trainable parameters across ranks')
    owned = {id(value) for value in engine.optimizer.single_partition_of_fp32_groups}
    base = engine.optimizer.optimizer
    for group_index, group in enumerate(base.param_groups):
        for parameter_index, master in enumerate(group['params']):
            moment = base.state.get(master, {}).get('exp_avg')
            if id(master) in owned and master.dtype == torch.float32 and torch.is_tensor(moment) and moment.dtype == torch.float32 and moment.device == master.device and moment.numel() and moment.is_contiguous():
                metadata = {'model_parameter': name, 'model_flat_index': 0, 'optimizer_param_group': group_index, 'optimizer_parameter_index': parameter_index, 'optimizer_state_key': 'exp_avg', 'optimizer_flat_index': 0, 'model_scalars_mutated': 1, 'optimizer_scalars_mutated': 1, 'frozen_parameters_selected': False}
                return trainable, parameter.view(-1)[0], moment.view(-1)[0], metadata
    raise NativeTrainingError('Native recovery audit requires an owned local FP32 Adam exp_avg tensor')


def checkpoint_roundtrip(engine, path, torch):
    trainable, model_scalar, moment_scalar, mutation = recovery_targets(engine, torch)
    before = tensor_state_identity(engine.optimizer.state_dict(), torch)
    model_before = tensor_state_identity(trainable, torch)
    counters = (int(engine.global_steps), int(engine.skipped_steps))
    learning_rates = [float(group['lr']) for group in engine.optimizer.param_groups]
    engine.save_checkpoint(str(path), client_state={'scope': 'engineering_optimizer_roundtrip'}, exclude_frozen_parameters=True)
    torch.distributed.barrier()
    with torch.no_grad():
        for label, scalar in (('model', model_scalar), ('optimizer_moment', moment_scalar)):
            original = float(scalar.item())
            if not math.isfinite(original):
                raise NativeTrainingError('Native recovery audit selected a nonfinite scalar')
            scalar.fill_(0.0 if original != 0.0 else 1.0)
            changed = float(scalar.item())
            if changed == original:
                raise NativeTrainingError('Native recovery audit did not perturb its selected scalar')
            mutation[label + '_scalar_before'] = original
            mutation[label + '_scalar_perturbed'] = changed
    torch.distributed.barrier()
    perturbed = tensor_state_identity(engine.optimizer.state_dict(), torch)
    model_perturbed = tensor_state_identity(trainable, torch)
    if perturbed == before or model_perturbed == model_before:
        raise NativeTrainingError('Native recovery perturbations did not change both model and optimizer hashes')
    loaded, client = engine.load_checkpoint(str(path), load_module_strict=False, load_optimizer_states=True, load_lr_scheduler_states=True)
    torch.distributed.barrier()
    after = tensor_state_identity(engine.optimizer.state_dict(), torch)
    model_after = tensor_state_identity({name: value for name, value in engine.module.named_parameters() if value.requires_grad}, torch)
    if loaded is None or client.get('scope') != 'engineering_optimizer_roundtrip' or counters != (int(engine.global_steps), int(engine.skipped_steps)) or before != after or model_before != model_after or len(learning_rates) != len(engine.optimizer.param_groups) or any(not math.isclose(left, float(group['lr']), rel_tol=1e-12, abs_tol=1e-15) for left, group in zip(learning_rates, engine.optimizer.param_groups)):
        raise NativeTrainingError('Native trainable-model/optimizer checkpoint recovery differed')
    return {'status': 'passed', 'verification': 'same_engine_trainable_model_and_optimizer_recovery_after_explicit_perturbation', 'checkpoint_path': str(path), 'loaded_path': str(loaded), 'optimizer_state_sha256_before': before, 'optimizer_state_sha256_perturbed': perturbed, 'optimizer_state_sha256_after': after, 'trainable_model_sha256_before': model_before, 'trainable_model_sha256_perturbed': model_perturbed, 'trainable_model_sha256_after': model_after, 'perturbation': mutation, 'global_step': counters[0], 'skipped_steps': counters[1], 'learning_rates': learning_rates, 'counters_and_learning_rates_perturbed': False, 'same_engine_zero_stage': 2, 'frozen_target_excluded': True, 'mid_epoch_dataset_resume_verified': False}


def install_step_adapter(deepspeed, plan, torch, rank, observation=None):
    original = deepspeed.initialize
    observation = {} if observation is None else observation
    directory = Path(plan['run_dir'])
    cap = plan['config']['budget']['optimizer_steps']
    journal_directory = directory / 'attempts' / ('attempt-' + os.environ.get('TAPS_NATIVE_ATTEMPT_ID', '1')) if plan['config'].get('recovery') is not None else directory
    journal_directory.mkdir(parents=True, exist_ok=True)
    step_log = journal_directory / ('steps-rank-' + str(rank) + '.jsonl')

    def initialize(*args, **kwargs):
        result = original(*args, **kwargs)
        engine = result[0]
        original_step, original_backward, original_forward = engine.step, engine.backward, engine.forward
        counts = {'microbatches': 0, 'input_slots': 0, 'nonpadding_tokens': 0, 'input_assistant_mask_tokens': 0}
        observation.update(engine=engine, token_counts=counts)
        last_loss = None

        def finish(successful):
            destination = Path(plan['trained_models']['resolved_path']) / ('budget_step_' + str(successful))
            if destination.resolve() != destination:
                raise NativeTrainingError('Native checkpoint child cannot be a symlink alias')
            if engine.save_16bit_model(str(destination), exclude_frozen_parameters=True) is False:
                raise NativeTrainingError('Native bounded checkpoint save failed')
            if rank == 0:
                shutil.copyfile(Path(plan['workdir']) / 'config.json', destination / 'config.json')
            torch.distributed.barrier()
            if plan['config'].get('checkpoint_roundtrip', False):
                observation['checkpoint_roundtrip'] = checkpoint_roundtrip(engine, destination / 'training-state', torch)
            raise BudgetComplete('Reached successful native optimizer step ' + str(successful))

        if plan['config'].get('recovery') is not None:
            def stop(path):
                raise RecoveryComplete('Committed native process checkpoint ' + path)
            helpers = SimpleNamespace(canonical=canonical, digest=digest, atomic_json=atomic_json, tensor_state_identity=tensor_state_identity)
            observation['recovery'] = recovery_module().Recovery(plan, torch, rank, observation, helpers, stop, finish)
            observation['recovery'].namespace = observation['native_namespace']

        def forward(*arguments, **options):
            if torch.is_grad_enabled():
                counts['microbatches'] += 1
                counts['input_slots'] += int(options['input_ids'].numel())
                counts['nonpadding_tokens'] += int(options['attention_mask'].sum().item())
                counts['input_assistant_mask_tokens'] += int(options['loss_mask'].sum().item())
            return original_forward(*arguments, **options)

        def backward(loss, *arguments, **options):
            nonlocal last_loss
            last_loss = float(loss.detach().float().item())
            if not math.isfinite(last_loss):
                raise NativeTrainingError('Native training produced nonfinite loss')
            return original_backward(loss, *arguments, **options)

        def step(*arguments, **options):
            before = int(engine.global_steps)
            skipped_before = int(engine.skipped_steps)
            value = original_step(*arguments, **options)
            after = int(engine.global_steps)
            skipped = int(engine.skipped_steps)
            successful = after - skipped
            if after > before:
                norms = engine.get_global_grad_norm()
                norms = None if norms is None else float(norms.item() if hasattr(norms, 'item') else norms)
                norm_nonfinite = norms is not None and not math.isfinite(norms)
                if norm_nonfinite and skipped <= skipped_before:
                    raise NativeTrainingError('Native optimizer applied an update with nonfinite gradient norm')
                item = {'gradient_norm_nonfinite': norm_nonfinite, 'optimizer_step': successful, 'global_step': after, 'skipped_steps': skipped, 'update_skipped': skipped > skipped_before, 'loss': last_loss, 'global_grad_norm': None if norm_nonfinite else norms, 'learning_rates': [float(group['lr']) for group in engine.optimizer.param_groups], 'native_schedule_unchanged': True, 'rank': rank, 'processed_tokens_cumulative_this_dp_rank': dict(counts), 'counts_include_distributed_sampler_padding': True, 'reduced_vocabulary_loss_coverage_not_exposed': True}
                with step_log.open('ab') as handle:
                    handle.write(canonical(item) + b'\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            if 'recovery' in observation:
                observation['recovery'].after_step(before, skipped_before)
            elif cap is not None and successful >= cap:
                finish(successful)
            return value

        engine.backward, engine.step, engine.forward = backward, step, forward
        return result

    deepspeed.initialize = initialize
    return original


def collator_definition(source, expected):
    source = Path(source)
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise NativeTrainingError('Frozen native main differs before collator exposure')
    tree = ast.parse(data)
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DataCollatorWithPadding']
    if len(definitions) != 1 or definitions[0].bases or definitions[0].decorator_list:
        raise NativeTrainingError('Expected one undecorated native collator class without bases')
    if any(isinstance(node, ast.Name) and node.id == '__file__' for node in ast.walk(tree)):
        raise NativeTrainingError('The pinned native main must not depend on its __file__ binding')
    module = ast.Module(body=definitions, type_ignores=[])
    identity = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
    return module, {'native_main_sha256': expected, 'collator_ast_sha256': identity, 'collator': 'DataCollatorWithPadding', 'upstream_bytes_modified': False, 'parent_setup_execution': 'once', 'child_exposes_only_collator': True, 'native_loader_workers_unchanged': 4, 'native_pin_memory_unchanged': True}


def run_native_main(plan, observation=None):
    directory = Path(plan['run_dir'])
    source, adapter = Path(plan['workdir']) / 'main.py', directory / 'inputs/native_training.py'
    if digest(adapter) != plan['frozen_files']['inputs/native_training.py']:
        raise NativeTrainingError('Frozen native adapter differs before spawning')
    _, metadata = collator_definition(source, plan['frozen_files']['work/main.py'])
    namespace = ModuleType('__main__')
    namespace.__file__, namespace.__spec__, namespace.__package__ = str(adapter), None, None
    previous = {name: sys.modules.get(name) for name in ('__main__', '__mp_main__')}
    key = 'TAPS_NATIVE_SPAWN_PLAN_IDENTITY'
    prior_identity = os.environ.get(key)
    os.environ[key] = plan['plan_identity_sha256']
    sys.modules['__main__'] = sys.modules['__mp_main__'] = namespace
    try:
        code = source.read_bytes()
        if plan['config'].get('recovery') is not None:
            observation['native_namespace'] = namespace.__dict__
            namespace._taps_recovery = SimpleNamespace(epochs=lambda *args: observation['recovery'].epochs(*args), batches=lambda *args: observation['recovery'].batches(*args), commit_batch=lambda: observation['recovery'].commit_batch())
            code = recovery_module().transform(source, plan['frozen_files']['work/main.py'])
        exec(compile(code, str(source), 'exec'), namespace.__dict__)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if prior_identity is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior_identity
    return metadata


def expose_spawn_collator(namespace):
    adapter = Path(__file__).resolve()
    directory = adapter.parent.parent
    plan = json.loads((directory / 'plan.json').read_bytes())
    expected = plan.pop('plan_identity_sha256')
    if hashlib.sha256(canonical(plan)).hexdigest() != expected or os.environ.get('TAPS_NATIVE_SPAWN_PLAN_IDENTITY') != expected or Path(plan['run_dir']) != directory:
        raise NativeTrainingError('Spawned native collator plan identity differs')
    if adapter != directory / 'inputs/native_training.py' or digest(adapter) != plan['frozen_files']['inputs/native_training.py'] or Path(plan['workdir']) != directory / 'work':
        raise NativeTrainingError('Spawned native adapter layout or bytes differ')
    source = directory / 'work/main.py'
    definition, metadata = collator_definition(source, plan['frozen_files']['work/main.py'])
    from typing import Any, Dict, List
    namespace.update(torch=importlib.import_module('torch'), Any=Any, Dict=Dict, List=List)
    exec(compile(definition, str(source), 'exec'), namespace)
    return metadata


def worker(plan_path):
    require_slurm()
    plan = load_plan(plan_path)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from packaging.specifiers import SpecifierSet
    expected_python = Path(plan['command'][0]).parent.parent
    if Path(sys.prefix).resolve() != expected_python.resolve() or any(constraint and importlib.metadata.version(name) not in SpecifierSet(constraint) for name, constraint in plan['config']['required_packages'].items()):
        raise NativeTrainingError('Native worker runtime identity differs')
    audit_receipt = json.loads((Path(plan['run_dir']) / 'tokenization-audit.json').read_bytes())
    if audit_receipt.get('status') != 'passed' or audit_receipt.get('plan_identity_sha256') != plan['plan_identity_sha256']:
        raise NativeTrainingError('A matching exact native tokenization audit is required')
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != plan['config']['world_size'] or not torch.cuda.is_available() or torch.cuda.device_count() != world or not 0 <= local_rank < world:
        raise NativeTrainingError('Native distributed worker GPU/world geometry differs')
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability(local_rank) != (8, 0):
        raise NativeTrainingError('The native engineering extension profile requires the allocated A100 architecture')
    import deepspeed
    directory, work = Path(plan['run_dir']), Path(plan['workdir'])
    status = {'status': 'running', 'rank': rank, 'local_rank': local_rank, 'world_size': world, 'scope': plan['scope'], 'plan_identity_sha256': plan['plan_identity_sha256'], 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(local_rank), 'package_versions': {name: importlib.metadata.version(name) for name in plan['config']['required_packages']}, 'scientific_results': False}
    atomic_json(directory / ('rank-' + str(rank) + '.json'), status)
    previous_path, previous_argv, previous_directory = sys.path[:], sys.argv[:], Path.cwd()
    cnets, original_model, original_initialize = None, None, None
    observation = {}
    try:
        os.chdir(work)
        sys.path.insert(0, str(work))
        cnets = importlib.import_module('cnets')
        if Path(cnets.__file__).resolve() != work / 'cnets.py':
            raise NativeTrainingError('Another native model module is already imported')
        original_model = install_model_adapter(cnets, plan, torch)
        original_initialize = install_step_adapter(deepspeed, plan, torch, rank, observation)
        sys.argv = [str(work / 'main.py'), *plan['upstream_arguments'], '--local_rank', str(local_rank)]
        _, status['spawn_compatibility'] = collator_definition(work / 'main.py', plan['frozen_files']['work/main.py'])
        try:
            run_native_main(plan, observation)
            status['status'] = 'native_epochs_complete'
        except BudgetComplete as error:
            status.update(status='bounded_complete', stop_reason=str(error))
        except RecoveryComplete as error:
            status.update(status='checkpointed_for_restart', stop_reason=str(error))
        if plan['config']['budget']['optimizer_steps'] is not None and status['status'] not in ('bounded_complete', 'checkpointed_for_restart'):
            raise NativeTrainingError('Native epochs ended before the requested optimizer-step ceiling')
        if any(digest(work / name) != expected for name, expected in UPSTREAM.items()):
            raise NativeTrainingError('Immutable staged upstream source changed during execution')
        status['completed_at'] = datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if 'engine' in observation:
            engine = observation['engine']
            status.update(successful_optimizer_steps=int(engine.global_steps) - int(engine.skipped_steps), native_global_steps=int(engine.global_steps), skipped_steps=int(engine.skipped_steps), processed_tokens_this_dp_rank=dict(observation['token_counts']), checkpoint_roundtrip=observation.get('checkpoint_roundtrip'), mid_epoch_dataset_resume_supported=False)
            if 'recovery' in observation:
                status.update(process_recovery=observation.get('process_recovery'), restored_rng_and_cursor=observation.get('restored_rng_and_cursor', False), final_state_identity=observation['recovery'].state_identity(), final_rng_identity=observation['recovery'].rng_identity(), mid_epoch_dataset_resume_supported=True, mid_epoch_recovery_independently_validated=False)
        if cnets is not None and original_model is not None:
            cnets.Model = original_model
        if original_initialize is not None:
            deepspeed.initialize = original_initialize
        sys.path[:], sys.argv[:] = previous_path, previous_argv
        os.chdir(previous_directory)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        atomic_json(directory / ('rank-' + str(rank) + '.json'), status)


def runtime_probe(python, expected_prefix, packages):
    script = "import importlib.metadata,json,sys;from packaging.specifiers import SpecifierSet;req=json.loads(sys.argv[1]);v={k:importlib.metadata.version(k) for k in req};assert all(not c or v[k] in SpecifierSet(c) for k,c in req.items()),v;print(json.dumps({'prefix':sys.prefix,'versions':v}))"
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = json.loads(subprocess.check_output([str(python), '-B', '-c', script, json.dumps(packages)], env=environment, text=True, timeout=60))
    if Path(result['prefix']).resolve() != Path(expected_prefix).resolve():
        raise NativeTrainingError('Runtime prefix differs from the dedicated native environment')
    return result




def fixture(root, destination, train_rows=32, dev_rows=8):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if not destination.is_relative_to(root / 'data/engineering') or destination.exists() or type(train_rows) is not int or type(dev_rows) is not int or train_rows < 8 or dev_rows < 8:
        raise NativeTrainingError('Choose fresh engineering data with at least eight train/dev examples')
    destination.mkdir(parents=True)
    output = {'scope': 'synthetic_engineering_only', 'not_official_benchmark_or_training_data': True, 'splits': {}}
    for split, count in (('train', train_rows), ('dev', dev_rows)):
        path = destination / (split + '-canonical.jsonl')
        with path.open('xb') as handle:
            for index in range(count):
                row = engineering_row(split, index)
                handle.write(canonical(row) + b'\n')
            handle.flush()
            os.fsync(handle.fileno())
        output['splits'][split] = convert(path, destination / (split + '.jsonl'), split=split)
    atomic_json(destination / 'fixture.json', output)
    return output


def cuda_environment(root, node, config):
    cuda_home = Path(config.get('cuda_home', '/usr/local/cuda-12.4'))
    compiler = cuda_home / 'bin/nvcc'
    if cuda_home != Path('/usr/local/cuda-12.4') or not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise NativeTrainingError('Verified CUDA12.4 nvcc is required for native FusedAdam startup')
    version = subprocess.check_output([str(compiler), '--version'], text=True, timeout=20)
    if not re.search(r'release\s+12\.4(?:,|\s)', version):
        raise NativeTrainingError('Native CUDA compiler version differs from Torch cu124')
    if not re.fullmatch('[A-Za-z0-9_.-]+', node):
        raise NativeTrainingError('Invalid node name for native extension cache')
    cache = Path(root) / 'cache/native-eagle3' / node / 'torch-extensions'
    cache.mkdir(parents=True, exist_ok=True)
    return {'CUDA_HOME': str(cuda_home), 'TORCH_CUDA_ARCH_LIST': '8.0', 'TORCH_EXTENSIONS_DIR': str(cache)}


def jit_environment(python, environment):
    prefix = Path(python).parent.parent.resolve()
    binary = prefix / 'bin/ninja'
    if not (prefix / 'pyvenv.cfg').is_file() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise NativeTrainingError('The dedicated native environment requires its installed Ninja executable')
    output = environment.copy()
    output['PATH'] = os.pathsep.join([str(prefix / 'bin'), str(Path(output['CUDA_HOME']) / 'bin'), output.get('PATH', os.defpath)])
    if shutil.which('ninja', path=output['PATH']) != str(binary):
        raise NativeTrainingError('Native JIT would select another Ninja executable')
    version = subprocess.check_output([str(binary), '--version'], env=output, text=True, timeout=30).strip()
    if not version:
        raise NativeTrainingError('The native Ninja executable did not report a version')
    return output, {'native_environment_prefix': str(prefix), 'ninja_executable': str(binary), 'ninja_version': version, 'path_prefix': output['PATH'].split(os.pathsep)[:2]}


def allocation(plan):
    require_slurm()
    path = Path(plan['run_dir']) / 'inputs/pool.py'
    spec = importlib.util.spec_from_file_location('native_frozen_pool_guard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hardware = module.allocation(plan['config']['world_size'], command=lambda arguments: subprocess.check_output(arguments, text=True, timeout=30))
    fields = hardware['slurm']['job']
    if fields.get('Account') != 'grad-students' or (fields.get('Partition'), fields.get('QOS')) not in {('high-priority', 'spec-priority'), ('high-priority', 'high-priority'), ('low-priority', 'normal')}:
        raise NativeTrainingError('Unexpected Slurm scheduling settings')
    if hardware['node'] == 'deep-chungus-10':
        holder_path = Path(plan['project_root']).parent / 'rl-infra/cluster/deep10/current.json'
        holder = json.loads(holder_path.read_bytes())
        if str(holder['job_id']) != hardware['job_id'] or fields.get('JobName') != 'deep10-hold':
            raise NativeTrainingError('Protected deep10 holder state differs')
    return {**hardware, 'gpu_count': plan['config']['world_size']}


def audit(plan_path):
    require_slurm()
    plan = load_plan(plan_path)
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    source = Path(plan['workdir']) / 'main.py'
    tree = ast.parse(source.read_bytes())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_dataset_rank']
    if len(functions) != 1:
        raise NativeTrainingError('Exact native dataset preprocessing is missing')
    namespace = {'torch': torch, 'load_dataset': load_dataset, 'train_config': {'max_len': 2048}, '__name__': '__native_dataset_audit__'}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    tokenizer = AutoTokenizer.from_pretrained(plan['target']['path'], local_files_only=True)
    output = {'plan_identity_sha256': plan['plan_identity_sha256'], 'native_main_sha256': UPSTREAM['main.py'], 'preprocessing': 'actual pinned build_dataset_rank extracted without changing its AST', 'splits': {}}
    for label in ('train', 'validation'):
        dataset = namespace['build_dataset_rank'](tokenizer, plan[label]['frozen_path'])
        lengths, supervised = [], []
        for item in dataset:
            ids, mask = item['input_ids'], item['loss_mask']
            if tuple(ids.shape) != tuple(mask.shape) or ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 2048 or not bool(((mask == 0) | (mask == 1)).all()):
                raise NativeTrainingError('Official preprocessing produced an invalid aligned example')
            count = int(mask.sum().item())
            if count <= 0:
                raise NativeTrainingError('Official preprocessing produced an example with no supervised tokens')
            lengths.append(int(ids.shape[1]))
            supervised.append(count)
        if not lengths:
            raise NativeTrainingError('All native examples were removed by official preprocessing')
        output['splits'][label] = {'input_sha256': plan[label]['sha256'], 'rows_before_filter': plan[label]['rows_before_official_length_filter'], 'rows_after_official_filter': len(lengths), 'token_slots': sum(lengths), 'supervised_mask_tokens': sum(supervised), 'minimum_length': min(lengths), 'maximum_length': max(lengths), 'no_truncation': True}
    output['status'] = 'passed'
    atomic_json(Path(plan['run_dir']) / 'tokenization-audit.json', output)
    return output



def run_training(command, cwd, environment, lock_descriptor):
    child = subprocess.Popen(command, cwd=cwd, env=environment, pass_fds=(lock_descriptor,), start_new_session=True)
    previous = {}
    def interrupted(number, frame):
        raise NativeTrainingError('Native launcher interrupted by signal ' + str(number))
    try:
        for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            previous[number] = signal.signal(number, interrupted)
        return child.wait()
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            pass
        else:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if child.poll() is None:
                child.wait(timeout=10)


def summarize_workers(plan, job_id):
    rows = []
    for rank in range(plan['config']['world_size']):
        row = json.loads((Path(plan['run_dir']) / ('rank-' + str(rank) + '.json')).read_bytes())
        if row.get('rank') != rank or row.get('world_size') != plan['config']['world_size'] or row.get('plan_identity_sha256') != plan['plan_identity_sha256'] or row.get('slurm_job_id') != job_id or row.get('status') not in ('bounded_complete', 'native_epochs_complete'):
            raise NativeTrainingError('Native worker completion identity differs')
        cap = plan['config']['budget']['optimizer_steps']
        if cap is not None and row.get('successful_optimizer_steps') != cap:
            raise NativeTrainingError('Native worker did not reach its exact successful-step budget')
        if plan['config'].get('checkpoint_roundtrip') and row.get('checkpoint_roundtrip', {}).get('status') != 'passed':
            raise NativeTrainingError('Native optimizer roundtrip did not pass on every rank')
        rows.append(row)
    keys = ('microbatches', 'input_slots', 'nonpadding_tokens', 'input_assistant_mask_tokens')
    return {'worker_count': len(rows), 'successful_optimizer_steps_by_rank': [row['successful_optimizer_steps'] for row in rows], 'processed_training_tokens_all_dp_ranks': {key: sum(row['processed_tokens_this_dp_rank'][key] for row in rows) for key in keys}, 'counts_include_sampler_padding': True, 'reduced_vocab_loss_coverage_tokens_not_exposed': True, 'checkpoint_roundtrip_all_ranks_passed': all((row.get('checkpoint_roundtrip') or {}).get('status') == 'passed' for row in rows)}


def launch(plan_path, *, resume=False):
    require_slurm()
    plan = load_plan(plan_path)
    config = plan['config']
    hardware = allocation(plan)
    visible = [item['token'] for item in hardware['device_bindings']]
    python = Path(plan['command'][0])
    runtime = runtime_probe(python, python.parent.parent, config['required_packages'])
    directory = Path(plan['run_dir'])
    lock = (directory / 'launch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((directory / 'launch.json').read_bytes()) if (directory / 'launch.json').exists() else None
    recovery_path = None
    attempt = 1
    if resume:
        if not config.get('recovery') or previous is None or previous['status'] not in ('checkpointed_for_restart', 'failed'):
            raise NativeTrainingError('Explicit native resume requires a stopped recovery-enabled attempt')
        latest_path = Path(plan['trained_models']['resolved_path']) / 'recovery/latest.json'
        latest = json.loads(latest_path.read_bytes())
        recovery_path = Path(latest['path'])
        if latest['plan_identity_sha256'] != plan['plan_identity_sha256'] or digest(recovery_path / 'manifest.json') != latest['manifest_sha256'] or previous['hardware']['hardware_identity_sha256'] != hardware['hardware_identity_sha256']:
            raise NativeTrainingError('Native resume checkpoint identity or physical GPU topology differs')
        attempt = previous.get('attempt', 1) + 1
    elif previous is not None:
        raise NativeTrainingError('Native launch is single-use; choose a fresh plan instead of silently restarting')
    environment = {key: value for key, value in os.environ.items() if key not in {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'WANDB_API_KEY'}}
    environment['CUDA_VISIBLE_DEVICES'] = ','.join(visible)
    environment.pop('TAPS_NATIVE_RESUME_CHECKPOINT', None)
    if config.get('recovery') is not None:
        environment['TAPS_NATIVE_ATTEMPT_ID'] = str(attempt)
        attempt_directory = directory / 'attempts' / ('attempt-' + str(attempt))
        attempt_directory.mkdir(parents=True, exist_ok=False)
        if previous is not None:
            for rank in range(config['world_size']):
                shutil.copyfile(directory / ('rank-' + str(rank) + '.json'), directory / 'attempts' / ('attempt-' + str(attempt-1)) / ('rank-' + str(rank) + '.json'))
        if recovery_path is not None:
            environment['TAPS_NATIVE_RESUME_CHECKPOINT'] = str(recovery_path)
    environment.update(cuda_environment(plan['project_root'], hardware['node'], config))
    environment, jit_tools = jit_environment(python, environment)
    cache = Path(plan['project_root']) / 'cache/native-eagle3'
    environment.update(HF_HOME=str(cache / 'huggingface'), HF_HUB_CACHE=str(cache / 'huggingface/hub'), HF_DATASETS_CACHE=str(cache / 'datasets'), XDG_CACHE_HOME=str(cache / 'xdg'), TORCH_HOME=str(cache / 'torch'), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='offline', WANDB_DIR=str(directory), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TORCHINDUCTOR_CACHE_DIR=str(directory / 'inductor'), TRITON_CACHE_DIR=str(directory / 'triton'))
    audit_environment = environment.copy()
    audit_environment['CUDA_VISIBLE_DEVICES'] = ''
    subprocess.run([str(python), '-B', str(directory / 'inputs/native_training.py'), 'audit', '--plan', str(Path(plan_path).resolve())], env=audit_environment, check=True)
    receipt = {'status': 'running', 'hardware': hardware, 'command': plan['command'], 'runtime': runtime, 'jit_tools': jit_tools, 'plan_identity_sha256': plan['plan_identity_sha256'], 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'cuda_visible_devices': visible, 'attempt': attempt, 'resume_checkpoint': str(recovery_path) if recovery_path else None, 'does_not_allocate_or_submit_jobs': True, 'started_at': datetime.now(timezone.utc).isoformat()}
    atomic_json(directory / 'launch.json', receipt)
    try:
        returncode = run_training(plan['command'], plan['workdir'], environment, lock.fileno())
        receipt.update(returncode=returncode)
        if returncode:
            raise NativeTrainingError('Native training subprocess failed: ' + str(returncode))
        statuses = [json.loads((directory / ('rank-' + str(rank) + '.json')).read_bytes()) for rank in range(config['world_size'])]
        if config.get('recovery') and all(row['status'] == 'checkpointed_for_restart' for row in statuses):
            receipt.update(status='checkpointed_for_restart', successful_optimizer_steps_by_rank=[row['successful_optimizer_steps'] for row in statuses])
        else:
            receipt.update(status='complete', worker_summary=summarize_workers(plan, hardware['job_id']))
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(directory / 'launch.json', receipt)
        if config.get('recovery') is not None:
            atomic_json(directory / 'attempts' / ('attempt-' + str(attempt)) / 'launch.json', receipt)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    conversion = commands.add_parser('convert')
    conversion.add_argument('--input', type=Path, required=True)
    conversion.add_argument('--output', type=Path, required=True)
    conversion.add_argument('--split', choices=['train', 'dev'], required=True)
    conversion.add_argument('--native-main', type=Path)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--root', type=Path, required=True)
    preparation.add_argument('--config', type=Path, required=True)
    preparation.add_argument('--train', type=Path, required=True)
    preparation.add_argument('--validation', type=Path, required=True)
    preparation.add_argument('--run-dir', type=Path, required=True)
    for name in ('launch', 'resume', 'worker', 'audit'):
        command = commands.add_parser(name)
        command.add_argument('--plan', type=Path, required=True)
    fixture_parser = commands.add_parser('fixture')
    fixture_parser.add_argument('--root', type=Path, required=True)
    fixture_parser.add_argument('--output-dir', type=Path, required=True)
    fixture_parser.add_argument('--train-rows', type=int, default=32)
    fixture_parser.add_argument('--dev-rows', type=int, default=8)
    commands.add_parser('contract')
    args = parser.parse_args()
    if args.command == 'fixture':
        require_slurm()
        value = fixture(args.root, args.output_dir, args.train_rows, args.dev_rows)
    elif args.command == 'convert':
        require_slurm()
        value = convert(args.input, args.output, split=args.split, prompt=system_prompt(args.native_main) if args.native_main else None)
    elif args.command == 'prepare':
        value = prepare(args.root, args.config, args.train, args.validation, args.run_dir)
    elif args.command in ('launch', 'resume'):
        launch(args.plan, resume=args.command == 'resume')
        return
    elif args.command == 'audit':
        value = audit(args.plan)
    elif args.command == 'worker':
        worker(args.plan)
        return
    else:
        value = {'source_revision': REVISION, 'upstream_files_sha256': UPSTREAM, 'input_format': 'id + conversations[{from:human|gpt,value:text}]', 'feature_format': 'online native target features; prototype .pt inputs unsupported', 'immutable_upstream': True, 'external_config_attributes': ['gradient_checkpointing', 'max_len'], 'new_runtime_required': 'envs/native-eagle3', 'scientific_results': False}
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
elif __name__ == '__mp_main__' and os.environ.get('TAPS_NATIVE_SPAWN_PLAN_IDENTITY'):
    expose_spawn_collator(globals())
