import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

from . import data, native_training, pool


MODULES = ('__init__.py', 'data_pool.py', 'data.py', 'pool.py', 'native_training.py')
ALIGNMENT = 'features[t]=teacher_features[t], input_ids[t]=original[t+1], feature_targets[t]=teacher_features[t+1], labels[t]=original[t+2], teacher_logits[t]=teacher_logits[t+1]'
FIXTURES = (
    ('chat', 'train', 'Say hello to a reader named Lin.', 'Hello, Lin.'),
    ('math', 'train', 'What is 3 plus 4? Give a short answer.', '3 plus 4 is 7.'),
    ('code', 'train', 'Write a Python expression for the square of x.', 'x ** 2'),
    ('chat', 'dev', 'Write a brief thank-you for lending a book.', 'Thank you for lending me your book.'),
    ('math', 'dev', 'What is 6 times 2? Give a short answer.', '6 times 2 is 12.'),
    ('code', 'dev', 'Name the Python function that counts items in a list.', 'Use len(items).'),
    ('chat', 'test', 'Wish a traveler a pleasant journey in one sentence.', 'Have a pleasant journey.'),
    ('code', 'test', 'Show a Python empty dictionary literal.', '{}'),
)


class DataPoolError(ValueError):
    pass


def fixture_rows():
    revision = data.identity(FIXTURES)
    rows = []
    for ordinal, (domain, split, prompt, answer) in enumerate(FIXTURES):
        messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]
        prompt_id = data.prompt_identity(messages)
        rows.append({'example_id': data.identity(['data-pool-engineering', revision, ordinal, prompt_id]), 'prompt_id': prompt_id, 'domain': domain, 'split': split, 'messages': messages, 'response_regime': 'synthetic_engineering_reference', 'source': {'id': 'authored-data-pool-engineering', 'revision': revision, 'row': ordinal}})
    return rows


def validated_rows(rows):
    result, examples, prompts = [], set(), set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict) or any(not isinstance(row.get(name), str) or not row[name] for name in ('example_id', 'prompt_id', 'domain', 'split', 'response_regime')):
            raise DataPoolError('Canonical examples require identified domains, splits and response regimes')
        messages = data.messages_from(row)
        if messages != row.get('messages') or row['split'] not in ('train', 'dev', 'test') or not re.fullmatch('[a-z0-9_-]+', row['domain']):
            raise DataPoolError('Use canonical messages and explicit train/dev/test splits')
        if row['prompt_id'] != data.prompt_identity(messages) or row['example_id'] in examples or row['prompt_id'] in prompts:
            raise DataPoolError('Duplicate IDs/prompts or changed canonical prompt identity')
        examples.add(row['example_id'])
        prompts.add(row['prompt_id'])
        result.append(row)
    if len(result) < 8:
        raise DataPoolError('A full-node data wave requires at least eight distinct examples')
    return result


def shard_rows(rows, count=8):
    if type(count) is not int or count != 8 or len(rows) < count:
        raise DataPoolError('Exactly eight nonempty GPU shards are required')
    return [[(ordinal, row) for ordinal, row in enumerate(rows) if ordinal % count == shard] for shard in range(count)]


def write_records(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        for row in rows:
            handle.write((data.canonical(row) + '\n').encode())
        handle.flush()
        os.fsync(handle.fileno())


def complete_records(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.endswith('\n') or not line.strip():
                raise DataPoolError('Completed output JSONL must contain only terminated nonempty records')
            rows.append(json.loads(line))
    return rows


def options(value):
    config = dict(value)
    if config.get('operation') not in ('regenerate', 'features', 'regenerate-features') or config.get('scope', 'engineering') not in ('engineering', 'research'):
        raise DataPoolError('Unsupported data operation or scope')
    config.setdefault('scope', 'engineering')
    config.setdefault('target_id', 'llama31_instruct_8b')
    config.setdefault('limit', 1000000)
    config.setdefault('seed', 20261109)
    config.setdefault('max_context', 2048)
    config.setdefault('max_new_tokens', 32)
    config.setdefault('temperature', 0.0)
    config.setdefault('include_teacher_logits', False)
    config.setdefault('merge_regenerated', False)
    config.setdefault('feature_layers', None)
    config.setdefault('input', None)
    for name in ('seed', 'limit', 'max_context', 'max_new_tokens'):
        if type(config[name]) is not int or config[name] < (0 if name == 'seed' else 1):
            raise DataPoolError('Invalid bounded data option: ' + name)
    if not 8 <= config['limit'] <= 1000000 or config['max_context'] < 4 or config['max_new_tokens'] >= config['max_context'] or isinstance(config['temperature'], bool) or not isinstance(config['temperature'], (int, float)) or not 0 <= config['temperature'] <= 10:
        raise DataPoolError('Invalid example/context/generation budget')
    if type(config['include_teacher_logits']) is not bool or type(config['merge_regenerated']) is not bool:
        raise DataPoolError('Data export switches must be boolean')
    if config['scope'] == 'engineering' and config['input'] is not None:
        raise DataPoolError('Engineering is restricted to the exact authored fixture; real datasets require the research gate')
    if config['scope'] == 'research' and config['input'] is None:
        raise DataPoolError('Research data must have an explicit canonical source')
    if config['scope'] == 'research' and (not isinstance(config.get('research_gate'), dict) or set(config['research_gate']) != {'reproduction_report', 'reproduction_config_sha256', 'full_model_temperature_one_report'}):
        raise DataPoolError('Research data require pinned all-method reproduction and full-model temperature-one gate evidence')
    return config


def output_directory(root, run_id):
    root = Path(root).resolve()
    if not isinstance(run_id, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id):
        raise DataPoolError('Invalid data run identifier')
    path = root / 'data/exports' / run_id
    if path.resolve() != path or path.exists() or path.is_symlink():
        raise DataPoolError('Choose a fresh export without path traversal or child symlink aliases')
    return path


def select_layers(configuration, requested=None):
    layers, hidden = configuration.get('num_hidden_layers'), configuration.get('hidden_size')
    if type(layers) is not int or layers < 6 or type(hidden) is not int or hidden < 1 or configuration.get('draft_vocab_size') is not None:
        raise DataPoolError('Select a prepared full target model with a valid hidden-layer configuration')
    values = [2, layers // 2, layers - 3] if requested is None else [int(item) for item in requested.split(',')] if isinstance(requested, str) else list(requested)
    if len(values) != len(set(values)) or not values or any(type(layer) is not int or not 0 <= layer <= layers for layer in values):
        raise DataPoolError('Feature layers must be distinct actual HF hidden-state indices')
    return values, len(values) * hidden



def runtime_identity(python):
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)
    environment.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    script = "import importlib.metadata as m,json,sys;print(json.dumps({'prefix':sys.prefix,'versions':{k:m.version(k) for k in ['torch','transformers']}}))"
    value = json.loads(subprocess.check_output([str(python), '-B', '-c', script], env=environment, cwd=python.parent.parent, text=True, timeout=300))
    if Path(value['prefix']).resolve() != python.parent.parent.resolve():
        raise DataPoolError('Data runtime virtual-environment prefix differs')
    value['prefix_resolved'] = str(Path(value['prefix']).resolve())
    return value


def immutable_bytes(path, contents):
    with Path(path).open('xb') as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())


def freeze(root, directory, run_id, config, rows, hardware):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    config = options(config)
    if config['scope'] == 'research':
        native_training.research_gate(root, {'scope': 'native_training', 'research_gate': config.get('research_gate', {})})
    rows = validated_rows(rows)
    if config['scope'] == 'engineering' and rows != fixture_rows():
        raise DataPoolError('Engineering input differs from the exact authored fixture')
    if len(hardware['device_bindings']) != 8 or len({binding['uuid'] for binding in hardware['device_bindings']}) != 8 or len({binding['token'] for binding in hardware['device_bindings']}) != 8 or [binding['shard_index'] for binding in hardware['device_bindings']] != list(range(8)):
        raise DataPoolError('Eight distinct allocated GPUs are required')
    source_files = {}
    for name in MODULES:
        path = root / 'src/taps_rebuild' / name
        if not path.is_file() or path.is_symlink():
            raise DataPoolError('Data source snapshots require regular authored files')
        source_files[name] = path.read_bytes()
    assets_path, registry_path = root / 'outputs/preparation/assets.json', root / 'configs/assets.json'
    asset_bytes, registry_bytes = assets_path.read_bytes(), registry_path.read_bytes()
    assets, registry = json.loads(asset_bytes), json.loads(registry_bytes)
    if assets.get('registry_sha256') != hashlib.sha256(registry_bytes).hexdigest():
        raise DataPoolError('Asset registry differs from its original preparation receipt')
    registered = [asset for asset in registry.get('assets', []) if asset.get('id') == config['target_id']]
    if len(registered) != 1 or assets['models'][config['target_id']]['source'] != registered[0]['source']:
        raise DataPoolError('Target source differs from the pinned asset registry')
    target = native_training.model_record(assets, config['target_id'], verify_content=True)
    target_config = json.loads((Path(target['path']) / 'config.json').read_bytes())
    layers, feature_dim = select_layers(target_config, config['feature_layers'])
    if config['max_context'] > target_config.get('max_position_embeddings', 0):
        raise DataPoolError('Requested data context exceeds the actual target model configuration')
    python = Path(registry['runtime_python'])
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise DataPoolError('The preserved explicit inference runtime Python is unavailable')
    runtime = runtime_identity(python)
    domain_map = {domain: index for index, domain in enumerate(sorted({row['domain'] for row in rows}))}
    directory.mkdir(parents=True, exist_ok=False)
    inputs = directory / 'inputs'
    source = inputs / 'source/src/taps_rebuild'
    source.mkdir(parents=True)
    frozen = []
    for name, contents in source_files.items():
        path = source / name
        immutable_bytes(path, contents)
        frozen.append(path)
    for name, contents in [('assets.json', asset_bytes), ('assets-registry.json', registry_bytes)]:
        path = inputs / name
        immutable_bytes(path, contents)
        frozen.append(path)
    canonical_path = inputs / 'canonical.jsonl'
    write_records(canonical_path, rows)
    frozen.append(canonical_path)
    domain_path = inputs / 'domain-map.json'
    pool.atomic_json(domain_path, domain_map)
    frozen.append(domain_path)
    view = directory / 'runtime-view'
    copied_receipt = view / 'outputs/preparation/assets.json'
    copied_receipt.parent.mkdir(parents=True)
    immutable_bytes(copied_receipt, asset_bytes)
    frozen.append(copied_receipt)
    assignments = []
    for binding, shard in zip(hardware['device_bindings'], shard_rows(rows)):
        rank = binding['shard_index']
        path = inputs / ('shard-' + str(rank) + '.jsonl')
        write_records(path, [row for _, row in shard])
        frozen.append(path)
        assignments.append({'shard_index': rank, 'binding': binding, 'input': str(path), 'ordinals': [ordinal for ordinal, _ in shard], 'example_ids': [row['example_id'] for _, row in shard], 'output_root': str(view / 'data/shards' / str(rank))})
    plan = {'schema_version': 1, 'run_id': run_id, 'root': str(root), 'run_dir': str(directory), 'scope': config['scope'], 'operation': config['operation'], 'config': config, 'target': target, 'target_config_sha256': pool.digest(Path(target['path']) / 'config.json'), 'target_configuration': {key: target_config[key] for key in ('num_hidden_layers', 'hidden_size', 'vocab_size', 'max_position_embeddings')}, 'feature_layers': layers, 'feature_dim': feature_dim, 'feature_contract': 'Prototype aligned teacher cache, not an input format for native official EAGLE3 training', 'domain_map': domain_map, 'runtime_python': str(python), 'runtime_metadata': runtime, 'runtime_root': str(view), 'asset_receipt_sha256': hashlib.sha256(asset_bytes).hexdigest(), 'asset_receipt_original_root': assets.get('root'), 'asset_receipt_is_unchanged_copy_not_new_preparation': True, 'input': {'original_path': config['input'], 'original_sha256': pool.digest(config['input']) if config['input'] else None, 'selected_canonical_sha256': pool.digest(canonical_path), 'selected_examples': len(rows)}, 'assignments': assignments, 'hardware': hardware, 'frozen_files': {str(path.relative_to(directory)): pool.digest(path) for path in frozen}, 'source_hashes': {name: hashlib.sha256(contents).hexdigest() for name, contents in source_files.items()}, 'regeneration_seed_rule': 'int(data.identity([base_seed,prompt_id,target_id])[:8],16); assistant turn adds message index inside unchanged data.regenerate', 'scientific_results': False, 'resume_supported': False, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    plan['plan_identity_sha256'] = pool.identity(plan)
    pool.atomic_json(directory / 'plan.json', plan)
    pool.atomic_json(directory / 'manifest.json', {'status': 'prepared', 'operation': config['operation'], 'scope': config['scope'], 'plan_identity_sha256': plan['plan_identity_sha256'], 'scientific_results': False})
    return plan


def load_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_bytes())
    identity = plan.pop('plan_identity_sha256')
    if pool.identity(plan) != identity:
        raise DataPoolError('Frozen data plan identity differs')
    plan['plan_identity_sha256'] = identity
    directory = Path(plan['run_dir'])
    if path != directory / 'plan.json' or directory.resolve() != directory:
        raise DataPoolError('Data plan or export directory moved')
    for relative, expected in plan['frozen_files'].items():
        file = directory / relative
        if file.resolve() != file or pool.digest(file) != expected:
            raise DataPoolError('Frozen data input/source differs: ' + relative)
    if plan['scope'] == 'research':
        native_training.research_gate(plan['root'], {'scope': 'native_training', 'research_gate': plan['config'].get('research_gate', {})})
    frozen_assets = json.loads((directory / 'inputs/assets.json').read_bytes())
    original_target = frozen_assets['models'][plan['config']['target_id']]['path']
    if Path(original_target).resolve() != Path(plan['target']['path']):
        raise DataPoolError('Frozen receipt target alias changed after full content verification')
    if pool.digest(Path(plan['target']['path']) / 'config.json') != plan['target_config_sha256'] or any((Path(plan['target']['path']) / name).stat().st_size != item['size_bytes'] for name, item in plan['target']['files'].items()):
        raise DataPoolError('Prepared target inventory/configuration changed')
    return plan


def command_for(plan, assignment, operation):
    directory = Path(plan['run_dir'])
    source = assignment['input'] if operation == 'regenerate' or plan['operation'] == 'features' else str(Path(assignment['output_root']) / 'regenerate/records.jsonl')
    destination = str(Path(assignment['output_root']) / operation)
    command = [plan['runtime_python'], '-B', '-m', 'taps_rebuild.data', '--root', plan['runtime_root'], operation, '--input', source, '--output', destination, '--target-id', plan['config']['target_id'], '--limit', str(len(assignment['example_ids'])), '--max-context', str(plan['config']['max_context'])]
    if operation == 'regenerate':
        command += ['--seed', str(plan['config']['seed']), '--temperature', str(plan['config']['temperature']), '--max-new-tokens', str(plan['config']['max_new_tokens'])]
    else:
        command += ['--domain-map', str(directory / 'inputs/domain-map.json'), '--feature-layers', ','.join(map(str, plan['feature_layers']))]
        if plan['config']['include_teacher_logits']:
            command.append('--include-teacher-logits')
    return command


def bounded_path(path, directory, suffix=None):
    path, directory = Path(path), Path(directory).resolve()
    if not path.is_absolute():
        path = directory / path
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not resolved.is_relative_to(directory) or not resolved.is_file() or (suffix is not None and resolved.suffix != suffix):
        raise DataPoolError('Data output path escapes its assigned directory or uses a symlink')
    cursor = path.parent
    while cursor != directory:
        if cursor.is_symlink():
            raise DataPoolError('Data output parent cannot be a symlink')
        cursor = cursor.parent
    return resolved


def validate_operation(plan, assignment, operation):
    destination = Path(assignment['output_root']) / operation
    manifest_path = bounded_path(destination / 'manifest.json', destination, '.json')
    manifest = json.loads(manifest_path.read_bytes())
    expected_input = assignment['input'] if operation == 'regenerate' or plan['operation'] == 'features' else str(Path(assignment['output_root']) / 'regenerate/records.jsonl')
    if manifest.get('status') != 'complete' or manifest.get('command') != operation or manifest.get('source_sha256') != plan['source_hashes']['data.py'] or manifest.get('input_sha256') != pool.digest(expected_input) or manifest.get('asset_receipt_sha256') != plan['asset_receipt_sha256'] or manifest.get('target_model_identity_sha256') != plan['target']['model_identity_sha256'] or manifest.get('target_model_id') != plan['config']['target_id']:
        raise DataPoolError('Completed shard manifest identity differs')
    processing = manifest.get('processing_config', {})
    expected_processing = {'root': plan['runtime_root'], 'command': operation, 'input': expected_input, 'output': str(destination), 'target_id': plan['config']['target_id'], 'limit': len(assignment['example_ids']), 'max_context': plan['config']['max_context']}
    if any(processing.get(name) != value for name, value in expected_processing.items()):
        raise DataPoolError('Shard processing configuration differs from the exact frozen command')
    originals = complete_records(expected_input)
    records = complete_records(bounded_path(destination / 'records.jsonl', destination, '.jsonl'))
    entries = manifest.get('examples', [])
    if len(originals) != len(assignment['example_ids']) or [row['example_id'] for row in originals] != assignment['example_ids'] or len(records) != len(originals) or len(entries) != len(originals):
        raise DataPoolError('Shard coverage differs from the frozen assignment')
    if operation == 'regenerate':
        for original, value, entry in zip(originals, records, entries):
            if any(value.get(key) != original.get(key) for key in original if key not in {'messages', 'response_regime', 'original_messages_sha256', 'generation', 'target_model_identity_sha256'}):
                raise DataPoolError('Regeneration changed an example identity, split or source')
            seed = int(data.identity([plan['config']['seed'], original['prompt_id'], plan['config']['target_id']])[:8], 16)
            generation = value.get('generation', {})
            if value.get('original_messages_sha256') != data.identity(original['messages']) or value.get('target_model_identity_sha256') != plan['target']['model_identity_sha256'] or value.get('response_regime') != 'target_regenerated_all_assistant_turns' or generation.get('seed') != seed or generation.get('temperature') != plan['config']['temperature'] or generation.get('max_new_tokens') != plan['config']['max_new_tokens'] or generation.get('max_context') != plan['config']['max_context'] or data.prompt_identity(data.messages_from(value)) != original['prompt_id']:
                raise DataPoolError('Regenerated prompt, target or deterministic seed differs')
            if entry != {'example_id': value['example_id'], 'prompt_id': value['prompt_id'], 'messages_sha256': data.identity(value['messages'])}:
                raise DataPoolError('Regeneration index differs from actual completed messages')
    else:
        if manifest.get('feature_layers') != plan['feature_layers'] or manifest.get('alignment') != ALIGNMENT or manifest.get('include_teacher_logits') != plan['config']['include_teacher_logits'] or records != entries:
            raise DataPoolError('Feature alignment, layers or actual completed index differs')
        seen = set()
        for original, entry in zip(originals, entries):
            if any(entry.get(key) != original[key] for key in ('example_id', 'prompt_id', 'domain', 'split', 'response_regime')) or entry.get('messages_sha256') != data.identity(original['messages']):
                raise DataPoolError('Feature export changed an example identity or split')
            path = bounded_path(entry['path'], destination, '.pt')
            if path in seen or pool.digest(path) != entry.get('sha256') or type(entry.get('tokens')) is not int or type(entry.get('loss_tokens')) is not int or not 1 <= entry['loss_tokens'] <= entry['tokens']:
                raise DataPoolError('Feature tensor hash, uniqueness or actual token counts differ')
            seen.add(path)
    return {'manifest_path': str(manifest_path), 'manifest_sha256': pool.digest(manifest_path), 'records_path': str(destination / 'records.jsonl'), 'records_sha256': pool.digest(destination / 'records.jsonl'), 'examples': entries, 'records': records}


def worker(plan_path, shard_index, lock_fd):
    plan = load_plan(plan_path)
    if not os.environ.get('SLURM_JOB_ID') or len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 1 or not os.environ.get('CUDA_VISIBLE_DEVICES') or type(shard_index) is not int or not 0 <= shard_index < 8:
        raise DataPoolError('Data workers require one allocated GPU and a valid shard')
    assignment = plan['assignments'][shard_index]
    if os.environ['CUDA_VISIBLE_DEVICES'] != assignment['binding']['token']:
        raise DataPoolError('Worker GPU token differs from its frozen physical assignment')
    lock_path = Path(plan['run_dir']) / 'controller.lock'
    descriptor = os.fstat(lock_fd)
    locked_path = lock_path.stat(follow_symlinks=False)
    if (descriptor.st_dev, descriptor.st_ino) != (locked_path.st_dev, locked_path.st_ino) or not stat.S_ISREG(locked_path.st_mode) or locked_path.st_uid != os.getuid():
        raise DataPoolError('Worker did not inherit the matching owned controller lock')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers')}
    if versions != plan['runtime_metadata']['versions'] or Path(sys.prefix).resolve() != Path(plan['runtime_metadata']['prefix_resolved']):
        raise DataPoolError('Worker runtime differs from the frozen CPU metadata probe')
    receipt = {'status': 'running', 'shard_index': shard_index, 'plan_identity_sha256': plan['plan_identity_sha256'], 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'physical_gpu_uuid': assignment['binding']['uuid'], 'example_ids': assignment['example_ids'], 'operations': {}, 'package_versions': versions, 'scientific_results': False}
    path = Path(plan['run_dir']) / 'workers' / (str(shard_index) + '.json')
    pool.atomic_json(path, receipt)
    try:
        operations = ('regenerate', 'features') if plan['operation'] == 'regenerate-features' else (plan['operation'],)
        for operation in operations:
            command = command_for(plan, assignment, operation)
            subprocess.run(command, check=True, pass_fds=(lock_fd,))
            receipt['operations'][operation] = {key: value for key, value in validate_operation(plan, assignment, operation).items() if key not in ('records', 'examples')}
        receipt.update(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        pool.atomic_json(path, receipt)


def merge(plan):
    directory = Path(plan['run_dir'])
    by_ordinal, regenerated, workers = {}, {}, []
    final_operation = 'features' if plan['operation'] != 'regenerate' else 'regenerate'
    for assignment in plan['assignments']:
        path = directory / 'workers' / (str(assignment['shard_index']) + '.json')
        receipt = json.loads(path.read_bytes())
        if receipt.get('status') != 'complete' or receipt.get('plan_identity_sha256') != plan['plan_identity_sha256'] or receipt.get('example_ids') != assignment['example_ids'] or receipt.get('physical_gpu_uuid') != assignment['binding']['uuid'] or receipt.get('shard_index') != assignment['shard_index'] or receipt.get('cuda_visible_devices') != assignment['binding']['token'] or receipt.get('slurm_job_id') != plan['hardware']['job_id'] or receipt.get('package_versions') != plan['runtime_metadata']['versions']:
            raise DataPoolError('Every GPU shard must have a matching completed receipt')
        result = validate_operation(plan, assignment, final_operation)
        if receipt.get('operations', {}).get(final_operation) != {key: value for key, value in result.items() if key not in ('examples', 'records')}:
            raise DataPoolError('Completed worker output hashes changed before merge')
        for ordinal, entry, original in zip(assignment['ordinals'], result['examples'], complete_records(assignment['input'])):
            if ordinal in by_ordinal:
                raise DataPoolError('Two shards claimed the same global input ordinal')
            by_ordinal[ordinal] = {**entry, 'domain': original['domain'], 'split': original['split'], 'global_ordinal': ordinal, 'shard_index': assignment['shard_index']}
        if 'regenerate' in plan['operation']:
            generation = validate_operation(plan, assignment, 'regenerate')
            if receipt.get('operations', {}).get('regenerate') != {key: value for key, value in generation.items() if key not in ('examples', 'records')}:
                raise DataPoolError('Completed regeneration output hashes changed before merge')
            for ordinal, row in zip(assignment['ordinals'], generation['records']):
                regenerated[ordinal] = row
        workers.append({'path': str(path), 'sha256': pool.digest(path), 'shard_index': assignment['shard_index']})
    if set(by_ordinal) != set(range(plan['input']['selected_examples'])):
        raise DataPoolError('Merged output is missing or duplicates selected examples')
    index = [by_ordinal[ordinal] for ordinal in sorted(by_ordinal)]
    index_path = directory / 'index.jsonl'
    write_records(index_path, index)
    merged_canonical = None
    if plan['config']['merge_regenerated'] and regenerated:
        path = directory / 'regenerated.jsonl'
        write_records(path, [regenerated[ordinal] for ordinal in sorted(regenerated)])
        merged_canonical = {'path': str(path), 'sha256': pool.digest(path), 'examples': len(regenerated)}
    manifest = {'schema_version': 1, 'status': 'complete', 'command': final_operation, 'pipeline_operation': plan['operation'], 'scope': plan['scope'], 'plan_identity_sha256': plan['plan_identity_sha256'], 'plan_sha256': pool.digest(directory / 'plan.json'), 'source_sha256': plan['source_hashes']['data.py'], 'data_pool_sha256': plan['source_hashes']['data_pool.py'], 'input_sha256': plan['input']['selected_canonical_sha256'], 'target_model_id': plan['config']['target_id'], 'target_model_identity_sha256': plan['target']['model_identity_sha256'], 'asset_receipt_sha256': plan['asset_receipt_sha256'], 'asset_receipt_original_root': plan['asset_receipt_original_root'], 'asset_receipt_is_unchanged_copy_not_new_preparation': True, 'feature_layers': plan['feature_layers'] if final_operation == 'features' else None, 'feature_dim': plan['feature_dim'] if final_operation == 'features' else None, 'target_vocab_size': plan['target_configuration']['vocab_size'], 'alignment': ALIGNMENT if final_operation == 'features' else None, 'include_teacher_logits': plan['config']['include_teacher_logits'] if final_operation == 'features' else False, 'examples': index, 'workers': workers, 'index': {'path': str(index_path), 'sha256': pool.digest(index_path)}, 'merged_regenerated_canonical': merged_canonical, 'actual_exported_tokens': sum(item['tokens'] for item in index) if final_operation == 'features' else None, 'actual_exported_loss_tokens': sum(item['loss_tokens'] for item in index) if final_operation == 'features' else None, 'scientific_results': False, 'native_eagle3_feature_input_compatible': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    pool.atomic_json(directory / 'manifest.json', manifest)
    return manifest


def run_wave(plan, lock_fd):
    directory = Path(plan['run_dir'])
    children, logs = [], []
    previous = {}
    def interrupted(number, frame):
        raise DataPoolError('Data controller interrupted by signal ' + str(number))
    try:
        for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            previous[number] = signal.signal(number, interrupted)
        for assignment in plan['assignments']:
            rank = assignment['shard_index']
            environment = pool.worker_environment(assignment['binding']['token'], directory / 'inputs/source', Path(plan['root']), Path(plan['runtime_python']))
            environment.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', TORCHINDUCTOR_CACHE_DIR=str(directory / 'cache' / str(rank) / 'inductor'), TRITON_CACHE_DIR=str(directory / 'cache' / str(rank) / 'triton'))
            command = [plan['runtime_python'], '-B', '-m', 'taps_rebuild.data_pool', 'worker', '--plan', str(directory / 'plan.json'), '--shard-index', str(rank), '--lock-fd', str(lock_fd)]
            log = (directory / 'logs' / (str(rank) + '.log')).open('xb')
            logs.append(log)
            children.append(subprocess.Popen(command, env=environment, cwd=plan['root'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock_fd,)))
        pending = set(range(len(children)))
        while pending:
            for rank in list(pending):
                result = children[rank].poll()
                if result is None:
                    continue
                if result:
                    raise DataPoolError('Data worker failed: ' + str(rank) + ' exit ' + str(result))
                pending.remove(rank)
            if pending:
                time.sleep(0.1)
    finally:
        try:
            pool.terminate_children(children)
        finally:
            for log in logs:
                log.close()
            for number, handler in previous.items():
                signal.signal(number, handler)


def controller(root, run_id, config):
    root = Path(root).resolve()
    config = options(config)
    if config['scope'] == 'research':
        native_training.research_gate(root, {'scope': 'native_training', 'research_gate': config.get('research_gate', {})})
    hardware = pool.allocation(8)
    if hardware['node'] == 'deep-chungus-10':
        holder = json.loads((root.parent / 'rl-infra/cluster/deep10/current.json').read_bytes())
        if str(holder['job_id']) != hardware['job_id'] or hardware['slurm']['job'].get('JobName') != 'deep10-hold':
            raise DataPoolError('Protected holder identity differs')
    directory = output_directory(root, run_id)
    rows = fixture_rows() if config['scope'] == 'engineering' else list(itertools.islice(data.records(config['input']), config['limit']))
    plan = freeze(root, directory, run_id, config, rows, hardware)
    lock = (directory / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (directory / 'workers').mkdir()
    (directory / 'logs').mkdir()
    pool.atomic_json(directory / 'manifest.json', {'status': 'running', 'operation': plan['operation'], 'scope': plan['scope'], 'plan_identity_sha256': plan['plan_identity_sha256'], 'scientific_results': False})
    try:
        run_wave(plan, lock.fileno())
        return merge(load_plan(directory / 'plan.json'))
    except BaseException as error:
        pool.atomic_json(directory / 'manifest.json', {'status': 'failed', 'operation': plan['operation'], 'scope': plan['scope'], 'plan_identity_sha256': plan['plan_identity_sha256'], 'error_type': type(error).__name__, 'error': str(error), 'partial_outputs_preserved': True, 'scientific_results': False})
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action', required=True)
    run = commands.add_parser('run')
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--run-id', required=True)
    run.add_argument('--operation', choices=['regenerate', 'features', 'regenerate-features'], required=True)
    run.add_argument('--scope', choices=['engineering', 'research'], default='engineering')
    run.add_argument('--input', type=Path)
    run.add_argument('--research-gate', type=Path)
    run.add_argument('--target-id', default='llama31_instruct_8b')
    run.add_argument('--limit', type=int, default=1000000)
    run.add_argument('--seed', type=int, default=20261109)
    run.add_argument('--max-context', type=int, default=2048)
    run.add_argument('--max-new-tokens', type=int, default=32)
    run.add_argument('--temperature', type=float, default=0)
    run.add_argument('--feature-layers')
    run.add_argument('--include-teacher-logits', action='store_true')
    run.add_argument('--merge-regenerated', action='store_true')
    worker_parser = commands.add_parser('worker')
    worker_parser.add_argument('--plan', type=Path, required=True)
    worker_parser.add_argument('--shard-index', type=int, required=True)
    worker_parser.add_argument('--lock-fd', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.plan, args.shard_index, args.lock_fd)
        return
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in ('action', 'root', 'run_id', 'research_gate')}
    if args.research_gate:
        config['research_gate'] = json.loads(args.research_gate.read_bytes())
    value = controller(args.root, args.run_id, config)
    print(json.dumps({'status': value['status'], 'examples': len(value['examples']), 'manifest': str(Path(args.root) / 'data/exports' / args.run_id / 'manifest.json'), 'scientific_results': False}))


if __name__ == '__main__':
    main()
