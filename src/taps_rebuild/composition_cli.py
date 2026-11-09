import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from . import composition


OPERATIONS = ('fit-calibrator', 'evaluate-calibrator', 'pool-tree', 'merge-weights')


class CompositionCLIError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def resolve_input(root, path):
    path = Path(path)
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def read_input(root, descriptor, inventory, name, scope):
    if not isinstance(descriptor, dict) or ('path' in descriptor) == ('inline' in descriptor):
        raise CompositionCLIError(name + ' requires exactly one path or inline payload')
    if 'inline' in descriptor:
        value = copy.deepcopy(descriptor['inline'])
        checksum = hashlib.sha256(canonical(value)).hexdigest()
        record = {'kind': 'inline_canonical_json', 'sha256': checksum}
    else:
        path = resolve_input(root, descriptor['path'])
        checksum = digest(path)
        if scope == 'research' and 'sha256' not in descriptor:
            raise CompositionCLIError('Research input file requires an explicit SHA256: ' + name)
        expected = descriptor.get('sha256', checksum)
        if not isinstance(expected, str) or not re.fullmatch('[a-f0-9]{64}', expected) or expected != checksum:
            raise CompositionCLIError('Input SHA256 differs: ' + name)
        if path.suffix == '.pt':
            if not os.environ.get('SLURM_JOB_ID'):
                raise CompositionCLIError('Tensor weight loading and merging requires Slurm; no GPU is used by this CPU loader')
            try:
                import torch
            except ImportError as error:
                raise CompositionCLIError('Use an existing Torch environment; no installation is performed') from error
            value = torch.load(path, map_location='cpu', weights_only=True)
        elif path.suffix == '.jsonl':
            value = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
        else:
            value = json.loads(path.read_bytes())
        if digest(path) != checksum:
            raise CompositionCLIError('Input changed while being loaded: ' + name)
        record = {'kind': 'file', 'path': str(path), 'sha256': checksum, 'bytes': path.stat().st_size}
    if 'sha256' in descriptor and descriptor['sha256'] != checksum:
        raise CompositionCLIError('Input SHA256 differs: ' + name)
    inventory[name] = record
    return value, checksum


def _examples(value):
    value = value.get('examples') if isinstance(value, dict) else value
    if not isinstance(value, list) or not value:
        raise CompositionCLIError('Calibration requires a nonempty list of verification examples')
    return value


def _calibrator(value):
    value = value.get('calibrator', value) if isinstance(value, dict) else value
    if not isinstance(value, dict):
        raise CompositionCLIError('A saved fitted calibrator metadata object is required')
    return composition.Calibrator.from_metadata(value)


def _scope(root, config, inventory):
    scope = config.get('scope', 'engineering')
    if scope not in ('engineering', 'research'):
        raise CompositionCLIError('scope must be engineering or research')
    if scope == 'engineering':
        return scope, {'status': 'engineering_only', 'scientific_experiment_authorized': False}
    if not os.environ.get('SLURM_JOB_ID'):
        raise CompositionCLIError('Research composition requires Slurm')
    if 'reproduction_report' not in config:
        raise CompositionCLIError('Research composition requires a passed official reproduction report')
    report, checksum = read_input(root, config['reproduction_report'], inventory, 'reproduction_report', scope)
    gate = report.get('reproduction_gate', {})
    cells = report.get('cells', [])
    referenced = [cell for cell in cells if cell.get('reference') is not None]
    if gate.get('passed') is not True or gate.get('status') != 'passed' or not referenced or any(cell.get('reproduction_gate', {}).get('passed') is not True for cell in referenced) or {cell.get('method') for cell in referenced} != {'eagle3', 'eagle2', 'hass'}:
        raise CompositionCLIError('All three official methods must have passed referenced reproduction cells')
    registry_path = resolve_input(root, 'configs/reproduction.json')
    registry, _ = read_input(root, {'path': str(registry_path), 'sha256': digest(registry_path)}, inventory, 'reproduction_config', scope)
    expected = registry.get('reference_cells', [])
    key = lambda cell: (cell['method'], cell['benchmark'], cell['temperature'])
    expected_cells = {key(cell): cell for cell in expected}
    actual_cells = {key(cell): cell for cell in referenced}
    if not expected or len(expected_cells) != len(expected) or len(actual_cells) != len(referenced) or set(actual_cells) != set(expected_cells) or gate.get('reference_cells') != len(expected) or gate.get('passed_reference_cells') != len(expected):
        raise CompositionCLIError('The complete configured reference-cell matrix must pass before research composition')
    for cell_key, registered in expected_cells.items():
        published = actual_cells[cell_key]['reference']
        if any(published.get(name) != registered.get(name) for name in ('published_tau', 'source_url', 'expected_questions', 'expected_turns', 'timing_repeats')):
            raise CompositionCLIError('Reproduction reference definition differs from the configured published cell')
    return scope, {'status': 'passed_numerical_reproduction', 'sha256': checksum, 'losslessness_established': False, 'qualification': gate.get('qualification'), 'temperature_one_distributional_correctness': 'requires_separate_statistical_check'}


def _state(value, descriptor):
    if not isinstance(value, dict) or not value:
        raise CompositionCLIError('A nonempty state dictionary or checkpoint wrapper is required')
    key = descriptor.get('state_key')
    if key is not None:
        if key not in value:
            raise CompositionCLIError('Configured state_key is absent from checkpoint')
        state = value[key]
    elif isinstance(value.get('state_dict'), dict):
        state = value['state_dict']
    elif isinstance(value.get('model'), dict):
        state = value['model']
    else:
        state = value
    if not isinstance(state, dict) or not state or any(not isinstance(name, str) or not name for name in state):
        raise CompositionCLIError('State dictionary requires identified parameter keys')
    return state, value.get('metadata', {}) if state is not value else {}


def execute(operation, value, root, *, config_sha256=None):
    if operation not in OPERATIONS:
        raise CompositionCLIError('Unsupported composition operation')
    config = copy.deepcopy(value)
    if config.get('operation', operation) != operation:
        raise CompositionCLIError('Configuration operation differs from selected command')
    root = Path(root).resolve()
    inventory = {}
    scope, gate = _scope(root, config, inventory)
    if operation == 'fit-calibrator':
        raw, source_identity = read_input(root, config['examples'], inventory, 'examples', scope)
        expected = config.get('dataset_identity', source_identity)
        if expected != source_identity:
            raise CompositionCLIError('Calibration dataset_identity differs from the actual input SHA256')
        fitted = composition.fit_calibrator(config['expert_id'], _examples(raw), method=config.get('method', 'temperature'), split=config.get('split', 'dev'), dataset_identity=expected, forbidden_ids=config.get('forbidden_ids', []))
        payload = {'calibrator': fitted.metadata()}
    elif operation == 'evaluate-calibrator':
        raw, source_identity = read_input(root, config['examples'], inventory, 'examples', scope)
        saved, _ = read_input(root, config['calibrator'], inventory, 'calibrator', scope)
        payload = {'diagnostics': composition.evaluate_calibrator(_calibrator(saved), _examples(raw), split=config['split'], dataset_identity=source_identity)}
    elif operation == 'pool-tree':
        drafts, _ = read_input(root, config['drafts'], inventory, 'drafts', scope)
        if isinstance(drafts, dict):
            drafts = drafts.get('drafts')
        if not isinstance(drafts, list):
            raise CompositionCLIError('Pooling input must contain a list of draft candidate trees')
        calibrators = {}
        for expert, descriptor in config.get('calibrators', {}).items():
            saved, _ = read_input(root, descriptor, inventory, 'calibrator:' + expert, scope)
            calibrators[expert] = _calibrator(saved)
        score_mode = config.get('score_mode', 'estimated')
        if scope != 'engineering' and score_mode != 'estimated':
            raise CompositionCLIError('Oracle and uncalibrated pooling are engineering diagnostics only')
        payload = {'tree': composition.pooled_budget_tree(drafts, root_token=config['root_token'], root_source=config['root_source'], budget=config['budget'], calibrators=calibrators, score_mode=score_mode)}
    else:
        ancestor_value, ancestor_identity = read_input(root, config['ancestor'], inventory, 'ancestor', scope)
        ancestor, _ = _state(ancestor_value, config['ancestor'])
        experts, metadata = {}, {}
        if not isinstance(config.get('experts'), dict) or not config['experts']:
            raise CompositionCLIError('Weight experts must be a nonempty expert-ID mapping')
        declared_ancestors = config.get('expert_ancestors', {})
        for expert, descriptor in config['experts'].items():
            saved, _ = read_input(root, descriptor, inventory, 'expert:' + expert, scope)
            experts[expert], metadata[expert] = _state(saved, descriptor)
            recorded = metadata[expert].get('shared_ancestor_identity_sha256', metadata[expert].get('ancestor_identity'))
            if recorded is not None and recorded != declared_ancestors.get(expert):
                raise CompositionCLIError('Expert checkpoint ancestor metadata differs from its declared identity')
            if scope == 'research' and recorded is None:
                raise CompositionCLIError('Research expert checkpoints must record their shared ancestor identity')
        diagnostic = config.get('diagnostic_independent', False)
        if diagnostic and scope != 'engineering':
            raise CompositionCLIError('Independent initialization merges are engineering diagnostics only')
        if config.get('ancestor_identity', ancestor_identity) != ancestor_identity:
            raise CompositionCLIError('Ancestor identity differs from the actual ancestor input SHA256')
        merged = composition.merge_weights(ancestor, experts, ancestor_identity=ancestor_identity, expert_ancestors=declared_ancestors, method=config.get('method', 'linear'), weights=config.get('weights'), scale=config.get('scale', 1.0), density=config.get('density', 0.2), seed=config.get('seed', 20261109), diagnostic_independent=diagnostic)
        payload = {'state_dict': merged['state_dict'], 'merge_metadata': merged['metadata']}
    return {'schema_version': 1, 'status': 'complete', 'operation': operation, 'scope': scope, 'scientific_result': False, 'qualification': 'Composition artifact only; no measured acceptance, wall-clock speedup or distributional-correctness claim is produced.', 'reproduction_gate': gate, 'provenance': {'config_sha256': config_sha256 or hashlib.sha256(canonical(config)).hexdigest(), 'inputs': inventory, 'composition_cli_sha256': digest(__file__), 'composition_sha256': digest(composition.__file__), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}, **payload}


def output_path(root, value, result):
    root = Path(root).resolve()
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if path.exists() or path.is_symlink():
        raise CompositionCLIError('Refusing to overwrite an existing output or input artifact')
    path = path.resolve()
    allowed = (root / 'models').resolve() / 'composed' if result['operation'] == 'merge-weights' else root / 'outputs/composition'
    if not path.is_relative_to(allowed) or path == allowed:
        raise CompositionCLIError('Output must remain under ' + str(allowed))
    if path.suffix not in ('.json', '.pt') or (path.suffix == '.pt' and result['operation'] != 'merge-weights'):
        raise CompositionCLIError('Use JSON output, or .pt for tensor weight merges')
    inputs = result['provenance']['inputs'].values()
    if any(item.get('path') == str(path) for item in inputs):
        raise CompositionCLIError('An input artifact cannot be used as an output')
    return path


def _commit(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.' + path.name + '-', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_result(root, path, result):
    path = output_path(root, path, result)
    state = result.get('state_dict', {})
    tensors = any(hasattr(item, 'detach') for item in state.values())
    if tensors and path.suffix != '.pt':
        raise CompositionCLIError('Tensor states require .pt output; tensor data is never silently converted to JSON')
    if path.suffix == '.pt':
        if not os.environ.get('SLURM_JOB_ID'):
            raise CompositionCLIError('Tensor checkpoint writing requires Slurm')
        import torch
        receipt_path = path.with_name(path.name + '.json')
        if receipt_path.exists() or receipt_path.is_symlink():
            raise CompositionCLIError('Refusing to overwrite an existing checkpoint receipt')
        _commit(path, lambda stream: torch.save({'state_dict': result['state_dict'], 'metadata': {key: value for key, value in result.items() if key != 'state_dict'}}, stream))
        receipt = {key: value for key, value in result.items() if key != 'state_dict'}
        receipt.update(checkpoint=str(path), checkpoint_sha256=digest(path), checkpoint_bytes=path.stat().st_size)
        _commit(receipt_path, lambda stream: stream.write(canonical(receipt) + b'\n'))
        return receipt
    _commit(path, lambda stream: stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n'))
    return {'status': 'complete', 'output': str(path), 'sha256': digest(path), 'operation': result['operation'], 'scope': result['scope']}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=OPERATIONS)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        path = resolve_input(root, args.config)
        contents = path.read_bytes()
        result = execute(args.operation, json.loads(contents), root, config_sha256=hashlib.sha256(contents).hexdigest())
        receipt = write_result(root, args.output, result)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({'status': 'failed', 'operation': args.operation, 'error_type': type(error).__name__, 'error': str(error)}), flush=True)
        return 1
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
