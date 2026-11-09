import argparse
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from . import sources


TARGET = (0.05, 0.15, 0.30, 0.50)
CASE_ORDERS = ((0,), (0, 1), (0, 0, 1), (3, 2, 1, 0))
METHODS = {
    'eagle3': ('EAGLE', 'eagle.model.ea_model', 'cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b', '2.6.0', '4.53.1'),
    'eagle2': ('EAGLE', 'eagle.model.ea_model', 'cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b', '2.6.0', '4.53.1'),
    'hass': ('HASS', 'model.ea_model', '051c4020141f5d2a365994ca9a8148700044c6d7', '2.1.0', '4.40.0'),
}
ALPHA = 0.05
MARGIN = 0.02
MAX_TRIALS = 1000000


class VerificationError(ValueError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_cpu_slurm():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise VerificationError('Finite-vocabulary verification requires Slurm and empty CUDA_VISIBLE_DEVICES')


def extract_rule_source(contents):
    text = contents.decode('utf-8') if isinstance(contents, bytes) else contents
    matches = [node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_posterior']
    if len(matches) != 1:
        raise VerificationError('Expected one actual evaluate_posterior function in pinned source')
    node = matches[0]
    parameters = [argument.arg for argument in node.args.posonlyargs + node.args.args]
    if parameters != ['logits', 'candidates', 'logits_processor'] or node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
        raise VerificationError('Official posterior signature differs from inspected implementation')
    segment = ast.get_source_segment(text, node)
    return {'text': segment, 'sha256': hashlib.sha256(segment.encode('utf-8')).hexdigest(), 'start_line': node.lineno, 'end_line': node.end_lineno, 'parameters': parameters}


def summarize_counts(counts, target=TARGET, case_count=len(CASE_ORDERS), alpha=ALPHA, margin=MARGIN):
    if len(counts) != len(target) or not target or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
        raise VerificationError('Outcome counts must be nonnegative integers matching the vocabulary')
    if not all(math.isfinite(value) and value >= 0 for value in target) or not math.isclose(sum(target), 1.0, abs_tol=1e-12):
        raise VerificationError('Target probabilities must form a finite distribution')
    trials = sum(counts)
    if trials < 1 or not isinstance(case_count, int) or case_count < 1 or not 0 < alpha < 1 or not 0 < margin < 1:
        raise VerificationError('Invalid trial count or confidence settings')
    empirical = [value / trials for value in counts]
    empirical_tv = sum(abs(value - expected) for value, expected in zip(empirical, target)) / 2
    coordinate_radius = math.sqrt(math.log(2 * len(target) * case_count / alpha) / (2 * trials))
    tv_radius = len(target) * coordinate_radius / 2
    upper = min(1.0, empirical_tv + tv_radius)
    return {'counts': list(counts), 'actual_trials': trials, 'empirical_probabilities': empirical, 'empirical_tv': empirical_tv, 'coordinate_hoeffding_radius': coordinate_radius, 'tv_uncertainty_radius': tv_radius, 'simultaneous_tv_upper_bound': upper, 'equivalence_margin': margin, 'equivalence_criterion_passed': upper <= margin}


def candidate_paths(order):
    if not order or any(not isinstance(token, int) or token < 0 or token >= len(TARGET) for token in order):
        raise VerificationError('Candidate children must belong to the finite vocabulary')
    return [[0, token] for token in order]


def integer(value):
    return int(value.item() if hasattr(value, 'item') else value)


def interpret_outcome(result, paths, fallback):
    if not isinstance(result, tuple) or len(result) != 3:
        raise VerificationError('Official posterior returned an unexpected structure')
    best, accepted, sample_p = result
    best = integer(best)
    accepted = integer(accepted)
    if not 0 <= best < len(paths):
        raise VerificationError('Official posterior returned an invalid candidate index')
    if accepted == 1:
        return paths[best][1], True
    if accepted == 0:
        token = integer(fallback(sample_p))
        if not 0 <= token < len(TARGET):
            raise VerificationError('Official residual sampling returned an invalid token')
        return token, False
    raise VerificationError('Official posterior acceptance exceeds the two-token path audit')


def identity_logits_processor(input_ids, scores):
    return scores


def validate_source(root, method):
    name, module_name, revision, _, _ = METHODS[method]
    receipt_path = root / 'outputs/preparation/sources.json'
    receipt_contents = receipt_path.read_bytes()
    receipt = json.loads(receipt_contents)
    registry_path = root / 'configs/sources.json'
    registry_contents = registry_path.read_bytes()
    registry = json.loads(registry_contents)
    selected, _ = sources.validate_registry(registry)
    source = next(item for item in selected if item['name'] == name)
    record = receipt['sources'][name]
    if receipt.get('status') != 'complete' or Path(receipt['root']).resolve() != root:
        raise VerificationError('A completed source preparation receipt for this project is required')
    if receipt['registry_sha256'] != hashlib.sha256(registry_contents).hexdigest() or receipt['source_sha256'] != file_sha256(root / 'src/taps_rebuild/sources.py'):
        raise VerificationError('Source preparation registry or builder hash differs')
    directory = root / source['path']
    if source['revision'] != revision or record['revision'] != revision or Path(record['path']).resolve() != directory.resolve():
        raise VerificationError('Source receipt differs from the pinned official method')
    sources.inspect_checkout(directory, source)
    tracked, identity = sources.tracked_manifest(directory)
    if tracked != record['tracked_files'] or identity != record['source_identity_sha256']:
        raise VerificationError('Actual source file hashes differ from the preparation receipt')
    module_file = directory / Path(*module_name.split('.')).with_suffix('.py')
    rule_file = module_file.with_name('utils.py')
    if module_file.is_symlink() or rule_file.is_symlink():
        raise VerificationError('Posterior source files must be regular tracked files')
    for path in (module_file, rule_file):
        relative = str(path.relative_to(directory))
        if relative not in tracked or tracked[relative]['kind'] != 'regular_file' or tracked[relative]['sha256'] != file_sha256(path):
            raise VerificationError('Official posterior source identity differs')
    rule = extract_rule_source(rule_file.read_bytes())
    return {'name': name, 'revision': revision, 'directory': directory, 'module_name': module_name, 'module_file': module_file, 'rule_file': rule_file, 'rule': rule, 'source_identity_sha256': identity, 'sources_receipt_sha256': hashlib.sha256(receipt_contents).hexdigest(), 'source_registry_sha256': hashlib.sha256(registry_contents).hexdigest()}


def load_actual_posterior(evidence, method):
    _, _, _, expected_torch, expected_transformers = METHODS[method]
    versions = {package: importlib.metadata.version(package) for package in ('torch', 'transformers')}
    if versions['torch'].split('+')[0] != expected_torch or versions['transformers'] != expected_transformers:
        raise VerificationError('Use the preserved taps-hass or taps-eagle Python profile appropriate to the method')
    os.environ.update({'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'})
    sys.dont_write_bytecode = True
    torch = importlib.import_module('torch')
    torch.set_num_threads(1)
    previous = sys.path[:]
    try:
        sys.path.insert(0, str(evidence['directory']))
        module = importlib.import_module(evidence['module_name'])
    finally:
        sys.path[:] = previous
    if Path(module.__file__).resolve() != evidence['module_file'].resolve():
        raise VerificationError('An official module from another checkout is already imported')
    function = module.evaluate_posterior
    if Path(inspect.getsourcefile(function)).resolve() != evidence['rule_file'].resolve() or tuple(inspect.signature(function).parameters) != ('logits', 'candidates', 'logits_processor'):
        raise VerificationError('Imported posterior is not the inspected official function')
    return torch, function, versions


def run_case(torch, posterior, order, trials, seed):
    paths = candidate_paths(order)
    candidates = torch.tensor(paths, dtype=torch.long, device='cpu')
    target = torch.tensor(TARGET, dtype=torch.float64, device='cpu')
    logits = target.log().reshape(1, 1, len(TARGET)).repeat(len(paths), 2, 1)
    random.seed(seed)
    torch.manual_seed(seed)
    counts = [0] * len(TARGET)
    accepted_counts = [0] * len(TARGET)
    fallback_counts = [0] * len(TARGET)

    def fallback(sample_p):
        if sample_p.device.type != 'cpu' or tuple(sample_p.shape) != (len(TARGET),) or not bool(torch.isfinite(sample_p).all()) or bool((sample_p < 0).any()) or not math.isclose(float(sample_p.sum()), 1.0, abs_tol=1e-9):
            raise VerificationError('Official residual probabilities are not a finite CPU distribution')
        return torch.multinomial(sample_p, 1).item()

    started = time.monotonic()
    with torch.inference_mode():
        for _ in range(trials):
            result = posterior(logits, candidates, identity_logits_processor)
            token, accepted = interpret_outcome(result, paths, fallback)
            counts[token] += 1
            (accepted_counts if accepted else fallback_counts)[token] += 1
    return dict(summarize_counts(counts), child_order=list(order), candidate_paths=paths, seed=seed, accepted_child_counts=accepted_counts, fallback_counts=fallback_counts, elapsed_seconds=time.monotonic() - started)


def run_audit(root, method, trials=100000, seed=20261109):
    require_cpu_slurm()
    if method not in METHODS or not isinstance(trials, int) or not 1 <= trials <= MAX_TRIALS or not isinstance(seed, int) or not 0 <= seed < 2 ** 63 - len(CASE_ORDERS):
        raise VerificationError('Invalid method, seed or bounded trial count')
    root = Path(root).resolve()
    output = root / 'outputs/verification' / (method + '-finite-vocab.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'preparing', 'method': method, 'root': str(root), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': file_sha256(Path(__file__)), 'requested_trials_per_case': trials, 'seed': seed, 'target_probabilities': list(TARGET), 'candidate_root_token': 0, 'candidate_path_length': 2, 'temperature': 1.0, 'logits_processor': 'non-None identity callable', 'input_logit_dtype': 'torch.float64', 'device': 'cpu', 'models_loaded': False, 'gpu_execution_performed': False, 'confidence': 1 - ALPHA, 'confidence_family': 'four candidate cases and four token outcomes for the selected method', 'bound': 'TV(q,p) <= empirical_TV(q,p_hat) + (vocab_size/2)*sqrt(log(2*vocab_size*case_count/alpha)/(2*actual_trials))', 'alpha': ALPHA, 'equivalence_margin': MARGIN, 'scope': 'Finite-vocabulary conditional posterior rule only', 'full_model_output_distribution_proof': False, 'automatic_pipeline_gate_pass': False, 'full_model_temperature1_validation_still_required': True, 'cases': []}
    with output.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise VerificationError('Another audit holds the method lock') from None
        sources.atomic_json(output, receipt)
        try:
            evidence = validate_source(root, method)
            torch, posterior, versions = load_actual_posterior(evidence, method)
            receipt.update(packages=versions, python=sys.version, source_revision=evidence['revision'], source_directory=str(evidence['directory']), source_identity_sha256=evidence['source_identity_sha256'], sources_receipt_sha256=evidence['sources_receipt_sha256'], source_registry_sha256=evidence['source_registry_sha256'], posterior_module=evidence['module_name'], posterior_source_file=str(evidence['rule_file']), posterior_source_file_sha256=file_sha256(evidence['rule_file']), posterior_function_sha256=evidence['rule']['sha256'], posterior_function_lines=[evidence['rule']['start_line'], evidence['rule']['end_line']], posterior_parameters=evidence['rule']['parameters'])
            receipt['status'] = 'sampling'
            sources.atomic_json(output, receipt)
            for index, order in enumerate(CASE_ORDERS):
                receipt['cases'].append(run_case(torch, posterior, order, trials, seed + index))
                sources.atomic_json(output, receipt)
            after = validate_source(root, method)
            if after['source_identity_sha256'] != evidence['source_identity_sha256']:
                raise VerificationError('Official sources changed during verification')
            receipt.update(status='complete', equivalence_criterion_passed=all(case['equivalence_criterion_passed'] for case in receipt['cases']), actual_total_trials=sum(case['actual_trials'] for case in receipt['cases']), source_identity_verified_before_after=True, completed_at_utc=datetime.now(timezone.utc).isoformat())
            sources.atomic_json(output, receipt)
        except Exception as error:
            receipt.update(status='failed', error_type=type(error).__name__, failed_at_utc=datetime.now(timezone.utc).isoformat())
            if isinstance(error, (VerificationError, sources.SourcePreparationError)):
                receipt['error'] = str(error)
            sources.atomic_json(output, receipt)
            raise
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=sources.DEFAULT_ROOT)
    parser.add_argument('--method', choices=tuple(METHODS), required=True)
    parser.add_argument('--trials', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=20261109)
    args = parser.parse_args()
    try:
        receipt = run_audit(args.root, args.method, args.trials, args.seed)
    except Exception as error:
        result = {'status': 'failed', 'error_type': type(error).__name__}
        if isinstance(error, (VerificationError, sources.SourcePreparationError)):
            result['error'] = str(error)
        print(json.dumps(result), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({'status': receipt['status'], 'equivalence_criterion_passed': receipt['equivalence_criterion_passed'], 'automatic_pipeline_gate_pass': False, 'receipt': str(Path(receipt['root']) / 'outputs/verification' / (args.method + '-finite-vocab.json'))}), flush=True)
    if not receipt['equivalence_criterion_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
