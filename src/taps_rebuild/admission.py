import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path


DRIVER_SHA256 = 'd6ea2b5216b0052aa68282646b343143c6523ed14e75f8b2b838f99bb6458934'
PREPARED_SHA256 = 'd3f236a73ab18940bd28f76112210449a2f6ef797061d50a1ab98fc059e0a865'
METHODS = ('eagle3', 'eagle2', 'hass')


class AdmissionError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AdmissionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def stamp(path):
    stat = Path(path).stat()
    return {'inode': stat.st_ino, 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def descriptor(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Admission artifact escaped the project')
    before = stamp(path)
    value = {'path': str(path), 'sha256': digest(path), 'stat': before}
    require(stamp(path) == before, 'Admission artifact changed during hashing')
    return value


def verify_certificate(root, report):
    require(report.get('format') == 'source-bound-full-model-audit-v1' and report.get('status') == 'passed' and report.get('automatic_pipeline_gate_pass') is True, 'A raw-bound validator certificate is required; claimed passed metadata cannot authorize training')
    require(report.get('validator_sha256') == digest(__file__) and report.get('driver_sha256') == DRIVER_SHA256 and report.get('prepared_sha256') == PREPARED_SHA256, 'Admission validator or registered inputs changed')
    unsigned = {key: value for key, value in report.items() if key != 'identity_sha256'}
    require(identity(unsigned) == report.get('identity_sha256'), 'Admission certificate identity differs')
    required_checks = {'greedy', 'conditional', 'sampling', 'complete_registered_phases', 'independent_validator_review'}
    require(report.get('review', {}).get('decision') == 'approved' and set(report.get('checks', {})) == required_checks and all(value is True for value in report['checks'].values()), 'Independent validator review and every registered check must pass')
    artifacts = report.get('artifacts', [])
    require(len(artifacts) >= 150 and len({item['path'] for item in artifacts}) == len(artifacts), 'Incomplete full-model audit provenance')
    for item in artifacts:
        path = Path(item['path']).resolve()
        require(path.is_relative_to(Path(root).resolve()) and stamp(path) == item['stat'], 'An admission artifact changed after CPU verification')
    return {'status': 'passed', 'certificate_identity_sha256': report['identity_sha256'], 'scope': report['scope'], 'scientific_training_authorized': True}


def conditional_checks(row, capture, driver, stops):
    import torch
    q, weights, masks = capture['normalized_ar_q'], capture['native_ar_weights'], capture['event_masks']
    require(q.ndim == 1 and len(q) >= 32000 and masks.shape == (8, len(q)) and masks.dtype == torch.bool, 'Conditional tensor dimensions differ')
    require(bool(torch.isfinite(q).all()) and bool((q >= 0).all()) and abs(float(q.sum()) - 1) < 1e-10, 'Invalid conditional target law')
    require(torch.allclose(q, weights.double() / weights.double().sum(), rtol=5e-14, atol=1e-18), 'Conditional reference does not normalize the independent AR weights')
    require(row['trials_per_sampler'] == 65536 and set(row['histograms']) == {'posterior', 'target_only'}, 'Conditional sample budget differs')
    expected_masks = torch.zeros_like(masks)
    order = torch.argsort(q, descending=True, stable=True)
    for index, count in enumerate((1, 8, 32, 128)):
        expected_masks[index, order[:count]] = True
    expected_masks[4, sorted(stops)] = True
    candidates = capture['candidates']
    require(candidates.ndim == 2 and candidates.shape[1] >= 2 and bool((candidates[:, 0] == candidates[0, 0]).all()), 'Invalid fixed candidate tree')
    children = sorted({int(token) for token in candidates[:, 1] if 0 <= int(token) < len(q)})
    expected_masks[5, children] = True
    for salt in (0, 1):
        expected_masks[6 + salt] = torch.tensor([driver.hash_partition(token, salt) for token in range(len(q))], dtype=torch.bool)
    require(torch.equal(masks, expected_masks), 'Conditional events differ from their registered definitions')
    require(driver.identity(capture['prefix'][0].tolist()) == row['ar_prefix_sha256'], 'Conditional prefix identity differs')
    probabilities = (masks.double() * q).sum(dim=1).tolist()
    passed = True
    for sampler, values in row['histograms'].items():
        require(len(values) == len(q) and all(type(value) is int and value >= 0 for value in values) and sum(values) == 65536, 'Conditional histogram is incomplete')
        require(len(row['checks'][sampler]) == 8, 'Conditional event checks are incomplete')
        counts = (masks.long() * torch.tensor(values)).sum(dim=1).tolist()
        for count, probability, reported in zip(counts, probabilities, row['checks'][sampler]):
            bound = abs(count / 65536 - probability) + driver.conditional_bound(65536)
            require(reported['count'] == count and abs(reported['expected'] - probability) < 1e-12 and abs(reported['simultaneous_error_upper_bound'] - bound) < 1e-12 and reported['passed'] == (bound <= 0.02), 'Conditional summary is inconsistent with its raw samples')
            passed &= bound <= 0.02
    return passed


def review_document(path, proof_sha256, protocol_sha256):
    if path is None or not Path(path).exists():
        return {'decision': 'pending', 'reason': 'The controlling temperature-one proposal requires independent review before native admission'}
    review = read(path)
    require(review.get('decision') == 'approved' and isinstance(review.get('reviewer'), str) and review['reviewer'].strip(), 'Independent review is not approved')
    require(review.get('validator_sha256') == digest(__file__) and review.get('driver_sha256') == DRIVER_SHA256 and review.get('proof_sha256') == proof_sha256 and review.get('protocol_sha256') == protocol_sha256, 'Review does not cover this validator, proof and registration')
    return review


def audit(root, driver_path, prepared_path, primary, screen, proof, review_path=None, phases=('greedy', 'conditional', 'sampling')):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Full-model admission audit requires CPU Slurm')
    root, primary, screen = Path(root).resolve(), Path(primary), Path(screen)
    require(digest(driver_path) == DRIVER_SHA256 and digest(prepared_path) == PREPARED_SHA256, 'Full-model driver or prepared panel differs')
    prepared = read(prepared_path)
    require(prepared['driver_sha256'] == DRIVER_SHA256, 'Prepared panel was created by another driver')
    spec = importlib.util.spec_from_file_location('registered_validation_driver', driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    artifacts = {}
    def bind(path):
        value = descriptor(root, path)
        artifacts[value['path']] = value
    bind(driver_path)
    bind(prepared_path)
    bind(proof)
    checks = {}
    from transformers import AutoTokenizer
    tokenizers = {method: AutoTokenizer.from_pretrained(prepared['methods'][method]['models']['target']['path'], local_files_only=True) for method in METHODS}
    vocabulary_sizes = {method: len(tokenizer) for method, tokenizer in tokenizers.items()}
    stops = {}
    for method, tokenizer in tokenizers.items():
        eos = tokenizer.eos_token_id
        values = list(eos) if isinstance(eos, (tuple, list)) else [eos]
        stops[method] = {value for value in [*values, tokenizer.convert_tokens_to_ids('<|eot_id|>')] if isinstance(value, int) and value >= 0}
    require(phases and len(phases) == len(set(phases)) and set(phases) <= {'greedy', 'conditional', 'sampling'}, 'Unknown validation phase selection')
    for phase in phases:
        directory = screen if phase == 'sampling' else primary
        manifest = read(directory / 'manifest.json')
        require(manifest['prepared_sha256'] == PREPARED_SHA256 and manifest['driver_sha256'] == DRIVER_SHA256 and len(manifest['hardware']['tokens']) == 8, 'Audit manifest differs from the full A100 registration')
        require(manifest['conditional_trials'] == 65536 and manifest['sampling_repeats'] == 1024, 'Engineering smoke budgets cannot authorize training')
        require(read(directory / 'completion.json')['status'] == 'complete', 'Validation pipeline is incomplete')
        bind(directory / 'manifest.json')
        bind(directory / 'completion.json')
        seen, passed, totals = set(), True, {}
        paths = sorted(directory.glob(phase + '-*.jsonl'))
        require(len(paths) == 24, 'Missing validation workers')
        for path in paths:
            receipt_path = path.with_suffix('.receipt.json')
            receipt = read(receipt_path)
            require(receipt['status'] == 'complete' and receipt['raw_sha256'] == digest(path) and receipt['prepared_sha256'] == PREPARED_SHA256, 'Worker completion proof changed')
            bind(path)
            bind(receipt_path)
            count, failures = 0, 0
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    method = row['method']
                    require(method in METHODS and row['phase'] == phase and row['prepared_sha256'] == PREPARED_SHA256 and row['provenance'] == prepared['methods'][method]['provenance'], 'Validation row provenance differs')
                    count += 1
                    if phase == 'greedy':
                        key = (method, driver.identity(row['case']))
                        require(row['case'] in prepared['methods'][method]['greedy_cases'] and row['precision'] == 'float32', 'Greedy panel or precision differs')
                        equal = row['outputs']['target_only']['comparison_token_ids'] == row['outputs']['speculative']['comparison_token_ids']
                        require(row['equal'] == equal, 'Greedy comparison disagrees with raw tokens')
                        failures += int(not equal)
                        passed &= equal
                    elif phase == 'conditional':
                        key = (method, row['prompt_index'], row['cycle'])
                        require(0 <= row['prompt_index'] < 10 and row['cycle'] in (1, 3) and row['prompt'] == prepared['methods'][method]['sampling_prompts'][row['prompt_index']], 'Conditional capture was substituted')
                        capture_path = directory / row['capture']
                        require(capture_path.resolve().parent == directory.resolve() and digest(capture_path) == row['capture_sha256'], 'Conditional capture changed')
                        bind(capture_path)
                        import torch
                        success = conditional_checks(row, torch.load(capture_path, map_location='cpu', weights_only=True), driver, stops[method])
                        failures += int(not success)
                        passed &= success
                    else:
                        key = (method, row['prompt_index'], row['repeat'], row['mode'])
                        require(0 <= row['prompt_index'] < 10 and 0 <= row['repeat'] < 1024 and row['mode'] in ('target_only', 'speculative'), 'Unregistered end-to-end sample')
                        require(row['seed'] == driver.sampling_seed(method, row['mode'], row['prompt_index'], row['repeat']) and row['prompt'] == prepared['methods'][method]['sampling_prompts'][row['prompt_index']], 'End-to-end prompt or seed differs')
                        tokens = row['comparison_token_ids']
                        raw = row['output_token_ids']
                        first_stop = next((index for index, token in enumerate(raw) if token in stops[method]), None)
                        require(tokens == raw[:min(32, first_stop + 1 if first_stop is not None else len(raw))] and all(type(token) is int and 0 <= token < vocabulary_sizes[method] for token in raw), 'End-to-end token or termination rule differs')
                        events = driver.continuation_events(tokens, stops[method])
                        require(events == row['events'] and driver.identity(row['input_token_ids']) == row['input_sha256'], 'End-to-end events or input identity differ')
                        values = totals.setdefault((method, row['mode']), [0] * 12)
                        for index, value in enumerate(events):
                            values[index] += value
                    require(key not in seen, 'Duplicate validation coverage')
                    seen.add(key)
            require(receipt['completed'] == count and receipt['failed_checks'] == failures, 'Worker count or failure receipt differs')
        if phase == 'greedy':
            expected = {(method, driver.identity(case)) for method in METHODS for case in prepared['methods'][method]['greedy_cases']}
        elif phase == 'conditional':
            expected = {(method, index, cycle) for method in METHODS for index in range(10) for cycle in (1, 3)}
        else:
            expected = {(method, index, repeat, mode) for method in METHODS for index in range(10) for repeat in range(1024) for mode in ('target_only', 'speculative')}
            passed = all(abs(a - b) / 10240 + driver.sampling_bound(10240) <= 0.06 for method in METHODS for a, b in zip(totals[(method, 'target_only')], totals[(method, 'speculative')]))
        require(seen == expected, 'Incomplete or substituted validation population')
        checks[phase] = bool(passed)
    review = review_document(review_path, digest(proof), prepared['protocol_sha256'])
    if review_path is not None and Path(review_path).exists():
        bind(review_path)
    checks['independent_validator_review'] = review['decision'] == 'approved'
    checks['complete_registered_phases'] = set(phases) == {'greedy', 'conditional', 'sampling'}
    passed = all(checks.values())
    report = {'format': 'source-bound-full-model-audit-v1', 'status': 'passed' if passed else 'blocked', 'scope': 'registered checks listed in phases_evaluated only; not universal losslessness', 'phases_evaluated': list(phases), 'models_loaded': True, 'automatic_pipeline_gate_pass': passed, 'full_model_temperature1_validation_still_required': not passed, 'validator_sha256': digest(__file__), 'driver_sha256': DRIVER_SHA256, 'prepared_sha256': PREPARED_SHA256, 'checks': checks, 'review': review, 'artifacts': sorted(artifacts.values(), key=lambda value: value['path']), 'hardware_scopes': 'Preserve each raw manifest; no cross-node timing comparison is authorized'}
    report['identity_sha256'] = identity(report)
    return report
