import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import subprocess


UTILITY_SHA256 = '87842c151e8bbbcd40b8e333ceb87106dad9add4fdf677e29eb5d87f6c65502b'
DRIVER_SHA256 = 'd6ea2b5216b0052aa68282646b343143c6523ed14e75f8b2b838f99bb6458934'
REGISTERED_SHA256 = 'd3f236a73ab18940bd28f76112210449a2f6ef797061d50a1ab98fc059e0a865'
UPSTREAM_SHA256 = '6daaf6f8010ff986eadfd5a5da7546b7ea9107c4ecdcfed57d694335dc78aef3'
REFERENCES = {'alpaca': 5.56, 'gsm8k': 4.59}
ARMS = ('upstream_seed_zero', 'registered_hash')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load(path, name, expected):
    require(digest(path) == expected, 'Dependency source identity differs: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def actual_seed(arm, case, repeat):
    require(arm in ARMS and type(repeat) is int and repeat in range(5), 'Unknown seed policy or repeat')
    if arm == ARMS[0]:
        return 0
    encoded = json.dumps([20261110, case['benchmark'], case['question_id'], 0, repeat], sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], 'big')


def prepare(args):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preparation requires CPU Slurm')
    registered = args.root / 'outputs/validation-inputs/registered-20261110-v2/cases.json'
    validation_driver = args.root / 'outputs/preparation/validation-queue-20261110-v2/driver.py'
    require(digest(registered) == REGISTERED_SHA256 and digest(validation_driver) == DRIVER_SHA256, 'Original validation registration changed')
    selected = read(registered)['methods']['eagle3']
    config = selected['config']
    require((config['total_token'], config['top_k'], config['max_new_tokens'], config['max_length'], selected['definition']['depth']) == (60, 10, 512, 2048, 7), 'Native settings differ from the seed diagnostic registration')
    require(digest(Path(selected['source']['path']) / 'eagle/evaluation/gen_ea_answer_llama3chat.py') == UPSTREAM_SHA256, 'Inspected upstream generator changed')
    from taps_rebuild.runner import verify_provenance

    verify_provenance(args.root, selected['run_config'], selected['run_manifest'], 'eagle3')
    names = ('benchmark', 'question_id', 'turn_id', 'input_token_ids', 'input_sha256')
    cases = [{name: case[name] for name in names} for case in selected['greedy_cases'] if case['benchmark'] in REFERENCES and case['turn_id'] == 0]
    require(len(cases) == 160 and all(sum(case['benchmark'] == benchmark for case in cases) == 80 for benchmark in REFERENCES), 'Incomplete two-benchmark first-turn panel')
    require(len({(case['benchmark'], case['question_id']) for case in cases}) == 160, 'Duplicate input question')
    args.prepared.mkdir(parents=True, exist_ok=False)
    write(args.prepared / 'inputs.json', {'selected': selected, 'cases': cases})
    write(args.prepared / 'manifest.json', {'scope': 'paired_first_turn_seed_diagnostic_not_reproduction_admission', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': digest(__file__), 'protocol_sha256': digest(Path(__file__).with_name('PREREGISTRATION.md')), 'utility_path': str(args.utility.resolve()), 'utility_sha256': digest(args.utility), 'validation_driver': str(validation_driver), 'validation_driver_sha256': DRIVER_SHA256, 'registered_cases_sha256': REGISTERED_SHA256, 'upstream_generator_sha256': UPSTREAM_SHA256, 'context_policy': str(args.policy.resolve()), 'context_policy_sha256': digest(args.policy), 'inputs_sha256': digest(args.prepared / 'inputs.json'), 'arms': ARMS, 'repeats': 5, 'legacy_base_seed': 20261110, 'training_admission': False})


def validated(directory):
    manifest = read(directory / 'manifest.json')
    require(manifest['source_sha256'] == digest(__file__) and manifest['protocol_sha256'] == digest(Path(__file__).with_name('PREREGISTRATION.md')), 'Seed diagnostic source or registration changed')
    require(manifest['inputs_sha256'] == digest(directory / 'inputs.json') and manifest['context_policy_sha256'] == digest(manifest['context_policy']), 'Diagnostic input or resource policy changed')
    require(manifest['utility_sha256'] == UTILITY_SHA256 and manifest['validation_driver_sha256'] == DRIVER_SHA256, 'Utility or collector identity differs')
    require(digest(manifest['utility_path']) == UTILITY_SHA256 and digest(manifest['validation_driver']) == DRIVER_SHA256, 'Dependency bytes changed')
    return manifest


def worker(args):
    require(os.environ.get('SLURM_JOB_ID'), 'GPU worker requires Slurm')
    import torch
    from taps_rebuild.runner import verify_provenance

    manifest = validated(args.output)
    inputs = read(args.output / 'inputs.json')
    selected = inputs['selected']
    verify_provenance(args.root, selected['run_config'], selected['run_manifest'], 'eagle3')
    driver = load(manifest['validation_driver'], 'seed_validation_driver', DRIVER_SHA256)
    backend = driver.backend_for(selected, 'eagle3', 'float16')
    cases = [case for index, case in enumerate(inputs['cases']) if index % 8 == args.index]
    require(len(cases) == 20, 'Unexpected worker question count')
    path = args.output / ('seed-' + str(args.index) + '.jsonl')
    count = 0
    try:
        for _ in range(3):
            driver.generate(backend, cases[0]['input_token_ids'], 0, 1, 512, 'speculative')
        with path.open('x') as stream:
            for case in cases:
                for repeat in range(5):
                    arms = ARMS if (case['question_id'] + repeat) % 2 == 0 else tuple(reversed(ARMS))
                    for arm in arms:
                        seed = actual_seed(arm, case, repeat)
                        result = driver.generate(backend, case['input_token_ids'], seed, 1, 512, 'speculative')
                        stream.write(json.dumps({'case': case, 'arm': arm, 'actual_seed': seed, 'repeat': repeat, 'temperature': 1, **result}, allow_nan=False) + '\n')
                        count += 1
                stream.flush()
            os.fsync(stream.fileno())
        write(path.with_suffix('.receipt.json'), {'status': 'complete', 'worker': args.index, 'records': count, 'warmups_excluded': 3, 'raw_sha256': digest(path), 'manifest_sha256': digest(args.output / 'manifest.json'), 'source_sha256': digest(__file__), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__})
    finally:
        backend.close()


def summarize_rows(rows, cases):
    expected = {(case['benchmark'], case['question_id']): case for case in cases}
    require(len(expected) == 160, 'Incomplete registered question population')
    seen, values, outputs = set(), {}, {}
    for row in rows:
        case = row['case']
        question = (case['benchmark'], case['question_id'])
        require(question in expected and case == expected[question], 'Input differs from registration')
        arm, repeat = row['arm'], row['repeat']
        require(row['actual_seed'] == actual_seed(arm, case, repeat) and row['seed'] == row['actual_seed'] and row['temperature'] == 1, 'Actual RNG policy differs')
        key = (question, arm, repeat)
        require(key not in seen, 'Duplicate measurement')
        seen.add(key)
        cycles = row['cycles']
        require(cycles and sum(cycle['emitted_tokens'] for cycle in cycles) == len(row['output_token_ids']), 'Cycle coverage differs from emitted tokens')
        values.setdefault((question, arm), []).append(len(row['output_token_ids']) / len(cycles))
        outputs.setdefault((question, arm), []).append(row['output_token_ids'])
    require(seen == {(question, arm, repeat) for question in expected for arm in ARMS for repeat in range(5)}, 'Incomplete seed comparison coverage')
    cells = []
    for benchmark, reference in REFERENCES.items():
        questions = sorted(question for question in expected if question[0] == benchmark)
        require(len(questions) == 80, 'Incomplete benchmark')
        means = {arm: [sum(values[(question, arm)]) / 5 for question in questions] for arm in ARMS}
        for arm in ARMS:
            tau = sum(means[arm]) / 80
            cells.append({'benchmark': benchmark, 'arm': arm, 'questions': 80, 'timing_repeats': 5, 'question_mean_tokens_per_cycle': tau, 'published_reference_tau': reference, 'relative_error': (tau - reference) / reference, 'within_five_percent_diagnostic': abs(tau - reference) / reference <= 0.05, 'questions_with_identical_replay_tokens': sum(all(sequence == outputs[(question, arm)][0] for sequence in outputs[(question, arm)]) for question in questions)})
        differences = [a - b for a, b in zip(means[ARMS[0]], means[ARMS[1]])]
        generator = random.Random(20261111)
        draws = sorted(sum(differences[generator.randrange(80)] for _ in range(80)) / 80 for _ in range(5000))
        cells.append({'benchmark': benchmark, 'contrast': 'upstream_seed_zero_minus_registered_hash', 'paired_question_mean_difference': sum(differences) / 80, 'bootstrap_ci95': [draws[124], draws[4874]], 'bootstrap_samples': 5000, 'bootstrap_unit': 'question', 'questions': 80, 'timing_repeats_are_not_independent_questions': True})
    return {'scope': 'paired_first_turn_seed_diagnostic_not_reproduction_admission', 'status': 'complete', 'measured_generations': len(seen), 'cells': cells, 'training_admission': False, 'speedup_claim': False}


def controller(args):
    from taps_rebuild.pool import allocation, worker_environment

    manifest = validated(args.prepared)
    utility = load(manifest['utility_path'], 'seed_resource_utility', UTILITY_SHA256)
    hardware = allocation(8)
    require(all(item['memory_total_mib'] >= 75000 for item in hardware['gpu_inventory']), 'Eight 80GB GPUs are required')
    policy = read(manifest['context_policy'])
    require(policy['node'] == hardware['node'], 'Resource policy belongs to another node')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('inputs.json', 'manifest.json'):
        (args.output / name).write_bytes((args.prepared / name).read_bytes())
    write(args.output / 'allocation.json', hardware)
    def query():
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=15)
        apps = [{'gpu_uuid': row[0].strip(), 'pid': int(row[1]), 'process_name': row[2].strip(), 'used_memory_mib': int(row[3])} for row in csv.reader(io.StringIO(text)) if row]
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15)
        free = {row[0].strip(): int(row[1]) for row in csv.reader(io.StringIO(text)) if row}
        require(set(free) == {item['uuid'] for item in hardware['gpu_inventory']}, 'GPU memory inventory changed')
        blocked = utility.blocking_contexts(apps, free, policy, lambda pid: Path('/proc', str(pid)).exists(), utility.host_proc_visible())
        temporary = args.output / 'gpu-context.tmp'
        temporary.write_text(json.dumps({'applications': apps, 'free_memory_mib': free, 'blocking_contexts': blocked, 'policy': policy, 'timing_benchmark_claim': False}, indent=2) + '\n')
        temporary.replace(args.output / 'gpu-context.json')
        return '\n'.join(str(item['pid']) for item in blocked)
    utility.wait_for_idle(query, lambda _: None)
    selected = read(args.output / 'inputs.json')['selected']
    python = Path(selected['config']['legacy_runtime_root']) / 'envs/taps-eagle/bin/python'
    processes = []
    try:
        for binding in hardware['device_bindings']:
            index = binding['shard_index']
            environment = worker_environment(binding['token'], Path(selected['authored_source_root']), args.root, python)
            environment['PATH'] = str(python.parent) + ':/usr/local/bin:/usr/bin:/bin'
            log = (args.output / ('worker-' + str(index) + '.log')).open('xb')
            command = [str(python), '-B', str(Path(__file__).resolve()), '--action', 'worker', '--root', str(args.root), '--prepared', str(args.prepared), '--output', str(args.output), '--index', str(index)]
            try:
                process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            processes.append((process, log))
        codes = [process.wait() for process, _ in processes]
        require(not any(codes), 'Seed diagnostic worker failed: ' + str(codes))
        rows, receipts = [], []
        for index in range(8):
            path = args.output / ('seed-' + str(index) + '.jsonl')
            receipt = read(path.with_suffix('.receipt.json'))
            require(receipt['status'] == 'complete' and receipt['worker'] == index and receipt['records'] == 200 and receipt['raw_sha256'] == digest(path) and receipt['manifest_sha256'] == digest(args.output / 'manifest.json') and receipt['source_sha256'] == digest(__file__), 'Worker receipt differs')
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
            receipts.append({'path': str(path), 'sha256': digest(path), 'receipt_sha256': digest(path.with_suffix('.receipt.json'))})
        report = summarize_rows(rows, read(args.output / 'inputs.json')['cases'])
        report.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), manifest_sha256=digest(args.output / 'manifest.json'), gpu_context_sha256=digest(args.output / 'gpu-context.json'), workers=receipts)
        write(args.output / 'report.json', report)
    finally:
        for process, log in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('prepare', 'controller', 'worker'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=int, choices=range(8))
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.utility = Path(__file__).with_name('utility.py')
    args.policy = Path(__file__).with_name('stale-context.json')
    require(args.action != 'worker' or args.index is not None, 'Worker index is required')
    {'prepare': prepare, 'controller': controller, 'worker': worker}[args.action](args)


if __name__ == '__main__':
    main()
