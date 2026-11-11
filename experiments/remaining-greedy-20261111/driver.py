import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import time


BENCHMARKS = ('alpaca', 'cnn_dm', 'gsm8k', 'humaneval', 'mt_bench')
REGISTERED_SHA256 = 'd3f236a73ab18940bd28f76112210449a2f6ef797061d50a1ab98fc059e0a865'
PRIMARY_SUMMARY_SHA256 = '9867dc88c16a9a8f5688ddac6ede6de087ad11910978890931e7f6d977f9a4df'
VALIDATION_DRIVER_SHA256 = 'd6ea2b5216b0052aa68282646b343143c6523ed14e75f8b2b838f99bb6458934'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def key(row):
    case = row['case']
    return case['benchmark'], case['question_id'], case['turn_id'], case['input_sha256']


def select(rows):
    require(len(rows) == 480 and len({key(row) for row in rows}) == 480, 'Incomplete or duplicate EAGLE-3 panel')
    for row in rows:
        case, outputs = row['case'], row['outputs']
        require(row['method'] == 'eagle3' and row['precision'] == 'float32', 'Unexpected method or precision')
        require(identity(case['input_token_ids']) == case['input_sha256'], 'Input token identity differs')
        require(row['equal'] == (outputs['target_only']['comparison_token_ids'] == outputs['speculative']['comparison_token_ids']), 'Stored comparison differs')
    failures = [row for row in rows if not row['equal']]
    require({key(row)[:3] for row in failures} == {('mt_bench', 146, 1), ('cnn_dm', 295, 0), ('cnn_dm', 288, 0)}, 'Original failure population differs')
    target = [row for row in failures if row['case']['benchmark'] == 'mt_bench' and row['case']['question_id'] == 146 and row['case']['turn_id'] == 1]
    require(len(target) == 1, 'Registered remaining token failure is missing')
    ar, spec = (target[0]['outputs'][mode]['comparison_token_ids'] for mode in ('target_only', 'speculative'))
    divergence = next((i for i, (a, b) in enumerate(zip(ar, spec)) if a != b), None)
    require(divergence == 200, 'Registered first token disagreement differs')
    candidates = {}
    for benchmark in BENCHMARKS:
        candidates[benchmark] = sorted((row for row in rows if row['equal'] and row['case']['benchmark'] == benchmark and len(row['outputs']['target_only']['comparison_token_ids']) >= 37), key=lambda row: identity(row['case']))
        require(candidates[benchmark], 'A registered control benchmark is missing')
    chosen = [(target[0], divergence, 'remaining_token_failure')]
    while len(chosen) < 8:
        before = len(chosen)
        for benchmark in BENCHMARKS:
            if candidates[benchmark] and len(chosen) < 8:
                chosen.append((candidates[benchmark].pop(0), 32, 'matched_control'))
        require(len(chosen) > before, 'Insufficient distinct controls')
    cases = []
    labels = []
    for row, probe, label in chosen:
        case = row['case']
        pair = {mode: {**row['outputs'][mode], **{name: case[name] for name in ('input_token_ids', 'input_sha256', 'benchmark', 'question_id', 'turn_id')}, 'seed': case['seed']} for mode in ('target_only', 'speculative')}
        cases.append({'divergence': probe, 'pair': pair})
        labels.append({'role': label, 'case': case, 'probe_offset': probe})
    return cases, labels


def prepare(args):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preparation requires a CPU Slurm task')
    require(digest(args.primary / 'greedy-summary.json') == PRIMARY_SUMMARY_SHA256, 'Primary summary identity differs')
    rows, inputs = [], []
    for path in sorted(args.primary.glob('greedy-eagle3-*.jsonl')):
        receipt = read(path.with_suffix('.receipt.json'))
        require(receipt['status'] == 'complete' and receipt['raw_sha256'] == digest(path), 'Primary raw proof differs')
        inputs.append({'path': str(path), 'sha256': digest(path)})
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    cases, labels = select(rows)
    prepared = read(args.registered / 'cases.json')
    receipt = read(args.registered / 'completion.json')
    require(receipt['status'] == 'complete' and receipt['cases_sha256'] == digest(args.registered / 'cases.json') == REGISTERED_SHA256, 'Registered preparation proof differs')
    selected = prepared['methods']['eagle3']
    require(digest(args.validation_driver) == prepared['driver_sha256'] == VALIDATION_DRIVER_SHA256, 'Registered validation driver differs')
    from taps_rebuild.runner import verify_provenance

    verify_provenance(args.root, selected['run_config'], selected['run_manifest'], 'eagle3')
    args.prepared.mkdir(parents=True, exist_ok=False)
    write(args.prepared / 'cases.json', {'eagle3': {'run_id': selected['run_id'], 'config': selected['config'], 'definition': selected['definition'], 'target_path': selected['models']['target']['path'], 'draft_path': selected['models']['drafter']['path'], 'source_path': selected['source']['path'], 'authored_source_root': selected['authored_source_root'], 'run_config': selected['run_config'], 'run_manifest': selected['run_manifest'], 'cases': cases}})
    policy = {'path': str(args.stale_context_policy.resolve()), 'sha256': digest(args.stale_context_policy)} if args.stale_context_policy else None
    write(args.prepared / 'manifest.json', {'scope': 'selected_fp32_failure_and_controls_not_benchmark', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': digest(__file__), 'audit_worker': str(args.audit_worker.resolve()), 'audit_worker_sha256': digest(args.audit_worker), 'protocol_sha256': digest(Path(__file__).with_name('PREREGISTRATION.md')), 'registered_cases_sha256': digest(args.registered / 'cases.json'), 'validation_driver_sha256': digest(args.validation_driver), 'input_raw': inputs, 'tasks': labels, 'cases_sha256': digest(args.prepared / 'cases.json'), 'stale_context_policy': policy, 'training_admission': False})


def validate(args, directory):
    manifest = read(directory / 'manifest.json')
    require(manifest['source_sha256'] == digest(__file__), 'Diagnostic driver changed')
    require(manifest['cases_sha256'] == digest(directory / 'cases.json'), 'Diagnostic cases changed')
    require(manifest['audit_worker_sha256'] == digest(manifest['audit_worker']), 'Observational worker changed')
    require(manifest['protocol_sha256'] == digest(Path(__file__).with_name('PREREGISTRATION.md')), 'Diagnostic registration changed')
    if manifest.get('stale_context_policy'):
        require(digest(manifest['stale_context_policy']['path']) == manifest['stale_context_policy']['sha256'], 'GPU context policy changed')
    return manifest


def worker(args):
    require(os.environ.get('SLURM_JOB_ID'), 'GPU worker requires Slurm')
    manifest = validate(args, args.output)
    selected = read(args.output / 'cases.json')['eagle3']
    from taps_rebuild.runner import verify_provenance

    verify_provenance(args.root, selected['run_config'], selected['run_manifest'], 'eagle3')
    spec = importlib.util.spec_from_file_location('observational_greedy_worker', manifest['audit_worker'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.worker(args.root, args.output, 'eagle3', args.index, 'float32')


def summarize(output):
    manifest = read(output / 'manifest.json')
    results = []
    for index, task in enumerate(manifest['tasks']):
        path = output / ('eagle3-float32-' + str(index) + '.json')
        receipt = read(path)
        require(receipt['status'] == 'complete' and receipt['worker'] == index and receipt['precision'] == 'float32' and receipt['method'] == 'eagle3', 'Incomplete worker')
        require(receipt['cases_sha256'] == manifest['cases_sha256'] and receipt['script_sha256'] == manifest['audit_worker_sha256'], 'Worker identity differs')
        require((receipt['benchmark'], receipt['question_id'], receipt['turn_id']) == (task['case']['benchmark'], task['case']['question_id'], task['case']['turn_id']) and receipt['divergence'] == task['probe_offset'], 'Worker case differs')
        records = receipt['records']
        require(set(records) == {'target_only', 'speculative'}, 'Missing paired mode')
        valid = all(record.get('captured') and record.get('prefix_matches_original_common_prefix') and record.get('replay_prefix_matches_original_through_divergence') and record.get('two_replays_identical') for record in records.values())
        results.append({'task': task, 'valid_original_prefix_capture': valid, 'comparison': receipt['comparison'], 'records': records, 'receipt': path.name, 'receipt_sha256': digest(path)})
    require(len(results) == 8, 'Incomplete worker coverage')
    context = output / 'gpu-context.json'
    write(output / 'report.json', {'scope': manifest['scope'], 'status': 'complete', 'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': digest(output / 'manifest.json'), 'gpu_context_sha256': digest(context) if context.is_file() else None, 'results': results, 'all_original_prefix_captures_valid': all(row['valid_original_prefix_capture'] for row in results), 'benchmark_claim': False, 'training_admission': False})


def wait_for_idle(query, observe, clock=time.monotonic, sleep=time.sleep, timeout=300):
    started = clock()
    observations = []
    while True:
        processes = query().strip()
        elapsed = clock() - started
        observations.append({'elapsed_seconds': elapsed, 'reported_processes': processes})
        observe(observations)
        if not processes:
            return observations
        require(elapsed < timeout, 'GPU processes remained after the bounded scheduler cleanup wait; no models were loaded')
        sleep(min(5, timeout - elapsed))


def blocking_contexts(applications, free_memory, policy, pid_exists, proc_visible):
    allowed = policy['contexts'] if policy else []
    blocked = []
    for app in applications:
        matching = [item for item in allowed if item['gpu_uuid'] == app['gpu_uuid'] and item['pid'] == app['pid'] and item['process_name'] == app['process_name'] and app['used_memory_mib'] <= item['max_used_memory_mib']]
        if not (matching and proc_visible and not pid_exists(app['pid'])):
            blocked.append(app)
    if not blocked:
        minimum = policy['min_free_gpu_mib'] if policy else 51200
        require(bool(free_memory) and all(value >= minimum for value in free_memory.values()), 'Insufficient free GPU memory; no models were loaded')
    return blocked


def host_proc_visible():
    mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines() if line.split()[1] == '/proc']
    namespace = [line.split()[1:] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('NSpid:')]
    return len(mounts) == 1 and mounts[0][2] == 'proc' and not any(option.startswith('hidepid=') and option != 'hidepid=0' for option in mounts[0][3].split(',')) and len(namespace) == 1 and len(namespace[0]) == 1 and Path('/proc/1/comm').read_text().strip() == 'systemd'


def controller(args):
    from taps_rebuild.pool import allocation, worker_environment

    hardware = allocation(8)
    require(all(item['memory_total_mib'] >= 75000 for item in hardware['gpu_inventory']), 'FP32 diagnostic requires eight 80GB-class GPUs')
    manifest = validate(args, args.prepared)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('cases.json', 'manifest.json'):
        (args.output / name).write_bytes((args.prepared / name).read_bytes())
    write(args.output / 'allocation.json', hardware)
    policy = read(manifest['stale_context_policy']['path']) if manifest.get('stale_context_policy') else None
    require(policy is None or policy['node'] == hardware['node'], 'GPU context exception belongs to another node')
    def query():
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=15)
        applications = [{'gpu_uuid': row[0].strip(), 'pid': int(row[1]), 'process_name': row[2].strip(), 'used_memory_mib': int(row[3])} for row in csv.reader(io.StringIO(text)) if row]
        memory = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15)
        free_memory = {row[0].strip(): int(row[1]) for row in csv.reader(io.StringIO(memory)) if row}
        require(set(free_memory) == {item['uuid'] for item in hardware['gpu_inventory']}, 'Memory inventory differs from the owned GPUs')
        visible = host_proc_visible()
        blocked = blocking_contexts(applications, free_memory, policy, lambda pid: Path('/proc', str(pid)).exists(), visible)
        temporary = args.output / 'gpu-context.tmp'
        temporary.write_text(json.dumps({'applications': applications, 'blocking_contexts': blocked, 'free_memory_mib': free_memory, 'host_proc_visible': visible, 'policy': policy, 'no_signals_or_gpu_resets': True, 'timing_benchmark_claim': False}, indent=2) + '\n')
        temporary.replace(args.output / 'gpu-context.json')
        return '\n'.join(str(item['pid']) for item in blocked)
    def observe(observations):
        temporary = args.output / 'idle-check.tmp'
        temporary.write_text(json.dumps({'scope': 'read_only_no_signals_or_gpu_computation', 'observations': observations, 'idle_observed': not observations[-1]['reported_processes']}, indent=2) + '\n')
        temporary.replace(args.output / 'idle-check.json')
    wait_for_idle(query, observe)
    selected = read(args.output / 'cases.json')['eagle3']
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
        require(not any(codes), 'Diagnostic worker failed: ' + str(codes))
        summarize(args.output)
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
    parser.add_argument('--primary', type=Path)
    parser.add_argument('--registered', type=Path)
    parser.add_argument('--validation-driver', type=Path)
    parser.add_argument('--audit-worker', type=Path)
    parser.add_argument('--stale-context-policy', type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.primary = args.primary or args.root / 'outputs/validation-pipelines/primary-2212576-validation-queue-20261110-v2-full'
    args.registered = args.registered or args.root / 'outputs/validation-inputs/registered-20261110-v2'
    args.validation_driver = args.validation_driver or args.root / 'outputs/preparation/validation-queue-20261110-v2/driver.py'
    args.audit_worker = args.audit_worker or Path(__file__).with_name('audit_worker.py')
    default_policy = Path(__file__).with_name('stale-context.json')
    args.stale_context_policy = args.stale_context_policy or (default_policy if default_policy.is_file() else None)
    require(args.action != 'worker' or args.index is not None, 'Worker index is required')
    {'prepare': prepare, 'controller': controller, 'worker': worker}[args.action](args)


if __name__ == '__main__':
    main()
