import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REGISTERED = 'd3f236a73ab18940bd28f76112210449a2f6ef797061d50a1ab98fc059e0a865'
COLLECTOR = 'd6ea2b5216b0052aa68282646b343143c6523ed14e75f8b2b838f99bb6458934'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def setup(args):
    bundle = Path(__file__).resolve().parent
    collector = module(bundle / 'collector.py', 'registered_collector')
    collector.require(collector.digest(bundle / 'collector.py') == COLLECTOR, 'Original collector differs')
    collector.require(collector.digest(args.prepared / 'cases.json') == REGISTERED, 'Original registered population differs')
    prepared = collector.read(args.prepared / 'cases.json')
    collector.require(prepared['driver_sha256'] == COLLECTOR, 'Prepared collector identity differs')
    budget = module(bundle / 'validation_budget.py', 'fixed_validation_budget')
    return collector, prepared, budget


def worker(args):
    collector, prepared, budget = setup(args)
    from taps_rebuild.runner import decode_completion, stop_ids

    original = collector.generate
    def generate(backend, ids, seed, temperature, requested, mode):
        cap = budget.output_budget(len(ids), requested, 2048, int(backend.model.ea_layer.total_tokens))
        result = original(backend, ids, seed, temperature, cap, mode)
        _, result['legacy_requested_projection'] = decode_completion(backend.tokenizer, result['output_token_ids'], requested)
        result['comparison_token_ids'] = budget.comparison_tokens(result['output_token_ids'], cap, stop_ids(backend.tokenizer))
        result['fixed_output_budget'] = cap
        result['reserved_tree_tokens'] = int(backend.model.ea_layer.total_tokens)
        result['policy'] = 'predetermined-context-budget-v1'
        return result
    collector.generate = generate
    collector.worker(args)


def identities(collector, bundle, prepared):
    return {'files': {name: collector.digest(bundle / name) for name in ('driver.py', 'collector.py', 'validation_budget.py', 'PREREGISTRATION.md', 'context.py', 'stale-context.json')}, 'registered_cases_sha256': collector.digest(prepared / 'cases.json')}


def replay(args):
    collector, prepared, budget = setup(args)
    manifest = collector.read(args.output / 'manifest.json')
    collector.require(manifest['identities'] == identities(collector, Path(__file__).resolve().parent, args.prepared), 'Frozen control identities differ')
    seen, failures, counts, artifacts = set(), [], {}, []
    expected = {(method, collector.identity(case)) for method, selected in prepared['methods'].items() for case in selected['greedy_cases']}
    paths = sorted(args.output.glob('greedy-*.jsonl'))
    collector.require(len(paths) == 24, 'Missing worker journals')
    for path in paths:
        receipt = collector.read(path.with_suffix('.receipt.json'))
        collector.require(receipt['status'] == 'complete' and receipt['raw_sha256'] == collector.digest(path), 'Raw worker receipt differs')
        count, failed = 0, 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            method, case = row['method'], row['case']
            key = method, collector.identity(case)
            collector.require(key in expected and key not in seen and row['precision'] == 'float32', 'Substituted or duplicate registered case')
            collector.require(row['prepared_sha256'] == REGISTERED and row['provenance'] == prepared['methods'][method]['provenance'], 'Row provenance differs')
            collector.require(collector.identity(case['input_token_ids']) == case['input_sha256'], 'Input identity differs')
            seen.add(key)
            stops = manifest['stop_ids'][method]
            for mode in ('target_only', 'speculative'):
                result = row['outputs'][mode]
                collector.require(result['reserved_tree_tokens'] == prepared['methods'][method]['config']['total_token'] - 1, 'Native tree reservation differs from pinned configuration')
                cap = budget.output_budget(len(case['input_token_ids']), 512, 2048, result['reserved_tree_tokens'])
                collector.require(result['fixed_output_budget'] == cap and result['policy'] == 'predetermined-context-budget-v1', 'Output budget changed')
                collector.require(result['comparison_token_ids'] == budget.comparison_tokens(result['output_token_ids'], cap, stops), 'Raw comparison projection differs')
                collector.require(result['seed'] == case['seed'], 'Seed differs')
            ar, tree = (row['outputs'][mode]['comparison_token_ids'] for mode in ('target_only', 'speculative'))
            equal = ar == tree
            collector.require(equal == row['equal'], 'Raw equality differs')
            count += 1
            failed += int(not equal)
            counts[method] = counts.get(method, 0) + 1
            if not equal:
                first = next((index for index, values in enumerate(zip(ar, tree)) if values[0] != values[1]), None)
                failures.append({'method': method, 'benchmark': case['benchmark'], 'question_id': case['question_id'], 'turn_id': case['turn_id'], 'input_sha256': case['input_sha256'], 'first_token_difference': first, 'lengths': [len(ar), len(tree)]})
        collector.require(receipt['completed'] == count and receipt['failed_checks'] == failed, 'Worker failure/count receipt differs')
        artifacts.append({'path': path.name, 'sha256': collector.digest(path), 'receipt_sha256': collector.digest(path.with_suffix('.receipt.json'))})
    collector.require(seen == expected and counts == {'eagle3': 480, 'eagle2': 320, 'hass': 320}, 'Incomplete registered population')
    report = {'status': 'complete', 'scope': 'fixed-budget greedy diagnostic; no training admission or timing claim', 'identities': manifest['identities'], 'counts': counts, 'pairs': len(seen), 'exact_mismatches': len(failures), 'mismatch_rate': len(failures) / len(seen), 'failures': failures, 'all_exact_equal': not failures, 'artifacts': artifacts, 'training_admission': False, 'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    collector.atomic(args.output / args.report_name, report)
    return report


def controller(args):
    collector, prepared, _ = setup(args)
    from taps_rebuild.pool import allocation, worker_environment

    hardware = allocation(8)
    collector.require(all(gpu['memory_total_mib'] >= 75000 for gpu in hardware['gpu_inventory']), 'Requires eight 80GB A100s')
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = Path(__file__).resolve().parent
    context = module(bundle / 'context.py', 'frozen_gpu_context')
    policy = collector.read(bundle / 'stale-context.json')
    policy = policy if hardware['node'] == policy['node'] else None
    def query():
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=15)
        apps = [{'gpu_uuid': row[0].strip(), 'pid': int(row[1]), 'process_name': row[2].strip(), 'used_memory_mib': int(row[3])} for row in csv.reader(io.StringIO(text)) if row]
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15)
        free = {row[0].strip(): int(row[1]) for row in csv.reader(io.StringIO(text)) if row}
        collector.require(set(free) == {gpu['uuid'] for gpu in hardware['gpu_inventory']}, 'Physical memory inventory differs')
        blocked = context.blocking_contexts(apps, free, policy, lambda pid: Path('/proc', str(pid)).exists(), context.host_proc_visible())
        collector.atomic(args.output / 'gpu-context.json', {'applications': apps, 'blocking_contexts': blocked, 'free_memory_mib': free, 'policy': policy, 'no_signals_or_resets': True})
        return str(blocked) if blocked else ''
    context.wait_for_idle(query, lambda observations: collector.atomic(args.output / 'idle-check.json', observations))
    stops = {}
    for method, selected in prepared['methods'].items():
        python = Path(selected['config']['legacy_runtime_root']) / 'envs' / ('taps-' + selected['definition']['profile']) / 'bin/python'
        environment = worker_environment(hardware['device_bindings'][0]['token'], Path(selected['authored_source_root']), args.root, python)
        code = 'import json; from transformers import AutoTokenizer; from taps_rebuild.runner import stop_ids; import sys; print(json.dumps(sorted(stop_ids(AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True)))))'
        result = subprocess.check_output([str(python), '-B', '-c', code, selected['models']['target']['path']], text=True, env=environment, timeout=90)
        stops[method] = json.loads(result.strip())
    collector.atomic(args.output / 'manifest.json', {'hardware': hardware, 'identities': identities(collector, bundle, args.prepared), 'stop_ids': stops, 'training_admission': False})
    for method, selected in prepared['methods'].items():
        python = Path(selected['config']['legacy_runtime_root']) / 'envs' / ('taps-' + selected['definition']['profile']) / 'bin/python'
        processes = []
        try:
            for binding in hardware['device_bindings']:
                environment = worker_environment(binding['token'], Path(selected['authored_source_root']), args.root, python)
                environment['PATH'] = str(python.parent) + ':/usr/local/bin:/usr/bin:/bin'
                index = binding['shard_index']
                log = (args.output / (method + '-' + str(index) + '.log')).open('xb')
                command = [str(python), '-B', str(Path(__file__).resolve()), 'worker', '--root', str(args.root), '--prepared', str(args.prepared), '--output', str(args.output), '--method', method, '--index', str(index)]
                processes.append((subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT), log))
            while any(process.poll() is None for process, _ in processes):
                collector.require(all(process.poll() in (None, 0) for process, _ in processes), 'A fixed-budget worker failed')
                time.sleep(1)
            collector.require(all(process.returncode == 0 for process, _ in processes), 'A fixed-budget wave failed')
        finally:
            for process, log in processes:
                if process.poll() is None:
                    process.terminate()
            for process, log in processes:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
        print(json.dumps({'completed_method': method}), flush=True)
    report = replay(args)
    collector.atomic(args.output / 'completion.json', {'status': 'complete', 'report_sha256': collector.digest(args.output / args.report_name), 'all_exact_equal': report['all_exact_equal'], 'training_admission': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('controller', 'worker', 'replay'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=('eagle3', 'eagle2', 'hass'))
    parser.add_argument('--index', type=int, choices=range(8))
    parser.add_argument('--report-name', default='report.json')
    args = parser.parse_args()
    args.expected_gpus, args.phase = 8, 'greedy'
    if args.action == 'worker' and (args.method is None or args.index is None):
        parser.error('worker requires method and index')
    {'controller': controller, 'worker': worker, 'replay': replay}[args.action](args)


if __name__ == '__main__':
    main()
