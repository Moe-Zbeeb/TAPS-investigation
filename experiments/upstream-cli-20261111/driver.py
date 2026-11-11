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


SOURCE_SHA256 = '6daaf6f8010ff986eadfd5a5da7546b7ea9107c4ecdcfed57d694335dc78aef3'
CASES_SHA256 = 'd3f236a73ab18940bd28f76112210449a2f6ef797061d50a1ab98fc059e0a865'


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows, questions):
    expected = {row['question_id']: len(row['turns']) for row in questions}
    if len(rows) != len(expected) or len({row['question_id'] for row in rows}) != len(rows) or {row['question_id'] for row in rows} != set(expected):
        raise ValueError('Native CLI question coverage differs')
    values, index_ratios = [], []
    for row in rows:
        if len(row['choices']) != 1 or row['choices'][0]['index'] != 0:
            raise ValueError('Native CLI choice population differs')
        choice = row['choices'][0]
        count = expected[row['question_id']]
        if any(len(choice[key]) != count for key in ('turns', 'idxs', 'new_tokens', 'wall_time')) or any(type(value) is not int or value < 0 for key in ('idxs', 'new_tokens') for value in choice[key]):
            raise ValueError('Native CLI turn counters are incomplete or invalid')
        tokens = sum(choice['new_tokens'])
        indices = sum(choice['idxs'])
        values.append(tokens / (indices + count))
        if indices:
            index_ratios.append(tokens / indices)
    return {'questions': len(rows), 'turns': sum(expected.values()), 'official_counter_question_mean_tau': sum(values) / len(values), 'zero_based_index_ratio_diagnostic': sum(index_ratios) / len(index_ratios) if len(index_ratios) == len(values) else None, 'raw_token_counter_equivalence_established': False}


def controller(args):
    from taps_rebuild.pool import allocation, atomic_json, worker_environment
    from taps_rebuild.runner import verify_provenance

    cases = args.root / 'outputs/validation-inputs/registered-20261110-v2/cases.json'
    if digest(cases) != CASES_SHA256:
        raise ValueError('Original native question/model registration changed')
    selected = read(cases)['methods']['eagle3']
    verify_provenance(args.root, selected['run_config'], selected['run_manifest'], 'eagle3')
    source = Path(selected['source']['path'])
    script = source / 'eagle/evaluation/gen_ea_answer_llama3chat.py'
    if digest(script) != SOURCE_SHA256:
        raise ValueError('Original upstream CLI changed')
    hardware = allocation(8)
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('native_cli_context', bundle / 'context.py')
    context = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context)
    policy = read(bundle / 'stale-context.json')
    policy = policy if hardware['node'] == policy['node'] else None
    def query():
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=15)
        apps = [{'gpu_uuid': row[0].strip(), 'pid': int(row[1]), 'process_name': row[2].strip(), 'used_memory_mib': int(row[3])} for row in csv.reader(io.StringIO(text)) if row]
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15)
        free = {row[0].strip(): int(row[1]) for row in csv.reader(io.StringIO(text)) if row}
        if set(free) != {gpu['uuid'] for gpu in hardware['gpu_inventory']}:
            raise ValueError('Native CLI physical inventory differs')
        blocked = context.blocking_contexts(apps, free, policy, lambda pid: Path('/proc', str(pid)).exists(), context.host_proc_visible())
        atomic_json(args.output / 'gpu-context.json', {'applications': apps, 'blocking_contexts': blocked, 'free_memory_mib': free, 'policy': policy, 'no_signals_or_resets': True})
        return str(blocked) if blocked else ''
    context.wait_for_idle(query, lambda values: None)
    python = Path(selected['config']['legacy_runtime_root']) / 'envs/taps-eagle/bin/python'
    manifest = {'hardware': hardware, 'driver_sha256': digest(__file__), 'protocol_sha256': digest(bundle / 'PREREGISTRATION.md'), 'upstream_script_sha256': digest(script), 'context_sha256': digest(bundle / 'context.py'), 'context_policy_sha256': digest(bundle / 'stale-context.json'), 'selected_provenance': selected['provenance'], 'question_files': {}, 'commands': []}
    summaries = {}
    for benchmark, published in [('gsm8k', 4.59), ('mt_bench', 4.24)]:
        questions_file = source / 'eagle/data' / benchmark / 'question.jsonl'
        questions = [json.loads(line) for line in questions_file.read_text().splitlines() if line.strip()]
        if len(questions) != 80:
            raise ValueError('Native CLI control requires the complete original 80-question file')
        manifest['question_files'][benchmark] = {'path': str(questions_file), 'sha256': digest(questions_file)}
        processes = []
        try:
            for binding in hardware['device_bindings']:
                index = binding['shard_index']
                environment = worker_environment(binding['token'], Path(selected['authored_source_root']), args.root, python)
                environment.update(PYTHONPATH=str(source), PATH=str(python.parent) + ':/usr/local/bin:/usr/bin:/bin')
                answer = args.output / (benchmark + '-' + str(index) + '.jsonl')
                command = [str(python), '-B', '-m', 'eagle.evaluation.gen_ea_answer_llama3chat', '--base-model-path', selected['models']['target']['path'], '--ea-model-path', selected['models']['drafter']['path'], '--model-id', 'taps-upstream-procedure-control', '--bench-name', benchmark, '--question-begin', str(index*10), '--question-end', str((index+1)*10), '--answer-file', str(answer), '--num-choices', '1', '--num-gpus-per-model', '1', '--num-gpus-total', '1', '--temperature', '1', '--total-token', '60', '--depth', '7', '--top-k', '10', '--use_eagle3', '--max-new-token', '1024']
                manifest['commands'].append({'benchmark': benchmark, 'shard': index, 'command': command, 'cuda_visible_devices': binding['token']})
                atomic_json(args.output / 'manifest.json', manifest)
                log = (args.output / (benchmark + '-' + str(index) + '.log')).open('xb')
                processes.append((subprocess.Popen(command, cwd=args.root, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT), log))
            while any(process.poll() is None for process, _ in processes):
                if any(process.poll() not in (None, 0) for process, _ in processes):
                    raise ValueError('An unmodified upstream CLI worker failed')
                time.sleep(1)
            if any(process.returncode != 0 for process, _ in processes):
                raise ValueError('Native CLI wave failed')
        finally:
            for process, _ in processes:
                if process.poll() is None:
                    process.terminate()
            for process, log in processes:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
        rows = [json.loads(line) for path in sorted(args.output.glob(benchmark+'-*.jsonl')) for line in path.read_text().splitlines()]
        summary = summarize(rows, questions)
        error = (summary['official_counter_question_mean_tau'] - published) / published
        summaries[benchmark] = {**summary, 'published_tau': published, 'relative_error': error, 'within_five_percent': abs(error) <= 0.05}
        atomic_json(args.output / 'progress.json', {'completed_benchmarks': summaries})
    report = {'status': 'complete', 'scope': 'Unmodified original CLI procedure diagnostic; official counters, no paired AR speedup or training admission', 'summaries': summaries, 'manifest_sha256': digest(args.output / 'manifest.json'), 'raw_files': {path.name: digest(path) for path in sorted(args.output.glob('*.jsonl'))}, 'training_admission': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(args.output / 'report.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    controller(parser.parse_args())


if __name__ == '__main__':
    main()
