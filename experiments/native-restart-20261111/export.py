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
import sys
import time


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def worker(args):
    import numpy
    import torch
    from taps_rebuild.backends.official import OfficialBackend
    from taps_rebuild.runner import require_gpu_slurm, seed_runtime, verify_provenance

    require_gpu_slurm()
    manifest = read(args.output / 'manifest.json')
    if manifest['driver_sha256'] != digest(__file__) or digest(Path(manifest['draft_path']) / 'pytorch_model.bin') != manifest['weights_sha256']:
        raise ValueError('Frozen native export bytes changed')
    selected = manifest['selected']
    verify_provenance(args.root, selected['run_config'], selected['run_manifest'], 'eagle3')
    state = torch.load(Path(manifest['draft_path']) / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    if any(name.startswith('target_model.') for name in state) or not {'d2t', 't2d'} <= set(state):
        raise ValueError('Native export contains target weights or lacks vocabulary buffers')
    backend = OfficialBackend('eagle3', selected['source']['path'], selected['models']['target']['path'], manifest['draft_path'], total_token=60, depth=7, top_k=10)
    try:
        loaded = backend.model.ea_layer.load_state_dict(state, strict=False)
        if loaded.unexpected_keys or set(loaded.missing_keys) - {'embed_tokens.weight'}:
            raise ValueError('Native exported draft keys differ from the actual inference architecture')
        case = manifest['cases'][args.index]
        outputs = {}
        for mode in ('target_only', 'speculative'):
            seed_runtime(case['seed'], torch, numpy)
            result = backend.generate(torch.tensor([case['input_token_ids']]), temperature=0, max_new_tokens=16, max_length=2048, generation_mode=mode)
            tokens = result['output_ids'][0, len(case['input_token_ids']):].cpu().tolist()
            if not tokens or any(type(token) is not int or not 0 <= token < backend.model.base_model.config.vocab_size for token in tokens) or result['truecycle_count'] < 1:
                raise ValueError('Exported draft failed native inference')
            outputs[mode] = {'raw_output_token_ids': tokens, 'cycles': result['cycles']}
        write(args.output / ('rank-' + str(args.index) + '.json'), {'rank': args.index, 'passed': True, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'weights_sha256': manifest['weights_sha256'], 'case': case, 'missing_frozen_keys': loaded.missing_keys, 'outputs': outputs, 'scientific_quality_result': False})
    finally:
        backend.close()


def controller(args):
    from taps_rebuild.pool import allocation, worker_environment

    proof = read(args.proof / 'report.json')
    if proof['status'] != 'passed':
        raise ValueError('Fresh-process qualification must pass before export qualification')
    prepared = read(args.proof / 'prepared.json')
    if digest(args.proof / 'prepared.json') != proof['prepared_sha256']:
        raise ValueError('Qualified native training plans changed')
    plan = read(prepared['plans']['control']['path'])
    draft = Path(plan['trained_models']['resolved_path']) / 'budget_step_2'
    if digest(draft / 'config.json') != plan['frozen_files']['work/config.json']:
        raise ValueError('Native exported configuration changed')
    registered = args.root / 'outputs/validation-inputs/registered-20261110-v2/cases.json'
    if digest(registered) != 'd3f236a73ab18940bd28f76112210449a2f6ef797061d50a1ab98fc059e0a865':
        raise ValueError('Frozen export input population changed')
    selected = read(registered)['methods']['eagle3']
    cases = sorted(selected['greedy_cases'], key=lambda case: hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest())
    chosen = []
    for benchmark in ('mt_bench', 'humaneval', 'gsm8k', 'alpaca', 'cnn_dm'):
        chosen.append(next(case for case in cases if case['benchmark'] == benchmark))
    chosen.extend(case for case in cases if case not in chosen)
    hardware = allocation(8)
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('export_context', bundle / 'context.py')
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
            raise ValueError('Export allocation physical inventory differs')
        blocked = context.blocking_contexts(apps, free, policy, lambda pid: Path('/proc', str(pid)).exists(), context.host_proc_visible())
        return str(blocked) if blocked else ''
    context.wait_for_idle(query, lambda values: None)
    manifest = {'driver_sha256': digest(__file__), 'context_sha256': digest(bundle / 'context.py'), 'policy_sha256': digest(bundle / 'stale-context.json'), 'hardware': hardware, 'proof_sha256': digest(args.proof / 'report.json'), 'draft_path': str(draft), 'weights_sha256': digest(draft / 'pytorch_model.bin'), 'selected': selected, 'cases': chosen[:8]}
    write(args.output / 'manifest.json', manifest)
    python = Path(selected['config']['legacy_runtime_root']) / 'envs/taps-eagle/bin/python'
    processes = []
    try:
        for binding in hardware['device_bindings']:
            index = binding['shard_index']
            environment = worker_environment(binding['token'], Path(selected['authored_source_root']), args.root, python)
            environment['PATH'] = str(python.parent) + ':/usr/local/bin:/usr/bin:/bin'
            log = (args.output / ('worker-' + str(index) + '.log')).open('xb')
            command = [str(python), '-B', str(Path(__file__).resolve()), 'worker', '--root', str(args.root), '--proof', str(args.proof), '--output', str(args.output), '--index', str(index)]
            processes.append((subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT), log))
        while any(process.poll() is None for process, _ in processes):
            if any(process.poll() not in (None, 0) for process, _ in processes):
                raise ValueError('A native export inference worker failed')
            time.sleep(1)
        if any(process.returncode != 0 for process, _ in processes):
            raise ValueError('Native export inference wave failed')
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
    rows = [read(args.output / ('rank-' + str(rank) + '.json')) for rank in range(8)]
    passed = {row['rank'] for row in rows} == set(range(8)) and all(row['passed'] and row['weights_sha256'] == manifest['weights_sha256'] for row in rows)
    write(args.output / 'report.json', {'status': 'passed' if passed else 'failed', 'scope': 'Eight-GPU stock native inference of the two-update engineering export; no quality or speed benchmark', 'native_training_sha256': proof['native_training_sha256'], 'native_recovery_sha256': proof['native_recovery_sha256'], 'manifest_sha256': digest(args.output / 'manifest.json'), 'ranks': [{'rank': row['rank'], 'passed': row['passed'], 'receipt_sha256': digest(args.output / ('rank-' + str(row['rank']) + '.json'))} for row in rows], 'training_admission': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()})
    if not passed:
        raise ValueError('Native export qualification failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('controller', 'worker'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=int, choices=range(8))
    args = parser.parse_args()
    if args.action == 'worker' and args.index is None:
        parser.error('Worker index required')
    {'controller': controller, 'worker': worker}[args.action](args)


if __name__ == '__main__':
    main()
