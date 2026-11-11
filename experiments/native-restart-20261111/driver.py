import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess

from taps_rebuild import native_training as native


def check_bundle(bundle):
    identity = json.loads((bundle / 'bundle.json').read_bytes())
    if any(native.digest(bundle / name) != digest for name, digest in identity['files'].items()):
        raise ValueError('Frozen restart bundle changed')


def prepare(root, bundle, destination):
    native.require_slurm()
    check_bundle(bundle)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Restart fixture preparation requires CPU Slurm')
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'prepared.json').exists():
        prepared = json.loads((destination / 'prepared.json').read_bytes())
        if prepared['native_training_sha256'] != native.digest(native.__file__) or any(native.digest(item['path']) != item['sha256'] for item in prepared['plans'].values()):
            raise ValueError('Completed restart preparation differs on requeue')
        return
    attempt = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    attempt_directory = destination / ('preparation-attempt-' + str(attempt))
    attempt_directory.mkdir(exist_ok=False)
    fixture = root / 'data/engineering' / (destination.name + '-attempt-' + str(attempt))
    native.fixture(root, fixture)
    plans = {}
    for name, stop in [('control', None), ('interrupted', 1)]:
        config = json.loads((bundle / 'configs/native-eagle3-training.json').read_bytes())
        config.update(checkpoint_roundtrip=False, recovery={'checkpoint_interval': 1, 'stop_after_optimizer_step': stop})
        config_path = attempt_directory / (name + '-config.json')
        native.atomic_json(config_path, config)
        run = root / 'outputs/native-training' / (destination.name + '-attempt-' + str(attempt) + '-' + name)
        plan = native.prepare(root, config_path, fixture / 'train.jsonl', fixture / 'dev.jsonl', run)
        native.audit(run / 'plan.json')
        plans[name] = {'path': str(run / 'plan.json'), 'sha256': native.digest(run / 'plan.json'), 'plan_identity_sha256': plan['plan_identity_sha256']}
    report = {'status': 'prepared', 'scope': 'synthetic engineering process restart; no scientific training', 'plans': plans, 'native_training_sha256': native.digest(native.__file__), 'native_recovery_sha256': native.digest(Path(native.__file__).with_name('native_recovery.py')), 'protocol_sha256': native.digest(bundle / 'PREREGISTRATION.md'), 'bundle_sha256': native.digest(bundle / 'bundle.json')}
    native.atomic_json(destination / 'prepared.json', report)


def journals(run, rank):
    rows = []
    for path in sorted((run / 'attempts').glob('attempt-*/batches-rank-' + str(rank) + '.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


def controller(root, bundle, destination):
    check_bundle(bundle)
    prepared = json.loads((destination / 'prepared.json').read_bytes())
    if prepared['native_training_sha256'] != native.digest(native.__file__) or prepared['native_recovery_sha256'] != native.digest(Path(native.__file__).with_name('native_recovery.py')) or prepared['protocol_sha256'] != native.digest(bundle / 'PREREGISTRATION.md') or prepared['bundle_sha256'] != native.digest(bundle / 'bundle.json'):
        raise ValueError('Restart qualification source changed after preparation')
    from taps_rebuild.pool import allocation

    hardware = allocation(8)
    spec = importlib.util.spec_from_file_location('restart_context_guard', bundle / 'context.py')
    context = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context)
    policy = json.loads((bundle / 'stale-context.json').read_bytes())
    policy = policy if hardware['node'] == policy['node'] else None
    def query():
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=15)
        apps = [{'gpu_uuid': row[0].strip(), 'pid': int(row[1]), 'process_name': row[2].strip(), 'used_memory_mib': int(row[3])} for row in csv.reader(io.StringIO(text)) if row]
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15)
        free = {row[0].strip(): int(row[1]) for row in csv.reader(io.StringIO(text)) if row}
        if set(free) != {gpu['uuid'] for gpu in hardware['gpu_inventory']}:
            raise ValueError('Restart allocation physical memory inventory differs')
        blocked = context.blocking_contexts(apps, free, policy, lambda pid: Path('/proc', str(pid)).exists(), context.host_proc_visible())
        native.atomic_json(destination / 'gpu-context.json', {'applications': apps, 'blocking_contexts': blocked, 'free_memory_mib': free, 'policy': policy, 'no_signals_or_resets': True})
        return str(blocked) if blocked else ''
    context.wait_for_idle(query, lambda observations: native.atomic_json(destination / 'idle-check.json', {'observations': observations}))
    plans = {}
    for name, item in prepared['plans'].items():
        path = Path(item['path'])
        if native.digest(path) != item['sha256']:
            raise ValueError('Restart qualification plan changed')
        plans[name] = path
    native.launch(plans['control'])
    native.launch(plans['interrupted'])
    stopped = json.loads((plans['interrupted'].parent / 'launch.json').read_bytes())
    if stopped['status'] != 'checkpointed_for_restart':
        raise ValueError('Interrupted process did not stop at a committed checkpoint')
    native.launch(plans['interrupted'], resume=True)
    rows = []
    for rank in range(8):
        control = json.loads((plans['control'].parent / ('rank-' + str(rank) + '.json')).read_bytes())
        resumed = json.loads((plans['interrupted'].parent / ('rank-' + str(rank) + '.json')).read_bytes())
        recovery = resumed.get('process_recovery') or {}
        cursor = recovery.get('cursor') or {}
        audit = json.loads((plans['interrupted'].parent / 'tokenization-audit.json').read_bytes())
        per_rank = (audit['splits']['train']['rows_after_official_filter'] + 7) // 8
        state_equal = control['final_state_identity'] == resumed['final_state_identity']
        same_batches = journals(plans['control'].parent, rank) == journals(plans['interrupted'].parent, rank)
        passed = control['status'] == resumed['status'] == 'bounded_complete' and control['successful_optimizer_steps'] == resumed['successful_optimizer_steps'] == 2 and recovery.get('status') == 'loaded_exact_state' and resumed.get('restored_rng_and_cursor') is True and state_equal and same_batches and control['final_rng_identity'] == resumed['final_rng_identity'] and 0 < cursor.get('next_batch', 0) < per_rank
        rows.append({'rank': rank, 'passed': passed, 'state_identity_equal': state_equal, 'rng_equal': control['final_rng_identity'] == resumed['final_rng_identity'], 'training_batches_equal': same_batches, 'restored_cursor': cursor, 'microbatches_per_epoch': per_rank})
    report = {'status': 'passed' if all(row['passed'] for row in rows) else 'failed', 'scope': 'actual fresh-process mid-epoch recovery on synthetic native EAGLE-3 engineering fixture', 'prepared_sha256': native.digest(destination / 'prepared.json'), 'native_training_sha256': prepared['native_training_sha256'], 'native_recovery_sha256': prepared['native_recovery_sha256'], 'ranks': rows, 'scientific_training': False, 'training_admission': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    native.atomic_json(destination / 'report.json', report)
    if report['status'] != 'passed':
        raise ValueError('Fresh-process restart qualification did not pass')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'controller'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'controller': controller}[args.action](args.root, args.bundle, args.output)


if __name__ == '__main__':
    main()
