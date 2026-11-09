import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess


UID = 29562
NODE = 'deep-chungus-10'
MAX_STEP_SECONDS = 12 * 60 * 60
LEGACY_ROOT = Path('/mnt/nfs/home/mohamadzbib/projects/rl-infra')


class AttachError(ValueError):
    pass


def duration_seconds(value):
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})', value)
    if not match:
        raise AttachError('Slurm duration is not a finite d-HH:MM:SS value')
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise AttachError('Slurm duration has invalid minute or second fields')
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def step_seconds(job):
    remaining = duration_seconds(job['TimeLimit']) - duration_seconds(job['RunTime'])
    seconds = min(MAX_STEP_SECONDS, remaining - 60)
    if seconds < 120:
        raise AttachError('Holder has insufficient remaining time for a bounded step')
    return seconds


def format_duration(seconds):
    return f'{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


def fields(text):
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise AttachError('Expected one live Slurm job record')
    result = {}
    for item in lines[0].split():
        if '=' not in item:
            continue
        key, value = item.split('=', 1)
        if key in result:
            raise AttachError('Duplicate Slurm job field')
        result[key] = value
    return result


def gpu_count(tres):
    generic = None
    typed = []
    for item in tres.split(','):
        if '=' not in item:
            continue
        key, value = item.split('=', 1)
        if key == 'gres/gpu' or key.startswith('gres/gpu:'):
            if not value.isdigit():
                raise AttachError('Invalid Slurm GPU allocation count')
            if key == 'gres/gpu':
                generic = int(value)
            else:
                typed.append(int(value))
    if generic is not None and typed and generic != sum(typed):
        raise AttachError('Generic and typed Slurm GPU counts disagree')
    return generic if generic is not None else sum(typed) if typed else None


def command_output(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise AttachError('Live Slurm job query failed')
    return result.stdout


def numeric_job_id(value):
    value = str(value)
    if not re.fullmatch(r'[1-9][0-9]*', value):
        raise AttachError('Holder metadata has an invalid job ID')
    return value


def holder_metadata(path):
    value = json.loads(path.read_bytes())
    if value.get('node') != NODE or value.get('gpus') != 8:
        raise AttachError('Holder metadata does not identify the protected eight-GPU node')
    current = numeric_job_id(value['job_id'])
    successor = numeric_job_id(value['successor']) if value.get('successor') else None
    if successor == current:
        raise AttachError('Holder and successor IDs must differ')
    return {'job_id': current, 'successor': successor, 'node': NODE, 'gpus': 8}


def verify_job(job, job_id, pending=False, holder=None):
    owner = re.fullmatch(r'[^()]+\(([0-9]+)\)', job.get('UserId', ''))
    if job.get('JobId') != job_id or not owner or int(owner.group(1)) != UID:
        raise AttachError('Live Slurm job owner or identity differs')
    required = {'JobName': 'deep10-hold', 'Account': 'grad-students', 'Partition': 'high-priority', 'QOS': 'spec-priority', 'JobState': 'PENDING' if pending else 'RUNNING'}
    node_counts = ('1', '1-1') if pending else ('1',)
    if any(job.get(key) != value for key, value in required.items()) or job.get('NumNodes') not in node_counts:
        raise AttachError('Live holder resources, state or account differ')
    if pending:
        if job.get('ReqNodeList') != NODE or job.get('NodeList') not in ('(null)', 'None', NODE):
            raise AttachError('Successor does not request the protected node')
        if not re.fullmatch(r'afterany:' + re.escape(holder) + r'(?:\(unfulfilled\))?', job.get('Dependency', '')):
            raise AttachError('Successor dependency is stale or unexpected')
        tres = job.get('ReqTRES') or job.get('TRES', '')
    else:
        if job.get('NodeList') != NODE or not job.get('NumCPUs', '').isdigit() or int(job['NumCPUs']) < 64:
            raise AttachError('Holder node or CPU allocation differs')
        tres = job.get('AllocTRES') or job.get('TRES', '')
    if gpu_count(tres) != 8:
        raise AttachError('Holder or successor does not own or request exactly eight GPUs')
    return {key: job.get(key) for key in ('JobId', 'JobName', 'UserId', 'JobState', 'Account', 'Partition', 'QOS', 'NodeList', 'ReqNodeList', 'NumNodes', 'NumCPUs', 'AllocTRES', 'ReqTRES', 'Dependency', 'TimeLimit', 'RunTime')}


def process_identity(pid):
    path = Path('/proc') / str(pid)
    try:
        value = path.lstat()
        raw = (path / 'stat').read_text()
        parts = raw[raw.rfind(')') + 2:].split()
        command = (path / 'cmdline').read_bytes()
        return {'pid': pid, 'uid': value.st_uid, 'starttime_ticks': parts[19], 'cmdline_sha256': hashlib.sha256(command).hexdigest()}
    except (FileNotFoundError, ProcessLookupError):
        return {'pid': pid, 'identity_unavailable_process_exited': True}


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_resume(root, run_directory, run_id, config, config_sha256, smoke, selected):
    if run_directory.resolve() != run_directory or not run_directory.is_dir():
        raise AttachError('--resume requires an existing run directory without aliases')
    values, hashes = {}, {}
    for name in ('manifest.json', 'pool.json'):
        path = run_directory / name
        try:
            value = path.lstat()
        except FileNotFoundError as error:
            raise AttachError('--resume requires existing manifest.json and pool.json') from error
        if not stat.S_ISREG(value.st_mode) or path.resolve() != path:
            raise AttachError('Resume metadata must be regular files without aliases')
        contents = path.read_bytes()
        try:
            values[name] = json.loads(contents)
        except (ValueError, UnicodeError) as error:
            raise AttachError('Resume metadata is not valid JSON') from error
        if not isinstance(values[name], dict):
            raise AttachError('Resume metadata must contain JSON objects')
        hashes[name] = hashlib.sha256(contents).hexdigest()
    manifest, state = values['manifest.json'], values['pool.json']
    order = config.get('method_order')
    if not isinstance(order, list) or not order or len(order) != len(set(order)) or any(method not in config['methods'] for method in order):
        raise AttachError('Resume requires an explicit valid configured method order')
    if selected is not None and not set(selected) <= set(order):
        raise AttachError('Resume methods filter differs from the configured method order')
    active = [method for method in order if selected is None or method in selected]
    scope = 'smoke' if smoke else config.get('scope', 'reference')
    if manifest.get('run_id') != run_id or state.get('run_id') != run_id or manifest.get('root') != str(root) or manifest.get('run_dir') != str(run_directory) or manifest.get('config_sha256') != config_sha256:
        raise AttachError('Resume run/root/config provenance differs')
    if manifest.get('scope') != scope or state.get('scope') != scope or manifest.get('active_methods') != active or state.get('active_methods') != active:
        raise AttachError('Resume scope or methods filter differs from the frozen run')
    effective = manifest.get('effective_config')
    if not isinstance(effective, dict) or effective.get('scope') != scope or effective.get('method_order') != active:
        raise AttachError('Resume effective scope or method order differs')
    if state.get('manifest_sha256') != hashes['manifest.json']:
        raise AttachError('Resume pool state does not identify the exact frozen manifest')
    for name in ('scientific_identity_sha256', 'hardware_identity_sha256'):
        if not isinstance(manifest.get(name), str) or not re.fullmatch('[a-f0-9]{64}', manifest[name]) or state.get(name) != manifest[name]:
            raise AttachError('Resume pool/manifest identity differs: ' + name)
    for name, expected in (('root', str(root)), ('config_sha256', config_sha256)):
        if name in state and state[name] != expected:
            raise AttachError('Resume pool provenance differs: ' + name)
    return {'manifest_sha256': hashes['manifest.json'], 'pool_state_sha256_at_validation': hashes['pool.json'], 'existing_pool_status': state.get('status'), 'scope': scope, 'active_methods': active, 'scientific_identity_sha256': manifest['scientific_identity_sha256'], 'hardware_identity_sha256': manifest['hardware_identity_sha256'], 'final_scientific_hardware_input_and_controller_lock_checks': 'delegated_to_existing_pool_before_gpu_work', 'completed_run_behavior': 'Existing pool revalidates frozen provenance and reuses completed worker cases; attach never overwrites the frozen manifest.'}


def launch(root, config_path, run_id, expected_gpus=8, smoke=False, methods=None, detach=False, query=None, now=None, resume=False):
    if os.getuid() != UID:
        raise AttachError('Run this entrypoint as the verified cluster user')
    if expected_gpus != 8 or isinstance(expected_gpus, bool):
        raise AttachError('The protected holder requires all eight GPUs')
    if type(resume) is not bool:
        raise AttachError('resume must be an explicit boolean')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id):
        raise AttachError('Invalid run ID')
    root = Path(root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    if not config_path.is_relative_to(root):
        raise AttachError('Configuration must be inside the selected project root')
    contents = config_path.read_bytes()
    config = json.loads(contents)
    if Path(config['cluster_root']).resolve() != root:
        raise AttachError('Configuration cluster root differs from --root')
    legacy = Path(config.get('legacy_runtime_root', str(LEGACY_ROOT)))
    if not legacy.is_absolute() or not (root / 'src/taps_rebuild/pool.py').is_file():
        raise AttachError('An absolute legacy runtime and prepared pool entrypoint are required')
    selected = None
    if methods is not None:
        selected = [value.strip() for value in methods.split(',')]
        if not selected or any(not value or value not in config['methods'] for value in selected) or len(set(selected)) != len(selected):
            raise AttachError('Methods must be unique configured method names')
    run_directory = root / 'outputs/runs' / run_id
    if run_directory.resolve() != run_directory:
        raise AttachError('Run directory cannot use symlink aliases')
    config_sha256 = hashlib.sha256(contents).hexdigest()
    resume_validation = None
    if resume:
        resume_validation = validate_resume(root, run_directory, run_id, config, config_sha256, smoke, selected)
    elif run_directory.exists() and (not run_directory.is_dir() or any(run_directory.iterdir())):
        raise AttachError('Run ID already has content; choose a fresh ID or explicit --resume to preserve existing results')
    metadata_path = legacy / 'cluster/deep10/current.json'
    metadata = holder_metadata(metadata_path)
    query = command_output if query is None else query
    job = fields(query(['scontrol', 'show', 'job', '-o', metadata['job_id']]))
    verified_holder = verify_job(job, metadata['job_id'])
    verified_successor = None
    if metadata['successor']:
        successor = fields(query(['scontrol', 'show', 'job', '-o', metadata['successor']]))
        verified_successor = verify_job(successor, metadata['successor'], pending=True, holder=metadata['job_id'])
    seconds = step_seconds(job)
    command = ['srun', '--jobid=' + metadata['job_id'], '--overlap', '--gres=none', '--ntasks=1', '--cpus-per-task=64', '--mem=0', '--time=' + format_duration(seconds), '--chdir=' + str(root), 'env', '-u', 'CUDA_VISIBLE_DEVICES', 'PYTHONPATH=' + str(root / 'src'), 'PYTHONUNBUFFERED=1', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', '/usr/bin/python3', '-B', '-m', 'taps_rebuild.pool', '--root', str(root), '--config', str(config_path), '--run-id', run_id, '--expected-gpus', '8']
    if smoke:
        command.append('--smoke')
    if selected is not None:
        command.extend(['--methods', ','.join(selected)])
    if holder_metadata(metadata_path) != metadata or config_path.read_bytes() != contents:
        raise AttachError('Holder identity or configuration changed during launch validation')
    if resume:
        current_resume = validate_resume(root, run_directory, run_id, config, config_sha256, smoke, selected)
        if current_resume['manifest_sha256'] != resume_validation['manifest_sha256']:
            raise AttachError('Frozen resume manifest changed during launch validation')
        resume_validation = current_resume
    now = datetime.now(timezone.utc) if now is None else now
    stamp = now.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root / 'outputs/launchers'
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / (run_id + '-' + stamp + '.json')
    log_path = receipt_path.with_suffix('.log')
    if receipt_path.exists() or receipt_path.is_symlink() or log_path.exists() or log_path.is_symlink():
        raise AttachError('Launch receipt or log already exists; refusing overwrite')
    with receipt_path.open('x'):
        pass
    receipt = {'status': 'launching', 'run_id': run_id, 'root': str(root), 'config': str(config_path), 'config_sha256': config_sha256, 'created_at_utc': now.astimezone(timezone.utc).isoformat(), 'detached': detach, 'resume_requested': resume, 'resume_validation': resume_validation, 'holder': verified_holder, 'successor': verified_successor, 'node': NODE, 'expected_gpus': 8, 'controller_cpus': 64, 'controller_gres': 'none', 'memory_request': '0', 'step_time_limit_seconds': seconds, 'step_time_limit': format_duration(seconds), 'command': command, 'log': str(log_path), 'receipt': str(receipt_path), 'attach_process': process_identity(os.getpid()), 'holder_and_successor_modified': False}
    atomic_json(receipt_path, receipt)
    try:
        with log_path.open('xb') as stream:
            process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=detach)
        receipt.update(status='launched', launcher_pid=process.pid, launcher_process=process_identity(process.pid))
        atomic_json(receipt_path, receipt)
        print(json.dumps({'status': receipt['status'], 'launcher_pid': process.pid, 'receipt': str(receipt_path), 'log': str(log_path)}), flush=True)
        result = process.poll() if detach else process.wait()
        if result is not None:
            receipt.update(status='completed' if result == 0 else 'failed', returncode=result, finished_at_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(receipt_path, receipt)
        return receipt
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__)
        atomic_json(receipt_path, receipt)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/mnt/nfs/home/mohamadzbib/projects/taps-rebuild'))
    parser.add_argument('--config', type=Path, default=Path('configs/reproduction.json'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--expected-gpus', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--methods')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        receipt = launch(args.root, args.config, args.run_id, args.expected_gpus, args.smoke, args.methods, args.detach, resume=args.resume)
    except Exception as error:
        value = {'status': 'failed', 'error_type': type(error).__name__}
        if isinstance(error, AttachError):
            value['error'] = str(error)
        print(json.dumps(value), flush=True)
        raise SystemExit(1) from None
    if receipt.get('returncode', 0) != 0:
        raise SystemExit(receipt['returncode'] if receipt['returncode'] > 0 else 1)


if __name__ == '__main__':
    main()
