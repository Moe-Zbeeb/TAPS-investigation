import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_bytes())


def require(value, message):
    if not value:
        raise ValueError(message)


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run(plan_path, expected_sha256, previous):
    require(os.environ.get('SLURM_JOB_ID') and previous.isdigit(), 'Corrected continuation requires Slurm and an exact prior allocation')
    require(hashlib.sha256(plan_path.read_bytes()).hexdigest() == expected_sha256, 'Corrected continuation plan bytes changed')
    plan = read(plan_path)
    require(plan['config']['pipeline_stage'] == 'corrected-full-v4' and plan_path == Path(plan['run_dir']) / 'plan.json' and plan_path.parent.name == 'sharegpt-ultrachat-corrected-full-20261113-v4', 'Continuation is restricted to the fresh corrected13November generalist')
    fields = dict(item.split('=', 1) for item in subprocess.check_output(['scontrol', 'show', 'job', '-o', os.environ['SLURM_JOB_ID']], text=True).split() if '=' in item)
    require(fields.get('JobState') == 'RUNNING' and fields.get('UserId', '').startswith('mohamadzbib(') and fields.get('Account') == 'grad-students' and (fields.get('Partition'), fields.get('QOS')) in {('high-priority', 'spec-priority'), ('low-priority', 'normal')} and fields.get('NodeList') == 'deep-chungus-11' and 'gres/gpu=8' in fields.get('TRES', ''), 'Continuation requires the owned full eight-GPU deep11 allocation in an authorized scheduling class')
    usage = subprocess.check_output(['squeue', '-h', '-u', 'mohamadzbib', '-p', 'high-priority', '-t', 'RUNNING', '-o', '%b'], text=True)
    require(sum(int(value) for value in re.findall(r'gpu(?::[A-Za-z0-9_.-]+)?:(\d+)', usage)) <= 18, 'Continuation exceeds the total high-priority GPU allowance')
    environment = dict(os.environ, TZ='UTC', SLURM_TIME_FORMAT='standard')
    command = ['sacct', '-X', '-n', '-P', '-j', previous, '-o', 'JobIDRaw,State,User,JobName%40,End']
    rows = []
    for attempt in range(20):
        raw = subprocess.check_output(command, env=environment, text=True)
        rows = [line.split('|') for line in raw.splitlines() if line.split('|')[0] == previous]
        if len(rows) == 1 and rows[0][1].split()[0].rstrip('+') in ('TIMEOUT', 'PREEMPTED', 'NODE_FAIL'):
            break
        if attempt < 19:
            time.sleep(2)
    require(len(rows) == 1 and rows[0][1].split()[0].rstrip('+') in ('TIMEOUT', 'PREEMPTED', 'NODE_FAIL') and rows[0][2:4] == ['mohamadzbib', 'taps-deep11-hold'], 'Prior owned deep11 holder did not end in a retryable allocation state')
    directory = plan_path.parent
    launch_path = directory / 'launch.json'
    launch = read(launch_path)
    require(launch['plan_identity_sha256'] == plan['plan_identity_sha256'] and launch['slurm_job_id'] == previous, 'Continuation launch belongs to another plan or allocation')
    attempt = launch.get('attempt')
    require(type(attempt) is int and attempt > 0, 'Continuation prior attempt is not identified')
    rank_evidence = {}
    for rank in range(8):
        path = directory / ('rank-' + str(rank) + '.json')
        value = read(path)
        require(value.get('rank') == rank and value.get('world_size') == 8 and value.get('plan_identity_sha256') == plan['plan_identity_sha256'] and value.get('slurm_job_id') == previous and value.get('status') in ('running', 'native_epochs_complete', 'checkpointed_for_restart') and not value.get('error'), 'Prior worker receipt is failed, missing or belongs to another plan/allocation')
        rank_evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        body_path = directory / 'attempts' / ('attempt-' + str(attempt)) / ('full-body-rank-' + str(rank) + '.json')
        if body_path.exists():
            body = read(body_path)
            require(body.get('rank') == rank and body.get('plan_identity_sha256') == plan['plan_identity_sha256'] and body.get('source_sha256') == plan['config']['full_body_v3']['source_sha256'] and body.get('status') == 'completed_corrected_native_execution_requires_independent_evidence' and not body.get('error'), 'Prior full-body worker recorded a failure or changed binding')
            rank_evidence[str(body_path)] = hashlib.sha256(body_path.read_bytes()).hexdigest()
    output = directory / 'continuation'
    output.mkdir(exist_ok=True)
    receipt = {'plan_path': str(plan_path), 'plan_sha256': expected_sha256, 'previous_allocation': previous, 'prior_attempt': attempt, 'prior_worker_receipt_sha256': rank_evidence, 'allocation_accounting': raw, 'new_allocation': os.environ['SLURM_JOB_ID'], 'scheduling': {name: fields[name] for name in ('Account', 'Partition', 'QOS')}, 'old_generalist_never_resumed': True, 'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    if launch['status'] == 'complete':
        atomic(output / 'already-complete.json', {**receipt, 'status': 'training_already_complete_evaluation_required'})
        return
    latest = read(Path(plan['trained_models']['resolved_path']) / 'recovery/latest.json')
    checkpoint = Path(latest['path'])
    require(checkpoint.name.startswith('attempt-' + str(attempt) + '-step-'), 'Committed checkpoint belongs to another attempt')
    require(checkpoint.resolve() == checkpoint and checkpoint.parent == Path(plan['trained_models']['resolved_path']) / 'recovery' and latest['plan_identity_sha256'] == plan['plan_identity_sha256'] and hashlib.sha256((checkpoint / 'manifest.json').read_bytes()).hexdigest() == latest['manifest_sha256'], 'Continuation lacks an exact committed checkpoint from the fresh run')
    manifest = read(checkpoint / 'manifest.json')
    require(manifest['plan_identity_sha256'] == plan['plan_identity_sha256'] and manifest['world_size'] == 8 and manifest['global_step'] > 0 and manifest['hardware_identity_sha256'] == launch['hardware']['hardware_identity_sha256'], 'Continuation checkpoint plan or physical hardware binding differs')
    if launch['status'] == 'running':
        atomic(output / ('unclosed-launch-' + previous + '.json'), launch)
        launch = {**launch, 'status': 'failed', 'terminal_state_reconciliation': {'sacct': raw, 'reason': 'Exact prior allocation ended before its launcher receipt closed'}}
        atomic(launch_path, launch)
    else:
        require(launch['status'] == 'failed' and 'finished_at' in launch, 'Continuation refuses a nonfailed prior attempt')
        require(launch.get('error_type') == 'NativeTrainingError' and launch.get('error', '').startswith('Native launcher interrupted by signal '), 'Continuation refuses a numerical, data or other noninterruption training failure')
        end = datetime.fromisoformat(rows[0][4]).replace(tzinfo=timezone.utc)
        finished = datetime.fromisoformat(launch['finished_at'].replace('Z', '+00:00'))
        require(abs((finished - end).total_seconds()) <= 120, 'Prior attempt failed before allocation expiry; automatic scientific retry refused')
    bridge = directory / 'inputs/native_full_body_bridge.py'
    require(hashlib.sha256(bridge.read_bytes()).hexdigest() == plan['frozen_files']['inputs/native_full_body_bridge.py'], 'Frozen continuation bridge source changed')
    atomic(output / ('resume-' + os.environ['SLURM_JOB_ID'] + '.json'), {**receipt, 'status': 'starting_exact_committed_checkpoint_resume', 'checkpoint': latest})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    subprocess.run([str(Path(plan['project_root']) / plan['config']['runtime_python']), '-B', str(bridge), 'resume', '--plan', str(plan_path)], env=env, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--previous-allocation', required=True)
    args = parser.parse_args()
    run(args.plan, args.plan_sha256, args.previous_allocation)
