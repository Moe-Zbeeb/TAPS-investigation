import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


RETRYABLE = {'TIMEOUT', 'PREEMPTED', 'NODE_FAIL'}


def read(path):
    return json.loads(Path(path).read_bytes())


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def completed_job(text, job_id):
    rows = [line.split('|') for line in text.splitlines() if line.split('|')[0] == str(job_id)]
    if len(rows) != 1 or len(rows[0]) != 5 or rows[0][3] != 'mohamadzbib' or rows[0][4] not in ('taps-full-train', 'taps-full-resume'):
        raise ValueError('Continuation requires the exact owned full-training job')
    state = rows[0][1].split()[0].rstrip('+')
    if state not in RETRYABLE | {'COMPLETED', 'FAILED', 'CANCELLED', 'OUT_OF_MEMORY'}:
        raise ValueError('Prior training allocation is not terminal')
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--previous-job', required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not args.previous_job.isdigit():
        raise ValueError('Training continuation runs only in CPU Slurm')
    schedule = read(args.schedule)
    root = Path(schedule['root'])
    bundle = args.schedule.parent
    identity = read(bundle / 'bundle.json')
    for name, expected in identity['files'].items():
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen training continuation source changed')
    config = read(root / schedule['config'])
    directory = root / 'outputs/training-handoff' / config['run_id']
    chain = directory / 'chain'
    chain.mkdir(parents=True, exist_ok=True)
    lock = (chain / 'continuation.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt_path = chain / ('after-' + args.previous_job + '.json')
    if receipt_path.exists() and read(receipt_path)['status'] in ('resubmitted', 'training_complete_evaluation_required', 'stopped_nonretryable_state', 'stopped_continuation_limit'):
        print(json.dumps(read(receipt_path), indent=2))
        return
    output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', args.previous_job, '-o', 'JobIDRaw,State,ExitCode,User,JobName%40'], text=True, timeout=30)
    state = completed_job(output, args.previous_job)
    receipt = {'previous_job': args.previous_job, 'state': state, 'attempt': args.attempt, 'sacct': output, 'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    if (directory / 'completion.json').exists():
        receipt['status'] = 'training_complete_evaluation_required'
    elif state not in RETRYABLE:
        receipt['status'] = 'stopped_nonretryable_state'
    elif args.attempt > 20:
        receipt['status'] = 'stopped_continuation_limit'
    else:
        handoff = read(directory / 'status.json')
        plan_path = Path(handoff['native_plan'])
        plan = read(plan_path)
        latest = Path(plan['trained_models']['resolved_path']) / 'recovery/latest.json'
        if handoff['status'] != 'passed' or not latest.is_file():
            raise ValueError('No admitted full plan and committed checkpoint are available')
        previous = read(plan_path.parent / 'launch.json')
        if previous['slurm_job_id'] != args.previous_job:
            raise ValueError('Another training attempt already owns the run')
        if previous['status'] == 'running':
            atomic(chain / ('unclosed-launch-' + args.previous_job + '.json'), previous)
            previous.update(status='failed', terminal_state_reconciliation={'sacct': output, 'reason': 'Prior allocation ended before launcher receipt closed'})
            atomic(plan_path.parent / 'launch.json', previous)
        if previous['status'] != 'failed':
            raise ValueError('Training continuation refuses a completed or still-active attempt')
        command = list(schedule['resume_command'])
        usage = subprocess.check_output(['squeue', '-h', '-u', 'mohamadzbib', '-p', 'high-priority', '-t', 'RUNNING', '-o', '%b'], text=True, timeout=30)
        total = sum(sum(int(value) for value in re.findall(r'gpu(?::[A-Za-z0-9_.-]+)?:(\d+)', line)) for line in usage.splitlines())
        if total > 10:
            node = subprocess.check_output(['scontrol', 'show', 'node', '-o', 'deep-chungus-11'], text=True, timeout=30)
            if 'State=IDLE ' in node and not re.search(r'AllocTRES=\S*(?:cpu=[1-9]|gres/gpu=[1-9])', node):
                command = ['--partition=low-priority' if value == '--partition=high-priority' else '--qos=normal' if value == '--qos=spec-priority' else value for value in command]
                receipt['scheduling_override'] = 'Authorized normal priority on the idle same-geometry node; high-priority total would exceed 18'
            else:
                followup = list(schedule['followup_command'])
                followup.extend(['--begin=now+5minutes', '--wrap', schedule['followup_wrap'] + ' --previous-job ' + args.previous_job + ' --attempt ' + str(args.attempt)])
                watcher = subprocess.run(followup, capture_output=True, text=True, check=True).stdout.strip().split(';')[0]
                receipt.update(status='waiting_for_capacity', high_priority_gpus=total, followup_job=watcher)
                atomic(receipt_path, receipt)
                return
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        job = result.stdout.strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Slurm did not return a valid continuation job')
        verified = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True, timeout=30)
        followup = list(schedule['followup_command'])
        followup.extend(['--dependency=afterany:' + job, '--wrap', schedule['followup_wrap'] + ' --previous-job ' + job + ' --attempt ' + str(args.attempt + 1)])
        watcher = subprocess.run(followup, capture_output=True, text=True, check=True).stdout.strip().split(';')[0]
        receipt.update(status='resubmitted', training_job=job, followup_job=watcher, command=command, verified_slurm=verified)
    atomic(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
