import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from . import corpus_generate, data_queue, pool


RECOVERABLE = {'TIMEOUT', 'PREEMPTED', 'NODE_FAIL'}


def job_record(job):
    try:
        return pool._fields(pool._command(['scontrol', 'show', 'job', '-o', str(job)]))
    except pool.PoolError:
        text = pool._command(['sacct', '-j', str(job), '--noheader', '--parsable2', '--format=JobIDRaw,State%40'])
        rows = [line.split('|') for line in text.splitlines() if line.split('|')[0] == str(job)]
        corpus_generate.check(len(rows) == 1, 'Cannot classify the previous Slurm job')
        return {'JobState': rows[0][1].split()[0].rstrip('+'), 'Restarts': '0'}


def allow_resume(holder_state, coordinator_state, restarts):
    if holder_state == 'CANCELLED' or coordinator_state == 'CANCELLED':
        return False
    return holder_state in RECOVERABLE or coordinator_state in RECOVERABLE or restarts > 0


def completed(plan):
    path = Path(plan['directory']) / 'completion.json'
    if not path.exists():
        return None
    value = corpus_generate.read(path)
    corpus_generate.check(value['status'] == 'complete' and value['plan_identity_sha256'] == plan['identity_sha256'] and value['scope'] == 'full_training_data_preparation', 'Completion belongs to another teacher plan')
    output = Path(value['corpus_output'])
    corpus_generate.check(all((output / name).stat().st_size == item['size_bytes'] for name, item in value['files'].items()), 'Completed target corpus inventory changed')
    return value


def run(root, plan_path, coordinator):
    root, plan_path = Path(root).resolve(), Path(plan_path).resolve()
    corpus_generate.check(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Data continuation requires CPU Slurm')
    plan = corpus_generate.load_plan(plan_path)
    directory = Path(plan['directory'])
    lock = (directory / 'continuation.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {'status': 'checking_completion', 'plan_identity_sha256': plan['identity_sha256'], 'training_admission': False, 'attempts': []}
    try:
        if completed(plan) is not None:
            status['status'] = 'complete'
            return status
        attempts = sorted(directory.glob('attempt-*.json'), key=lambda path: path.stat().st_mtime_ns)
        corpus_generate.check(attempts, 'No GPU attempt exists to classify for continuation')
        previous = corpus_generate.read(attempts[-1])
        corpus_generate.check(previous['plan_identity_sha256'] == plan['identity_sha256'], 'Previous GPU attempt belongs to another plan')
        previous_job = previous['hardware']['job_id']
        original = job_record(coordinator)
        recovery = allow_resume(job_record(previous_job).get('JobState'), original.get('JobState'), int(original.get('Restarts', '0')))
        while recovery:
            deadline = time.monotonic() + 3600
            while True:
                try:
                    current = data_queue.holder(root)
                    break
                except (corpus_generate.GenerationError, pool.PoolError):
                    corpus_generate.check(time.monotonic() < deadline, 'No healthy successor holder became available')
                    time.sleep(30)
            command = ['srun', '--jobid=' + current, '--exclusive', '--nodes=1', '--ntasks=1', '--gres=gpu:a100:8', '--cpus-per-task=48', '--kill-on-bad-exit=1', '--job-name=taps-teacher-resume', '/usr/bin/python3', '-B', '-S', '-m', 'taps_rebuild.corpus_generate', 'run', '--plan', str(plan_path)]
            environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
            for name in ('CUDA_VISIBLE_DEVICES', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
                environment.pop(name, None)
            status['status'] = 'resuming'
            pool.atomic_json(directory / 'continuation.json', status)
            result = subprocess.run(command, env=environment)
            status['attempts'].append({'holder_job_id': current, 'returncode': result.returncode, 'finished_at_utc': datetime.now(timezone.utc).isoformat()})
            if result.returncode == 0 and completed(plan) is not None:
                status['status'] = 'complete'
                return status
            recovery = job_record(current).get('JobState') in RECOVERABLE
        raise corpus_generate.GenerationError('Data computation failed or was cancelled; automatic retry is prohibited')
    except Exception as error:
        status.update(status='blocked', reason=str(error), error_type=type(error).__name__)
        raise
    finally:
        status['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        pool.atomic_json(directory / 'continuation.json', status)
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--coordinator-job', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.plan, args.coordinator_job), indent=2))


if __name__ == '__main__':
    main()
