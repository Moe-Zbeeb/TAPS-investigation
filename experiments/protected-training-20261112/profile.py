import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from taps_rebuild import admission, attach, data_queue, pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    admission.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Profile attachment requires CPU Slurm')
    manifest = admission.read(args.bundle / 'bundle.json')
    for name, expected in manifest['files'].items():
        admission.require(pool.digest(args.bundle / name) == expected, 'Qualified frozen profile bundle changed')
    config = admission.read(args.bundle / 'configs/training-queue-pipeline-v2.json')
    status = admission.read(args.root / 'outputs/training-queue' / config['run_id'] / 'status.json')
    admission.require(status['status'] == 'passed' and status['config_sha256'] == pool.digest(args.bundle / 'configs/training-queue-pipeline-v2.json') and pool.digest(status['native_plan']) == status['native_plan_sha256'], 'Real-data profile admission differs')
    job_id = data_queue.holder(args.root)
    job = pool._fields(subprocess.check_output(['scontrol', 'show', 'job', '-o', job_id], text=True))
    remaining = attach.duration_seconds(job['TimeLimit']) - attach.duration_seconds(job['RunTime']) - 60
    admission.require(remaining >= 8 * 3600, 'Profile requires eight hours remaining on the current protected holder')
    command = ['srun', '--jobid=' + job_id, '--exclusive', '--nodes=1', '--ntasks=1', '--gres=gpu:a100:8', '--cpus-per-task=48', '--kill-on-bad-exit=1', '--time=07:50:00', '--job-name=taps-native-profile', '/bin/bash', str(args.bundle / 'profile.sh')]
    receipt = {'status': 'launching', 'cpu_job_id': os.environ['SLURM_JOB_ID'], 'holder_job_id': job_id, 'command': command, 'bundle_sha256': pool.digest(args.bundle / 'bundle.json'), 'native_plan_sha256': status['native_plan_sha256'], 'holder_and_successor_modified': False, 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    pool.atomic_json(args.receipt, receipt)
    environment = dict(os.environ)
    for key in ('CUDA_VISIBLE_DEVICES', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
        environment.pop(key, None)
    result = subprocess.run(command, env=environment)
    receipt.update(status='complete' if result.returncode == 0 else 'failed', returncode=result.returncode, finished_at_utc=datetime.now(timezone.utc).isoformat())
    pool.atomic_json(args.receipt, receipt)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
