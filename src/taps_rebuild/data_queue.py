import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from . import corpus_generate, pool


def holder(root):
    authority = Path(root).parent / 'rl-infra/cluster/deep10/current.json'
    value = json.loads(authority.read_bytes())
    job = pool._fields(pool._command(['scontrol', 'show', 'job', '-o', value['job_id']]))
    owner = str(os.getuid())
    corpus_generate.check(value['node'] == 'deep-chungus-10' and value['gpus'] == 8 and job.get('JobName') == 'deep10-hold' and job.get('NodeList') == value['node'] and job.get('JobState') == 'RUNNING' and job.get('UserId', '').endswith('(' + owner + ')') and job.get('Account') == 'grad-students' and job.get('Partition') == 'high-priority' and job.get('QOS') == 'spec-priority' and pool._gpu_count(job.get('AllocTRES') or job.get('TRES', '')) == 8, 'Protected holder authority differs from live Slurm')
    return value['job_id']


def gpu(root, smoke, full):
    hardware = pool.allocation(8)
    corpus_generate.run(smoke)
    receipt = corpus_generate.read(Path(smoke).parent / 'completion.json')
    corpus_generate.check(receipt['scope'] == 'teacher_generation_smoke' and receipt['counts'].get('accepted:train', 0) >= 1, 'Real-data teacher generation smoke did not produce a complete conversation')
    pool.atomic_json(Path(full).parent / 'gpu-admission.json', {'status': 'passed', 'scope': 'target_response_generation_only', 'smoke_completion_sha256': pool.digest(Path(smoke).parent / 'completion.json'), 'hardware': hardware, 'training_admission': False, 'created_at_utc': datetime.now(timezone.utc).isoformat()})
    return corpus_generate.run(full)


def attached(root, config, corpus, run_id):
    root = Path(root).resolve()
    pool.allocation(8)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        smoke = corpus_generate.prepare(root, config, corpus, run_id + '-smoke', 64)
        full = corpus_generate.prepare(root, config, corpus, run_id)
    finally:
        if visible is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = visible
    smoke_path, full_path = Path(smoke['directory']) / 'plan.json', Path(full['directory']) / 'plan.json'
    return gpu(root, smoke_path, full_path)


def launch(root, config, corpus, run_id):
    root = Path(root).resolve()
    job = holder(root)
    source = Path(__file__).parents[1]
    command = ['srun', '--jobid=' + job, '--exclusive', '--nodes=1', '--ntasks=1', '--gres=gpu:a100:8', '--cpus-per-task=48', '--kill-on-bad-exit=1', '--job-name=taps-teacher-data', '/usr/bin/python3', '-B', '-S', '-m', 'taps_rebuild.data_queue', 'attached', '--root', str(root), '--config', str(config), '--corpus', str(corpus), '--run-id', run_id]
    pool.atomic_json(root / 'outputs/launchers' / (run_id + '-handoff.json'), {'status': 'waiting_for_owned_gpu_step', 'holder_job_id': job, 'command': command, 'source_root': str(source), 'source_sha256': pool.digest(__file__), 'training_admission': False})
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1')
    environment.pop('CUDA_VISIBLE_DEVICES', None)
    for name in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
        environment.pop(name, None)
    subprocess.run(command, env=environment, check=True)
    return corpus_generate.read(root / 'outputs/teacher-generation' / run_id / 'completion.json')


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('launch', 'attached'):
        launch_parser = actions.add_parser(action)
        launch_parser.add_argument('--root', type=Path, required=True)
        launch_parser.add_argument('--config', type=Path, required=True)
        launch_parser.add_argument('--corpus', type=Path, required=True)
        launch_parser.add_argument('--run-id', required=True)
    gpu_parser = actions.add_parser('gpu')
    gpu_parser.add_argument('--root', type=Path, required=True)
    gpu_parser.add_argument('--smoke', type=Path, required=True)
    gpu_parser.add_argument('--full', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'gpu':
        result = gpu(args.root, args.smoke, args.full)
    else:
        operation = launch if args.action == 'launch' else attached
        result = operation(args.root, args.config, args.corpus, args.run_id)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
