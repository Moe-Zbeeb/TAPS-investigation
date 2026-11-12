import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from . import native_training as native, pool
from .pilot_cache import read, require


def holder(root):
    authority = read(Path(root).parent / 'rl-infra/cluster/deep10/current.json')
    job = pool._fields(pool._command(['scontrol', 'show', 'job', '-o', authority['job_id']]))
    require(authority['node'] == 'deep-chungus-10' and authority['gpus'] == 8 and job.get('JobName') == 'deep10-hold' and job.get('NodeList') == authority['node'] and job.get('JobState') == 'RUNNING' and job.get('UserId', '').endswith('(' + str(os.getuid()) + ')') and job.get('Account') == 'grad-students' and job.get('Partition') == 'high-priority' and job.get('QOS') == 'spec-priority' and pool._gpu_count(job.get('AllocTRES') or job.get('TRES', '')) == 8, 'Protected holder authority differs from live Slurm')
    return authority['job_id']


def verify_bundle():
    bundle = Path(__file__).parents[2]
    manifest = read(bundle / 'bundle.json')
    require(all(pool.digest(bundle / name) == expected for name, expected in manifest['files'].items()), 'Frozen pilot bundle source changed')
    return bundle, manifest


def attached(root, cache, output, config):
    bundle, manifest = verify_bundle()
    hardware = pool.allocation(8)
    require(hardware['node'] == 'deep-chungus-10', 'Pilot scan requires the registered eight-GPU node')
    python = Path(root) / 'envs/native-eagle3/bin/python'
    environment = dict(os.environ, PYTHONPATH=str(bundle / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment.update(native.cuda_environment(root, hardware['node'], {}))
    environment, tools = native.jit_environment(python, environment)
    command = [str(python), '-B', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=8', '-m', 'taps_rebuild.pilot_gate', 'worker', '--root', str(root), '--cache', str(cache), '--output', str(output)]
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pool.atomic_json(output / 'launch.json', {'command': command, 'hardware': hardware, 'tools': tools, 'bundle': manifest, 'scope': 'target vocabulary eligibility only', 'training_admitted': False})
        subprocess.run(command, env=environment, check=True)


def launch(root, cache, output, config):
    bundle, manifest = verify_bundle()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Pilot coordinator requires CPU Slurm')
    from .pilot_gate import verify_cache
    verify_cache(cache)
    job = holder(root)
    command = ['srun', '--jobid=' + job, '--account=grad-students', '--partition=high-priority', '--qos=spec-priority', '--exclusive', '--nodes=1', '--ntasks=1', '--gres=gpu:a100:8', '--cpus-per-task=48', '--mem=0', '--immediate=60', '--kill-on-bad-exit=1', '--job-name=taps-pilot-gate', '/usr/bin/python3', '-B', '-S', '-m', 'taps_rebuild.pilot_queue', 'attached', '--root', str(root), '--cache', str(cache), '--output', str(output), '--config', str(config)]
    environment = dict(os.environ, PYTHONPATH=str(bundle / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1')
    environment.pop('CUDA_VISIBLE_DEVICES', None)
    for name in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
        environment.pop(name, None)
    subprocess.run(command, env=environment, check=True)
    python = Path(root) / 'envs/native-eagle3/bin/python'
    environment['CUDA_VISIBLE_DEVICES'] = ''
    subprocess.run([str(python), '-B', '-m', 'taps_rebuild.pilot_gate', 'finalize', '--root', str(root), '--cache', str(cache), '--output', str(output), '--config', str(config)], env=environment, check=True)
    return read(output / 'completion.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('launch', 'attached'))
    for name in ('root', 'cache', 'output', 'config'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps((launch if args.action == 'launch' else attached)(args.root, args.cache, args.output, args.config), indent=2))


if __name__ == '__main__':
    main()
