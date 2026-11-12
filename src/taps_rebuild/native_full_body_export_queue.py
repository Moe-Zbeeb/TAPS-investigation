import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import native_training as native, pool
from .native_full_body_queue import verify_holder
from .pilot_cache import require


def run(root, plan, receipt, output, attached):
    native.require_slurm()
    if not attached:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Export coordinator requires CPU Slurm')
        job = verify_holder(root)
        command = ['srun', '--jobid=' + job, '--account=grad-students', '--partition=high-priority', '--qos=spec-priority', '--exclusive', '--nodes=1', '--ntasks=1', '--gres=gpu:a100:8', '--cpus-per-task=48', '--mem=0', '--immediate=60', '--kill-on-bad-exit=1', '--job-name=taps-native-export-stock', sys.executable, '-B', '-S', '-m', 'taps_rebuild.native_full_body_export_queue', '--attached', '--root', str(root), '--plan', str(plan), '--receipt', str(receipt), '--output', str(output)]
        environment = dict(os.environ)
        for name in ('CUDA_VISIBLE_DEVICES', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
            environment.pop(name, None)
    else:
        allocation = pool.allocation(8)
        require(allocation['node'] == 'deep-chungus-11', 'Native stock export requires the owned deep11 allocation')
        python = Path(root) / 'envs/native-eagle3/bin/python'
        environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        environment.update(native.cuda_environment(root, allocation['node'], {}))
        environment, _ = native.jit_environment(python, environment)
        command = [str(python), '-B', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=8', '-m', 'taps_rebuild.native_full_body_export', 'stock', '--plan', str(plan), '--receipt', str(receipt), '--output', str(output)]
        pool.atomic_json(Path(output).parent / 'stock-launch.json', {'command': command, 'hardware': allocation, 'controller': {'path': str(Path(__file__).resolve()), 'sha256': pool.digest(__file__)}, 'scope': 'bounded corrected native stock compatibility only'})
    subprocess.run(command, env=environment, check=True)
    return {'status': 'stock_export_execution_complete_requires_independent_review'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attached', action='store_true')
    for key in ('root', 'plan', 'receipt', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.plan, args.receipt, args.output, args.attached)))


if __name__ == '__main__':
    main()
