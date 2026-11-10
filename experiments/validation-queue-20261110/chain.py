import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--expected-gpus', type=int, required=True)
    parser.add_argument('--role', choices=['primary', 'screen', 'h100'], required=True)
    args = parser.parse_args()
    job = os.environ['SLURM_JOB_ID']
    prefix = args.root / 'outputs/validation-pipelines' / (args.role + '-' + job + '-' + args.bundle.name)
    common = [sys.executable, '-B', str(args.bundle / 'driver.py'), '--action', 'controller', '--root', str(args.root), '--prepared', str(args.prepared), '--expected-gpus', str(args.expected_gpus)]
    subprocess.run(common + ['--output', str(prefix) + '-smoke', '--phases', 'conditional,sampling', '--conditional-trials', '16', '--sampling-repeats', '2'], check=True)
    phases = {'primary': 'greedy,conditional', 'screen': 'sampling', 'h100': 'greedy'}[args.role]
    subprocess.run(common + ['--output', str(prefix) + '-full', '--phases', phases], check=True)
    if args.role == 'screen':
        environment = dict(os.environ, PYTHONPATH=str(args.root / 'src'))
        command = [sys.executable, '-B', '-m', 'taps_rebuild.pool', '--root', str(args.root), '--config', str(args.bundle / 'replication.json'), '--run-id', 'official-eagle3-replication-20261110-a10080-v1', '--expected-gpus', str(args.expected_gpus), '--methods', 'eagle3']
        subprocess.run(command, env=environment, check=True)
    print(json.dumps({'chain_complete': args.role, 'job_id': job, 'training_started': False}), flush=True)


if __name__ == '__main__':
    main()
