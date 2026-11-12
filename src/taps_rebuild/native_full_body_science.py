import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from . import native_full_body as corrected, native_training as native
from .native_full_body_queue import descriptor, verify_holder
from .pilot_cache import read, require


def prepare(root, preparation, approval, run_id):
    native.require_slurm()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Corrected full preparation requires CPU Slurm')
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id), 'Corrected scientific run ID must be a simple directory name')
    queue = read(Path(preparation) / 'queue-plan.json')
    require(queue['source_sha256'] == corrected.sources(), 'Corrected full controller source differs from qualified source')
    profile_path = Path(queue['plans']['profile']['path'])
    require(native.digest(profile_path) == queue['plans']['profile']['sha256'], 'Corrected profile plan changed')
    profile = read(profile_path)
    require(read(profile_path.parent / 'launch.json')['status'] == 'complete', 'Corrected full-data profile has not completed')
    config = {**profile['config'], 'pipeline_stage': 'corrected-full-v3', 'budget': {'optimizer_steps': None}, 'recovery': {'checkpoint_interval': 1000, 'stop_after_optimizer_step': None}, 'qualification': 'Fresh corrected scratch generalist; forty native epochs; independent scientific admission required', 'full_body_v3': {**profile['config']['full_body_v3'], 'maximum_attempted_optimizer_steps': None, 'capture_gradient_partitions': False, 'scientific_admission': descriptor(approval)}}
    output = Path(root) / 'outputs/native-full-body-science' / run_id
    require(not output.exists(), 'Corrected scientific preparation already exists')
    output.mkdir(parents=True)
    native.atomic_json(output / 'config.json', config)
    plan = corrected.prepare(root, output / 'config.json', profile['train']['frozen_path'], profile['validation']['frozen_path'], Path(root) / 'outputs/native-training' / run_id)
    status = {'status': 'prepared_fresh_corrected_scientific_generalist', 'plan': descriptor(Path(plan['run_dir']) / 'plan.json'), 'approval': descriptor(approval), 'source_sha256': corrected.sources(), 'controller': descriptor(__file__), 'old_generalist_never_resumed': True}
    native.atomic_json(output / 'preparation.json', status)
    return status


def run(root, preparation, approval, run_id, attached=False):
    if attached:
        status = read(Path(root) / 'outputs/native-full-body-science' / run_id / 'preparation.json')
        require(status['source_sha256'] == corrected.sources() and native.digest(status['plan']['path']) == status['plan']['sha256'], 'Corrected scientific prepared plan changed')
        require(status['controller'] == descriptor(__file__) and status['approval'] == descriptor(approval), 'Corrected scientific controller or approval content changed')
        path = Path(status['plan']['path'])
        command = [str(Path(root) / 'envs/native-eagle3/bin/python'), '-B', str(path.parent / 'inputs/native_full_body.py'), 'launch', '--plan', str(path)]
        subprocess.run(command, check=True)
        return
    native.require_slurm()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Corrected scientific coordinator requires CPU Slurm')
    deadline = time.monotonic() + 86400
    while not Path(approval).is_file():
        native.atomic_json(Path(preparation) / 'scientific-queue-status.json', {'status': 'waiting_actual_independent_native_full_body_admission', 'gpu_step_started': False, 'old_generalist_never_resumed': True})
        require(time.monotonic() < deadline, 'Corrected native scientific admission wait expired')
        time.sleep(30)
    status = prepare(root, preparation, approval, run_id)
    job = verify_holder(root)
    command = ['srun', '--jobid=' + job, '--account=grad-students', '--partition=high-priority', '--qos=spec-priority', '--exclusive', '--nodes=1', '--ntasks=1', '--gres=gpu:a100:8', '--cpus-per-task=48', '--mem=0', '--immediate=60', '--kill-on-bad-exit=1', '--job-name=taps-corrected-generalist', sys.executable, '-B', '-S', '-m', 'taps_rebuild.native_full_body_science', '--attached', '--root', str(root), '--preparation', str(preparation), '--approval', str(approval), '--run-id', run_id]
    environment = dict(os.environ)
    for name in ('CUDA_VISIBLE_DEVICES', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
        environment.pop(name, None)
    native.atomic_json(Path(root) / 'outputs/native-full-body-science' / run_id / 'handoff.json', {'command': command, 'holder_job_id': job, 'preparation': status})
    subprocess.run(command, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attached', action='store_true')
    for key in ('root', 'preparation', 'approval'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run(args.root, args.preparation, args.approval, args.run_id, args.attached)


if __name__ == '__main__':
    main()
