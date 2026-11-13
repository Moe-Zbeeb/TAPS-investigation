import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run(plan_path, output, attached, restore_teacher_buffers=False, restore_preinitialization_buffers=False):
    plan = json.loads(Path(plan_path).read_bytes())
    sys.path.insert(0, str(Path(plan['source_directory']).parent))
    from taps_rebuild import pool
    from taps_rebuild.pilot_cache import require
    from taps_rebuild.pilot_queue import holder

    require(os.environ.get('SLURM_JOB_ID'), 'Diagnostic coordinator requires Slurm')
    output = Path(output)
    if attached:
        from taps_rebuild.pilot_pipeline import environment
        hardware = pool.allocation(8)
        require(hardware['node'] == 'deep-chungus-10', 'Diagnostic requires the registered pilot node')
        python, env, tools = environment(plan['project_root'], hardware['node'], Path(plan['source_directory']).parents[1])
        command = [str(python), '-B', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=8', str(Path(__file__).with_name('pilot_teacher_gate_probe_20261113.py')), '--plan', str(plan_path), '--output', str(output)]
        if restore_teacher_buffers:
            command.append('--restore-teacher-buffers')
        if restore_preinitialization_buffers:
            env['PYTHONPATH'] = str(Path(__file__).parent / 'src')
            command.append('--restore-preinitialization-buffers')
        output.mkdir(parents=True, exist_ok=True)
        pool.atomic_json(output / 'launch.json', {'scope': 'Engineering-only exact failed-step teacher gate diagnosis; zero optimizer updates', 'command': command, 'hardware': hardware, 'tools': tools, 'launcher_sha256': pool.digest(__file__), 'probe_sha256': pool.digest(Path(__file__).with_name('pilot_teacher_gate_probe_20261113.py'))})
        subprocess.run(command, env=env, check=True)
        require(all((output / ('rank-' + str(rank) + '.json')).is_file() for rank in range(8)), 'Diagnostic rank coverage incomplete')
        pool.atomic_json(output / 'completion.json', {'status': 'completed_eight_rank_engineering_probe', 'scientific_training_admitted': False, 'scientific_run_resumed': False, 'optimizer_updates': 0, 'ranks': [{'rank': rank, 'sha256': pool.digest(output / ('rank-' + str(rank) + '.json'))} for rank in range(8)]})
        return
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Diagnostic coordinator must be CPU-only')
    job = holder(plan['project_root'])
    command = ['srun', '--jobid=' + job, '--account=grad-students', '--partition=high-priority', '--qos=spec-priority', '--exclusive', '--nodes=1', '--ntasks=1', '--gres=gpu:a100:8', '--cpus-per-task=48', '--mem=0', '--time=00:40:00', '--immediate=60', '--kill-on-bad-exit=1', '--job-name=taps-teacher-gate-probe', '/usr/bin/python3', '-B', '-S', str(Path(__file__).resolve()), '--plan', str(plan_path), '--output', str(output), '--attached']
    if restore_teacher_buffers:
        command.append('--restore-teacher-buffers')
    if restore_preinitialization_buffers:
        command.append('--restore-preinitialization-buffers')
    env = dict(os.environ)
    for name in ('CUDA_VISIBLE_DEVICES', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
        env.pop(name, None)
    subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attached', action='store_true')
    parser.add_argument('--restore-teacher-buffers', action='store_true')
    parser.add_argument('--restore-preinitialization-buffers', action='store_true')
    args = parser.parse_args()
    run(args.plan, args.output, args.attached, args.restore_teacher_buffers, args.restore_preinitialization_buffers)
