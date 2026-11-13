import argparse
import os
from pathlib import Path
import subprocess
import sys


def run(root, study, attached):
    bundle = Path(__file__).resolve().parent
    sys.path.insert(0, str(bundle / 'src'))
    from taps_rebuild import pool
    from taps_rebuild.pilot_cache import read, require
    from taps_rebuild.pilot_pipeline import environment, execute, qualify_attached, study_manifest
    from taps_rebuild.pilot_queue import holder, verify_bundle

    require(os.environ.get('SLURM_JOB_ID'), 'Teacher policy qualification requires Slurm')
    require(verify_bundle()[0] == bundle, 'Teacher policy qualification bundle differs')
    manifest = study_manifest(study)
    if attached:
        hardware = pool.allocation(8)
        require(hardware['node'] == 'deep-chungus-10', 'Teacher policy qualification requires the registered eight-GPU node')
        matches = [item for item in manifest['plans'] if (lambda value: value['cohort'] == 'code' and value['policy'] == 'original' and value['seed'] == 0)(read(item['path']))]
        require(len(matches) == 1, 'Teacher policy failing condition is not unique')
        item = matches[0]
        plan = read(item['path'])
        output = Path(root) / 'outputs/pilot-qualifications' / plan['study_id'] / 'teacher-gate-prefix'
        require(not output.exists(), 'Teacher policy engineering prefix already exists')
        python, env, tools = environment(root, hardware['node'], bundle)
        options = ['--plan', item['path'], '--directory', str(output), '--attempt', 'fresh-prefix', '--stop', '208', '--qualification', '--teacher-gate-prefix']
        process = execute(python, env, 'taps_rebuild.pilot_train', options, output / 'launch.log')
        for rank in range(8):
            value = read(output / 'attempts/fresh-prefix' / ('rank-' + str(rank) + '.complete.json'))
            require(value['teacher_gate_engineering_prefix'] is True and value['qualification'] is True and value['start_step'] == 0 and value['stop_step'] == value['successful_updates_this_attempt'] == 208 and value['plan_identity_sha256'] == plan['identity_sha256'] and value['all_thirteen_trainables_received_finite_nonzero_gradients_each_update'] is True, 'Teacher gate engineering prefix coverage differs')
            policy = value['checkpoint_policy']['teacher_buffers']
            require(policy['policy'] == 'exact_pre_deepspeed_fp32_teacher_buffers' and policy['pristine_buffer_identity_sha256'] == policy['restored_buffer_identity_sha256'], 'Teacher gate prefix pristine buffer policy differs')
        pool.atomic_json(output / 'completion.json', {'status': 'completed208_update_engineering_teacher_gate_prefix', 'scientific_training_admitted': False, 'plan_identity_sha256': plan['identity_sha256'], 'process': process, 'hardware': hardware, 'tools': tools})
        qualify_attached(root, study)
        return
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Teacher qualification coordinator must be CPU-only')
    job = holder(root)
    command = ['srun', '--jobid=' + job, '--account=grad-students', '--partition=high-priority', '--qos=spec-priority', '--exclusive', '--nodes=1', '--ntasks=1', '--gres=gpu:a100:8', '--cpus-per-task=48', '--mem=0', '--time=01:30:00', '--immediate=60', '--kill-on-bad-exit=1', '--job-name=taps-teacher-qualification', '/usr/bin/python3', '-B', '-S', str(Path(__file__).resolve()), '--root', str(root), '--study', str(study), '--attached']
    env = dict(os.environ)
    for name in ('CUDA_VISIBLE_DEVICES', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
        env.pop(name, None)
    subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--attached', action='store_true')
    args = parser.parse_args()
    run(args.root, args.study, args.attached)
