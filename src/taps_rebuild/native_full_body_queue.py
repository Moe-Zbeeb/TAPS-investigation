import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import native_full_body as corrected, native_training as native
from .pilot_cache import preprocessor, read, records, require


def descriptor(path):
    return {'path': str(Path(path).resolve()), 'sha256': native.digest(path)}


def prepare(root, parent_path, output):
    native.require_slurm()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Corrected native preparation requires CPU Slurm')
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root / 'outputs/preparation') and not output.exists(), 'Choose a fresh corrected native preparation directory')
    parent = read(parent_path)
    require(hashlib.sha256(native.canonical({key: value for key, value in parent.items() if key != 'plan_identity_sha256'})).hexdigest() == parent['plan_identity_sha256'], 'Historical data parent identity changed')
    require(all(native.digest(parent[key]['frozen_path']) == parent[key]['sha256'] for key in ('train', 'validation')), 'Historical data parent files changed')
    require('work/cache.pt' in parent['frozen_files'] and 'inputs/vocabulary-audit.json' in parent['frozen_files'], 'Complete training-only vocabulary provenance is required')
    qa = read(Path(parent['run_dir']) / 'inputs/vocabulary-audit.json')
    require(native.digest(Path(parent['run_dir']) / 'inputs/vocabulary-audit.json') == parent['frozen_files']['inputs/vocabulary-audit.json'], 'Historical vocabulary audit bytes changed')
    require(qa['status'] == 'passed' and qa['vocabulary_uses_train_only'] is True and qa['splits']['train']['input_sha256'] == parent['train']['sha256'] and qa['splits']['dev']['input_sha256'] == parent['validation']['sha256'], 'Historical input/vocabulary QA does not match the full data')
    output.mkdir(parents=True)
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(parent['target']['path'], local_files_only=True)
    function = preprocessor(Path(parent['workdir']) / 'main.py', tokenizer, torch)
    selected, inspected = [], 0
    for row in records(parent['train']['frozen_path']):
        inspected += 1
        encoded = function({key: [value] for key, value in row.items()})
        if encoded['input_ids'] and encoded['input_ids'][0].shape == (1, 2048):
            selected.append(row)
            if len(selected) == 64:
                break
    require(len(selected) == 64, 'The registered maximum-shape qualification requires64 distinct real native2048 examples')
    validation = []
    for row in records(parent['validation']['frozen_path']):
        encoded = function({key: [value] for key, value in row.items()})
        if encoded['input_ids']:
            validation.append(row)
            if len(validation) == 8:
                break
    require(len(validation) == 8, 'Qualification validation subset has fewer than8 usable examples')
    for name, values in (('qualification-train', selected), ('qualification-dev', validation)):
        with (output / (name + '.jsonl')).open('x') as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    selection = {'scope': 'real-data maximum-shape native engineering subset; no scientific outcomes', 'parent_plan': descriptor(parent_path), 'full_train': descriptor(parent['train']['frozen_path']), 'full_validation': descriptor(parent['validation']['frozen_path']), 'inspected_train_rows': inspected, 'selection': 'first64 distinct native examples retained at exactly2048 tokens in frozen input order; first8 retained validation examples', 'train_ids': [value['id'] for value in selected], 'validation_ids': [value['id'] for value in validation], 'train_rows': 64, 'validation_rows': 8, 'all_training_rows_have2048_tokens': True}
    native.atomic_json(output / 'qualification-data.json', selection)
    base = {**parent['config']}
    base.pop('pipeline_stage_evidence', None)
    base['qualification'] = 'Corrected full-body native engineering; no scientific training result'
    policy = {'schema_version': 3, 'source_sha256': corrected.sources(), 'checkpoint_mode': 'non_reentrant', 'rotary_capacity': 2054, 'maximum_attempted_optimizer_steps': 32, 'capture_gradient_partitions': True, 'data_parent_plan': descriptor(parent_path), 'vocabulary_cache': descriptor(Path(parent['workdir']) / 'cache.pt'), 'vocabulary_audit': descriptor(Path(parent['run_dir']) / 'inputs/vocabulary-audit.json'), 'full_data_sha256': {key: parent[key]['sha256'] for key in ('train', 'validation')}, 'qualification_subset': descriptor(output / 'qualification-data.json')}
    plans = {}
    for label in ('uninterrupted', 'direct-reference', 'split', 'profile'):
        profile = label == 'profile'
        config = {**base, 'pipeline_stage': 'corrected-profile-v3' if profile else 'corrected-qualification-v3', 'budget': {'optimizer_steps': 200 if profile else 4}, 'recovery': {'checkpoint_interval': 100 if profile else 2, 'stop_after_optimizer_step': 2 if label == 'split' else None}, 'full_body_v3': {**policy, 'checkpoint_mode': 'off_reference' if label == 'direct-reference' else 'non_reentrant', 'maximum_attempted_optimizer_steps': 512 if profile else 32, 'capture_gradient_partitions': not profile}}
        config_path = output / (label + '-config.json')
        native.atomic_json(config_path, config)
        run = root / 'outputs/native-training' / (output.name + '-' + label)
        train = parent['train']['frozen_path'] if profile else output / 'qualification-train.jsonl'
        validation_path = parent['validation']['frozen_path'] if profile else output / 'qualification-dev.jsonl'
        corrected.prepare(root, config_path, train, validation_path, run)
        plans[label] = descriptor(run / 'plan.json')
    receipt = {'status': 'prepared_corrected_native_qualifications_and200_profile', 'source_sha256': corrected.sources(), 'plans': plans, 'data': descriptor(output / 'qualification-data.json'), 'scientific_training_admitted': False, 'full_generalist_not_resumed': True}
    native.atomic_json(output / 'queue-plan.json', receipt)
    return receipt


def verify_holder(root):
    authority = read(Path(root) / 'outputs/allocation-holders/deep11/current.json')
    job = str(authority['job_id'])
    record = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True)
    fields = dict(item.split('=', 1) for item in record.split() if '=' in item)
    require(fields.get('JobState') == 'RUNNING' and fields.get('UserId', '').startswith('mohamadzbib(') and fields.get('Account') == 'grad-students' and fields.get('Partition') == 'high-priority' and fields.get('QOS') == 'spec-priority' and fields.get('NodeList') == 'deep-chungus-11' and 'gres/gpu=8' in fields.get('TRES', ''), 'Corrected native work requires the live owned eight-GPU deep11 allocation')
    return job


def run_attached(root, queue):
    corrected.install()
    plan = read(queue)
    require(plan['source_sha256'] == corrected.sources(), 'Corrected native queue source changed')
    phases = []
    for label, resume in (('uninterrupted', False), ('direct-reference', False), ('split', False), ('split', True), ('profile', False)):
        item = plan['plans'][label]
        require(native.digest(item['path']) == item['sha256'], 'Corrected native plan changed')
        path = Path(item['path'])
        log = path.parent / ('launch-' + ('restore' if resume else 'initial') + '.log')
        require(not log.exists(), 'A corrected native phase already has a launch log')
        command = [str(Path(root) / 'envs/native-eagle3/bin/python'), '-B', str(path.parent / 'inputs/native_full_body.py'), 'resume' if resume else 'launch', '--plan', str(path)]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        with log.open('xb', buffering=0) as stream:
            subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        launch = read(path.parent / 'launch.json')
        require(launch['status'] == ('checkpointed_for_restart' if label == 'split' and not resume else 'complete'), 'Corrected native phase did not complete its exact boundary')
        for rank in range(8):
            value = read(path.parent / ('rank-' + str(rank) + '.json'))
            require(value['successful_optimizer_steps'] == (2 if label == 'split' and not resume else 200 if label == 'profile' else 4), 'Corrected native phase successful-step coverage differs')
        phases.append({'label': label, 'resume': resume, 'launch': descriptor(path.parent / 'launch.json'), 'log': descriptor(log)})
        native.atomic_json(Path(queue).parent / 'execution-progress.json', {'status': 'running_engineering', 'phases': phases, 'scientific_training_admitted': False})
    result = {'status': 'corrected_native_engineering_execution_complete_requires_independent_replay', 'phases': phases, 'source_sha256': corrected.sources(), 'scientific_training_admitted': False}
    native.atomic_json(Path(queue).parent / 'execution.json', result)
    return result


def launch(root, queue):
    native.require_slurm()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Corrected native coordinator requires CPU Slurm')
    job = verify_holder(root)
    command = ['srun', '--jobid=' + job, '--account=grad-students', '--partition=high-priority', '--qos=spec-priority', '--exclusive', '--nodes=1', '--ntasks=1', '--gres=gpu:a100:8', '--cpus-per-task=48', '--mem=0', '--immediate=60', '--kill-on-bad-exit=1', '--job-name=taps-corrected-native', sys.executable, '-B', '-S', '-m', 'taps_rebuild.native_full_body_queue', 'run-attached', '--root', str(root), '--queue', str(queue)]
    environment = dict(os.environ)
    for name in ('CUDA_VISIBLE_DEVICES', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
        environment.pop(name, None)
    native.atomic_json(Path(queue).parent / 'handoff.json', {'command': command, 'holder_job_id': job, 'source_sha256': corrected.sources()})
    subprocess.run(command, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'launch', 'run-attached'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--queue', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        value = prepare(args.root, args.parent, args.output)
        print(json.dumps(value, indent=2))
    elif args.action == 'launch':
        launch(args.root, args.queue)
    else:
        print(json.dumps(run_attached(args.root, args.queue), indent=2))


if __name__ == '__main__':
    main()
