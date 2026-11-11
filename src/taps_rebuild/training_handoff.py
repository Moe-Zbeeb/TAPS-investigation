import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from . import admission, native_data_qa, native_training as native


def qualifications(root, config):
    root = Path(root).resolve()
    bound = {}
    for name in ('restart_qualification', 'export_qualification'):
        path = (root / config[name]).resolve()
        admission.require(path.is_relative_to(root / 'outputs'), 'Qualification must be a project execution artifact')
        report = admission.read(path)
        admission.require(report.get('status') == 'passed' and report.get('training_admission') is False, 'Engineering qualification is incomplete: ' + name)
        rows = report.get('ranks', [])
        admission.require(len(rows) == 8 and {row.get('rank') for row in rows} == set(range(8)) and all(row.get('passed') is True for row in rows), 'Engineering qualification lacks eight passing ranks: ' + name)
        bound[name] = {'path': str(path), 'sha256': admission.digest(path)}
        admission.require(report.get('native_training_sha256') == native.digest(native.__file__) and report.get('native_recovery_sha256') == native.digest(Path(native.__file__).with_name('native_recovery.py')), 'Engineering qualification covers another training implementation')
    return bound


def prepare(root, config_path):
    native.require_slurm()
    admission.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Full training preparation requires CPU Slurm')
    root = Path(root).resolve()
    config = admission.read(config_path)
    admission.require(all(isinstance(config.get(key), str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', config[key]) for key in ('run_id', 'profile_run_id')) and (config.get('requires_original_scientific_gates') is True or config.get('requires_scientific_gates') is True and config.get('admission_policy') == 'pipeline-v2'), 'Full training requires valid run identities and scientific gates')
    directory = root / 'outputs/training-handoff' / config['run_id']
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'status.json').exists():
        previous = admission.read(directory / 'status.json')
        if previous['status'] == 'passed':
            admission.require(previous['config_sha256'] == admission.digest(config_path) and admission.digest(previous['native_plan']) == previous['native_plan_sha256'] and qualifications(root, config) == previous['qualifications'], 'Completed full training preparation changed')
            return previous
    status = {'status': 'checking', 'config_sha256': admission.digest(config_path), 'training_started': False, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        status['qualifications'] = qualifications(root, config)
        profile_directory = root / 'outputs/training-queue' / config['profile_run_id']
        profile = admission.read(profile_directory / 'profile-completion.json')
        admission.require(profile.get('status') == 'complete' and profile.get('successful_optimizer_steps_by_rank') == [200] * 8, 'The real-data 200-update profile must complete on eight ranks')
        prior_status = admission.read(profile_directory / 'status.json')
        admission.require(prior_status.get('status') == 'passed', 'Scientific profile admission is closed')
        plan_path = Path(prior_status['native_plan'])
        admission.require(admission.digest(plan_path) == prior_status['native_plan_sha256'] == profile['native_plan_sha256'], 'Completed profile plan changed')
        profile_plan = native.load_plan(plan_path)
        if config.get('admission_policy') == 'pipeline-v2':
            admission.require(profile_plan['config']['research_gate'].get('policy') == 'pipeline-v2', 'The full handoff requires the independently reviewed v2 pipeline gate')
        admission.require(profile_plan['scope'] == 'native_training' and profile_plan['config']['initialize_from_released_draft'] is False, 'Full reproduction requires an admitted scratch-native profile')
        admission.require('work/cache.pt' in profile_plan['frozen_files'] and 'inputs/vocabulary-audit.json' in profile_plan['frozen_files'], 'A source-bound CPU-prepared native vocabulary is required')
        full_config = {**profile_plan['config'], 'budget': {'optimizer_steps': None}, 'checkpoint_roundtrip': False, 'recovery': {'checkpoint_interval': 1000, 'stop_after_optimizer_step': None}, 'qualification': 'Full forty-epoch scratch native ShareGPT/UltraChat generalist reproduction; quality requires subsequent held-out evaluation'}
        attempt = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
        preparation = directory / ('preparation-attempt-' + str(attempt))
        preparation.mkdir(exist_ok=False)
        native.atomic_json(preparation / 'native-full-config.json', full_config)
        run = root / 'outputs/native-training' / (config['run_id'] + '-preparation-' + str(attempt))
        plan = native.prepare(root, preparation / 'native-full-config.json', profile_plan['train']['frozen_path'], profile_plan['validation']['frozen_path'], run)
        plan = native_data_qa.bind_vocabulary(run / 'plan.json', Path(profile_plan['run_dir']) / 'inputs/vocabulary-audit.json', Path(profile_plan['workdir']) / 'cache.pt')
        native.audit(run / 'plan.json')
        status.update(status='passed', native_plan=str(run / 'plan.json'), native_plan_sha256=admission.digest(run / 'plan.json'), source_profile_plan_sha256=admission.digest(plan_path), full_epochs=plan['native_recipe']['epochs'], automatic_training_after_checks=True)
    except Exception as error:
        status.update(status='blocked', error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        status['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
        native.atomic_json(directory / 'status.json', status)
    return status


def launch(root, config_path, *, resume=False):
    root = Path(root).resolve()
    config = admission.read(config_path)
    directory = root / 'outputs/training-handoff' / config['run_id']
    status = admission.read(directory / 'status.json')
    admission.require(status.get('status') == 'passed' and status['config_sha256'] == admission.digest(config_path), 'Full training handoff is closed')
    admission.require(qualifications(root, config) == status['qualifications'], 'Engineering qualification evidence changed')
    path = Path(status['native_plan'])
    admission.require(admission.digest(path) == status['native_plan_sha256'], 'Full training plan changed')
    native.launch(path, resume=resume)
    receipt = admission.read(path.parent / 'launch.json')
    if receipt['status'] == 'complete':
        plan = native.load_plan(path)
        exports = []
        for epoch in range(plan['native_recipe']['epochs']):
            checkpoint = Path(plan['trained_models']['resolved_path']) / ('state_' + str(epoch))
            admission.require((checkpoint / 'pytorch_model.bin').is_file(), 'Missing native epoch export')
            source = Path(plan['workdir']) / 'config.json'
            destination = checkpoint / 'config.json'
            if destination.exists():
                admission.require(native.digest(destination) == native.digest(source), 'Native exported config differs')
            else:
                destination.write_bytes(source.read_bytes())
            exports.append({'epoch': epoch, 'path': str(checkpoint), 'config_sha256': native.digest(destination)})
        native.atomic_json(directory / 'completion.json', {'status': 'training_complete_evaluation_required', 'native_plan_sha256': native.digest(path), 'exports': exports, 'trained_quality_result': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'launch', 'resume'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root, args.config)
    else:
        launch(args.root, args.config, resume=args.action == 'resume')


if __name__ == '__main__':
    main()
