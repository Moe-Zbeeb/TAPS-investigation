import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import admission, native_training, pool


def audit(root, config_path):
    root = Path(root).resolve()
    config = admission.read(config_path)
    directory = root / 'outputs/training-queue' / config['run_id']
    directory.mkdir(parents=True, exist_ok=True)
    status = {'status': 'checking_validation', 'config_sha256': admission.digest(config_path), 'training_started': False, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    pool.atomic_json(directory / 'status.json', status)
    try:
        if config.get('admission_policy') == 'pipeline-v2':
            research_gate = {'policy': 'pipeline-v2', 'pipeline_certificate': config['pipeline_certificate'], 'independent_review': config['independent_review']}
        else:
            report = admission.audit(root, root / config['driver'], root / config['prepared'], root / config['primary'], root / config['screen'], root / config['proof'], root / config['review'])
            pool.atomic_json(directory / 'full-model-certificate.json', report)
            research_gate = {'reproduction_config_sha256': config['reproduction_config_sha256'], 'reproduction_report': config['reproduction_report'], 'full_model_temperature_one_report': {'path': str(directory / 'full-model-certificate.json'), 'sha256': admission.digest(directory / 'full-model-certificate.json')}}
        research_config = {**admission.read(root / config['native_template']), 'scope': 'native_training', 'initialize_from_released_draft': False, 'budget': {'optimizer_steps': config['profile_optimizer_steps']}, 'checkpoint_roundtrip': False, 'qualification': 'Scratch native real-data profiling; not the completed forty-epoch generalist or a trained-quality benchmark', 'research_gate': research_gate}
        if config.get('admission_policy') == 'pipeline-v2':
            research_config['pipeline_stage'] = 'real-data-profile'
        pool.atomic_json(directory / 'native-profile-config.json', research_config)
        native_training.research_gate(root, research_config)
        completion_path = root / config['generated_corpus'] / 'completion.json'
        corpus = admission.read(completion_path)
        admission.require(corpus['status'] == 'complete' and corpus['scope'] == 'full_training_data_preparation' and corpus['response_regime'] == 'target_generated', 'A completed full target-generated corpus is required')
        for name, item in corpus['files'].items():
            admission.require(admission.digest(completion_path.parent / name) == item['sha256'], 'Target-generated corpus changed')
        run = root / 'outputs/native-training' / (config['run_id'] + '-profile')
        native_training.prepare(root, directory / 'native-profile-config.json', completion_path.parent / 'native-train.jsonl', completion_path.parent / 'native-dev.jsonl', run)
        cache = root / 'cache/native-eagle3'
        os.environ.update(HF_HOME=str(cache / 'huggingface'), HF_HUB_CACHE=str(cache / 'huggingface/hub'), HF_DATASETS_CACHE=str(cache / 'datasets'), XDG_CACHE_HOME=str(cache / 'xdg'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        native_training.audit(run / 'plan.json')
        status.update(status='passed', native_plan=str(run / 'plan.json'), native_plan_sha256=admission.digest(run / 'plan.json'), research_gate=research_gate)
    except Exception as error:
        status.update(status='blocked', error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        status['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        pool.atomic_json(directory / 'status.json', status)
    return status


def profile(root, config_path):
    root = Path(root).resolve()
    config = admission.read(config_path)
    directory = root / 'outputs/training-queue' / config['run_id']
    status = admission.read(directory / 'status.json')
    admission.require(status['status'] == 'passed' and status['config_sha256'] == admission.digest(config_path), 'Training admission is closed')
    plan_path = Path(status['native_plan'])
    admission.require(admission.digest(plan_path) == status['native_plan_sha256'], 'CPU-prepared native plan changed')
    started = time.monotonic()
    native_training.launch(plan_path)
    run = plan_path.parent
    rows = [admission.read(run / ('rank-' + str(rank) + '.json')) for rank in range(8)]
    profile_report = {'status': 'complete', 'scope': '200-update scratch-native real-data profiling only', 'native_plan_sha256': admission.digest(plan_path), 'wall_seconds': time.monotonic() - started, 'successful_optimizer_steps_by_rank': [row['successful_optimizer_steps'] for row in rows], 'forty_epoch_reproduction_complete': False, 'mid_epoch_resume_validated': False, 'training_quality_result': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    pool.atomic_json(directory / 'profile-completion.json', profile_report)
    return profile_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit', 'profile'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.config) if args.action == 'audit' else profile(args.root, args.config)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
