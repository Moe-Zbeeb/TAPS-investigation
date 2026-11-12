import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from . import native_training as native, pool
from .pilot_cache import read, require
from .pilot_queue import holder, verify_bundle
from .pilot_study import verify_identity
from .pilot_train import admission, model_directory


def environment(root, node, bundle):
    python = Path(root) / 'envs/native-eagle3/bin/python'
    result = dict(os.environ, PYTHONPATH=str(bundle / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result.update(native.cuda_environment(root, node, {}))
    result, tools = native.jit_environment(python, result)
    result.update(HF_HOME=str(Path(root) / 'cache/native-eagle3/huggingface'), HF_DATASETS_CACHE=str(Path(root) / 'cache/native-eagle3/datasets'), WANDB_MODE='offline')
    return python, result, tools


def execute(python, env, module, options, log):
    command = [str(python), '-B', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=8', '-m', module, *map(str, options)]
    log.parent.mkdir(parents=True, exist_ok=True)
    require(not log.exists(), 'GPU phase log already exists')
    started = time.monotonic()
    with log.open('xb', buffering=0) as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    receipt = {'command': command, 'returncode': result.returncode, 'wall_seconds': time.monotonic() - started, 'occupied_node_gpu_seconds': 8 * (time.monotonic() - started), 'log': str(log), 'log_sha256': pool.digest(log)}
    pool.atomic_json(log.with_suffix('.receipt.json'), receipt)
    require(result.returncode == 0, 'GPU phase failed; inspect ' + str(log))
    return receipt


def study_manifest(study):
    manifest = verify_identity(read(Path(study) / 'manifest.json'))
    require(manifest['status'] == 'prepared_24_waiting_qualification' and len(manifest['plans']) == 24 and all(pool.digest(item['path']) == item['sha256'] for item in manifest['plans']), 'Frozen24-plan study changed')
    return manifest


def qualify_attached(root, study):
    bundle, frozen = verify_bundle()
    hardware = pool.allocation(8)
    require(hardware['node'] == 'deep-chungus-10', 'Pilot qualification requires registered deep10 hardware')
    manifest = study_manifest(study)
    python, env, tools = environment(root, hardware['node'], bundle)
    output = Path(root) / 'outputs/pilot-qualifications' / manifest['common']['study_id']
    output.mkdir(parents=True, exist_ok=True)
    phases = []
    with (output / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pool.atomic_json(output / 'launch.json', {'hardware': hardware, 'tools': tools, 'bundle': frozen, 'bundle_directory': str(bundle), 'study_manifest_sha256': pool.digest(Path(study) / 'manifest.json'), 'scope': 'source-matched engineering qualifications only'})
        for policy in ('original', 'target'):
            candidates = [item for item in manifest['plans'] if (lambda plan: plan['seed'] == 0 and plan['cohort'] == 'generalist' and plan['policy'] == policy)(read(item['path']))]
            require(len(candidates) == 1, 'Pilot qualification plan selection differs')
            plan_path = Path(candidates[0]['path'])
            plan = read(plan_path)
            for variant, stop, attempt, flags in (('uninterrupted', 4, 'full', ['--capture']), ('native-reference', 4, 'reference', ['--reference']), ('split', 2, 'prefix', []), ('split', 4, 'restore', []), ('alternate-pad', 4, 'alternate', ['--capture', '--alternate-pad'])):
                directory = output / policy / variant
                options = ['--plan', plan_path, '--directory', directory, '--attempt', attempt, '--stop', stop, '--qualification', *flags]
                if attempt == 'restore':
                    latest = read(directory / 'latest.json')
                    require(latest['next_step'] == 2 and pool.digest(Path(latest['path']) / 'manifest.json') == latest['manifest_sha256'], 'Qualification restore boundary is not committed step2')
                    options += ['--resume', latest['path']]
                phases.append(execute(python, env, 'taps_rebuild.pilot_train', options, output / 'logs' / (policy + '-' + variant + '-' + attempt + '.log')))
            trained = model_directory(plan, output / policy / 'uninterrupted', True) / 'exports/step-4'
            phases.append(execute(python, env, 'taps_rebuild.pilot_smoke', ['--plan', plan_path, '--export', trained, '--output', output / policy / 'stock-inference'], output / 'logs' / (policy + '-stock-inference.log')))
        report = {'status': 'gpu_qualification_execution_complete_requires_independent_replay', 'scope': 'engineering qualifications, no pilot outcomes or training admission', 'study_manifest_sha256': pool.digest(Path(study) / 'manifest.json'), 'source_sha256': manifest['common']['source_sha256'], 'phases': phases, 'hardware': hardware, 'scientific_training_admitted': False}
        pool.atomic_json(output / 'execution.json', report)
        return report


def summarize_run(plan, directory):
    reports, attempts = [], []
    for rank in range(8):
        phases = sorted((read(path) for path in (directory / 'attempts').glob('*/rank-' + str(rank) + '.complete.json')), key=lambda value: value['start_step'])
        cursor = 0
        for value in phases:
            require(value['start_step'] == cursor and value['qualification'] is False and value['physical_attempts_this_process'] == value['stop_step'] - value['start_step'] and value['plan_identity_sha256'] == plan['identity_sha256'] and value['vocabulary_identity_sha256'] == plan['vocabulary_identity_sha256'], 'Pilot clean segment coverage differs')
            cursor = value['stop_step']
        require(cursor == 1536 and phases[-1]['status'] == 'completed_fixed_budget', 'Pilot fixed-budget completion coverage differs')
        reports.append(phases[-1])
        attempts.append(phases)
    require(len({value['state_identity']['model'] for value in reports}) == 1, 'Replicated final pilot parameters disagree across ranks')
    report = {'status': 'completed_training_requires_inference_evaluation', 'run_id': plan['run_id'], 'seed': plan['seed'], 'cohort': plan['cohort'], 'policy': plan['policy'], 'plan_identity_sha256': plan['identity_sha256'], 'successful_updates': 1536, 'logical_examples': 12288, 'logical_effective_base_contexts': 12288 * plan['quota'], 'dense_slots': 25165824, 'physical_attempts': 12288, 'replayed_work': False, 'training_loop_gpu_seconds': 8 * max(sum(value['rank_loop_seconds_this_attempt'] for value in phases) for phases in attempts), 'cuda_default_stream_rank_seconds': [sum(value['cuda_default_stream_seconds_this_attempt'] for value in phases) for phases in attempts], 'checkpoint_rank_seconds': [sum(value['checkpoint_seconds_this_attempt'] for value in phases) for phases in attempts], 'ranks': reports, 'segments_by_rank': attempts, 'trained_export': str(Path(plan['trained_models']) / 'exports/step-1536')}
    pool.atomic_json(directory / 'completion.json', report)
    return report


def train_attached(root, study):
    bundle, frozen = verify_bundle()
    hardware = pool.allocation(8)
    require(hardware['node'] == 'deep-chungus-10', 'Scientific pilots require registered physical deep10 GPUs')
    manifest = study_manifest(study)
    python, env, tools = environment(root, hardware['node'], bundle)
    results = []
    with (Path(study) / 'training-controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in manifest['plans']:
            plan = verify_identity(read(item['path']))
            admission(plan)
            directory = Path(plan['output'])
            if (directory / 'completion.json').exists():
                result = summarize_run(plan, directory)
                results.append(result)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            job = pool._fields(pool._command(['scontrol', 'show', 'job', '-o', hardware['job_id']]))
            deadline = datetime.fromisoformat(job['EndTime']).timestamp() - 1200
            if time.time() >= deadline:
                result = {'status': 'waiting_successor_allocation', 'completed': len(results), 'current_run': plan['run_id'], 'holder_job_id': hardware['job_id']}
                pool.atomic_json(Path(study) / 'progress.json', result)
                return result
            previous = list((directory / 'attempts').glob('*')) if (directory / 'attempts').exists() else []
            attempt = 'segment-' + str(len(previous) + 1).zfill(4)
            options = ['--plan', item['path'], '--directory', directory, '--attempt', attempt, '--stop', 1536, '--allocation-stop-epoch', deadline]
            if previous:
                latest = read(directory / 'latest.json')
                require(pool.digest(Path(latest['path']) / 'manifest.json') == latest['manifest_sha256'] and all((phase / ('rank-' + str(rank) + '.complete.json')).is_file() for phase in previous for rank in range(8)), 'Only clean completed pilot segments can resume automatically')
                require(all(max(read(phase / ('rank-' + str(rank) + '.complete.json'))['stop_step'] for phase in previous) == latest['next_step'] for rank in range(8)), 'Pilot completion reports and committed resume cursor differ')
                options += ['--resume', latest['path']]
            process = execute(python, env, 'taps_rebuild.pilot_train', options, directory / (attempt + '.log'))
            last = read(directory / 'attempts' / attempt / 'rank-0.complete.json')
            if last['stop_step'] < 1536:
                result = {'status': 'waiting_successor_allocation', 'completed': len(results), 'current_run': plan['run_id'], 'logical_updates': last['stop_step'], 'holder_job_id': hardware['job_id']}
                pool.atomic_json(Path(study) / 'progress.json', result)
                return result
            result = summarize_run(plan, directory)
            result['process'] = process
            results.append(result)
            pool.atomic_json(Path(study) / 'progress.json', {'status': 'training', 'completed': len(results), 'total': 24, 'latest_run': plan['run_id'], 'results': results})
        matching = {}
        for seed in (0, 1, 2):
            times = [value['training_loop_gpu_seconds'] for value in results if value['seed'] == seed]
            require(len(times) == 8, 'Pilot seed-block coverage differs')
            median = statistics.median(times)
            matching[seed] = {'median_gpu_seconds': median, 'max_min_ratio': max(times) / min(times), 'all_within_10_percent_of_median': all(abs(value / median - 1) <= 0.1 for value in times), 'run_gpu_seconds': times}
        result = {'status': '24_training_runs_complete_requires_evaluation', 'runs': results, 'compute_matching': matching, 'hardware': hardware, 'bundle': frozen, 'tools': tools, 'full_pdf_complete': False}
        pool.atomic_json(Path(study) / 'training-completion.json', result)
        return result


def launch(root, study, stage, wait_seconds):
    bundle, frozen = verify_bundle()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Pilot coordinator requires CPU Slurm')
    manifest = study_manifest(study)
    if stage == 'train':
        deadline = time.monotonic() + wait_seconds
        while not (Path(study) / 'admission.json').is_file():
            pool.atomic_json(Path(study) / 'queue-status.json', {'status': 'waiting_independent_qualification_admission', 'planned_runs': 24, 'gpu_step_started': False, 'source_bundle': frozen})
            require(time.monotonic() < deadline, 'Pilot admission wait expired; no scientific GPU step was launched')
            time.sleep(30)
        admission(read(manifest['plans'][0]['path']))
    env = dict(os.environ, PYTHONPATH=str(bundle / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    for name in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
        env.pop(name, None)
    previous_job = None
    while True:
        try:
            job = holder(root)
        except (ValueError, pool.PoolError, subprocess.CalledProcessError):
            require(previous_job is not None and read(Path(root).parent / 'rl-infra/cluster/deep10/current.json')['job_id'] == previous_job, 'Protected holder authority became incompatible')
            time.sleep(30)
            continue
        if job == previous_job:
            time.sleep(30)
            continue
        command = ['srun', '--jobid=' + job, '--account=grad-students', '--partition=high-priority', '--qos=spec-priority', '--exclusive', '--nodes=1', '--ntasks=1', '--gres=gpu:a100:8', '--cpus-per-task=48', '--mem=0', '--immediate=60', '--kill-on-bad-exit=1', '--job-name=taps-pilot-' + stage, '/usr/bin/python3', '-B', '-S', '-m', 'taps_rebuild.pilot_pipeline', stage + '-attached', '--root', str(root), '--study', str(study)]
        pool.atomic_json(Path(study) / (stage + '-handoff-' + job + '.json'), {'command': command, 'holder_job_id': job, 'stage': stage, 'bundle': frozen})
        subprocess.run(command, env=env, check=True)
        if stage != 'train' or (Path(study) / 'training-completion.json').exists():
            break
        require(read(Path(study) / 'progress.json')['status'] == 'waiting_successor_allocation', 'Pilot controller ended without completion or a clean successor boundary')
        previous_job = job
    return {'status': 'gpu_step_completed', 'stage': stage}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('qualify', 'qualify-attached', 'train', 'train-attached'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--admission-wait-seconds', type=int, default=86400)
    args = parser.parse_args()
    if args.action.endswith('-attached'):
        result = (qualify_attached if args.action == 'qualify-attached' else train_attached)(args.root, args.study)
    else:
        result = launch(args.root, args.study, args.action, args.admission_wait_seconds)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
