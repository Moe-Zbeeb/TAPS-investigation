import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def fields(text):
    return {match.group(1): match.group(2) for match in re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:]*)=(.*?)(?=\s[A-Za-z][A-Za-z0-9_:]*=|$)', text.strip())}


def job(identifier):
    return fields(subprocess.check_output(['scontrol', 'show', 'job', '-o', identifier], text=True))


def deploy(root, adapter, review_path, code_commit):
    root = Path(root).resolve()
    require(str(root) == '/mnt/nfs/home/mohamadzbib/projects/taps-rebuild', 'This migration is scoped to the verified CSAIL TAPS project')
    old = root / 'outputs/preparation/pipeline-training-20261111-v1'
    bundle = root / 'outputs/preparation/parallel-generation-20261111-v1'
    generation = root / 'outputs/teacher-generation/sharegpt-ultrachat-parallel-20261111-v1'
    require(not bundle.exists() and not generation.exists(), 'Frozen migration already exists; inspect its submission receipt instead of resubmitting')
    review = read(review_path)
    require(review['status'] == 'approved' and review['scope'] == 'parallel_teacher_migration_only' and review['adapter_sha256'] == digest(adapter), 'Exact independent migration approval is required')
    authority = read(root.parent / 'rl-infra/cluster/deep10/current.json')
    protected = {authority['job_id'], authority['successor']}
    live = job(authority['job_id'])
    require(live['JobState'] == 'RUNNING' and live['JobName'] == 'deep10-hold' and live['NodeList'] == 'deep-chungus-10' and live['Account'] == 'grad-students' and live['QOS'] == 'spec-priority' and live['UserId'].endswith('(' + str(os.getuid()) + ')'), 'Protected holder identity differs')
    coordinator = job('2214983')
    require(coordinator['JobState'] == 'RUNNING' and coordinator['JobName'] == 'taps-data-finish' and coordinator['UserId'] == live['UserId'], 'Original TAPS coordinator changed')
    old_jobs = {'2215985': 'taps-native-data-qa', '2216031': 'taps-v2-prepare', '2216032': 'taps-v2-bind', '2216033': 'taps-v2-profile', '2216034': 'taps-v2-full-prep', '2216035': 'taps-full-train', '2216036': 'taps-full-continue'}
    for identifier, name in old_jobs.items():
        current = job(identifier)
        require(identifier not in protected and current['JobState'] == 'PENDING' and current['JobName'] == name and current['UserId'] == live['UserId'], 'Superseded pending job identity differs')
    used = 0
    for line in subprocess.check_output(['scontrol', 'show', 'job', '-o'], text=True).splitlines():
        current = fields(line)
        if current.get('UserId') == live['UserId'] and current.get('Partition') == 'high-priority' and current.get('JobState') in {'RUNNING', 'COMPLETING'}:
            matches = re.findall(r'(?:^|,)gres/gpu=(\d+)', current.get('AllocTRES', ''))
            used += int(matches[0]) if matches else 0
    require(used + 8 <= 18, 'A second full GPU node would exceed the high-priority allowance')
    bundle.mkdir()
    manifest = read(old / 'bundle.json')
    for relative, expected in manifest['files'].items():
        source = old / relative
        require(digest(source) == expected, 'Previously qualified frozen bundle changed: ' + relative)
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(adapter, bundle / 'src/taps_rebuild/corpus_parallel.py')
    shutil.copyfile(review_path, bundle / 'migration-review.json')
    context = root / 'outputs/preparation/remaining-greedy-20261111-v3'
    for source, destination, expected in [('driver.py', 'context.py', '87842c151e8bbbcd40b8e333ceb87106dad9add4fdf677e29eb5d87f6c65502b'), ('stale-context.json', 'stale-context.json', '65266223bb7f2c4895b874e0c6c4fbeeef22718c0b291db1c7ca0925c5ae4983')]:
        require(digest(context / source) == expected, 'Qualified context guard changed')
        shutil.copyfile(context / source, bundle / destination)
    replacements = {str(old): str(bundle), 'outputs/preparation/pipeline-training-20261111-v1': 'outputs/preparation/parallel-generation-20261111-v1', 'sharegpt-ultrachat-full-pipeline-20261111-v2': 'sharegpt-ultrachat-full-parallel-20261111-v1', 'sharegpt-ultrachat-pipeline-20261111-v2': 'sharegpt-ultrachat-profile-parallel-20261111-v1', 'sharegpt-ultrachat-teacher-20261110-v3-target-generated': generation.name + '-target-generated', 'outputs/native-data-qa/full-20261111-v1': 'outputs/native-data-qa/parallel-full-20261111-v1'}
    for path in list(bundle.glob('*.sh')) + list((bundle / 'configs').glob('*.json')):
        text = path.read_text()
        for previous, current in replacements.items():
            text = text.replace(previous, current)
        path.write_text(text)
    schedule = read(old / 'schedule.json')
    text = json.dumps(schedule)
    for previous, current in replacements.items():
        text = text.replace(previous, current)
    write(bundle / 'schedule.json', json.loads(text))
    header = '\n'.join(['set -euo pipefail', 'export PYTHONPATH=' + str(bundle / 'src'), 'export PYTHONDONTWRITEBYTECODE=1', 'export PYTHONNOUSERSITE=1', 'export PYTHONUNBUFFERED=1', 'export OMP_NUM_THREADS=1', 'export MKL_NUM_THREADS=1', 'export TOKENIZERS_PARALLELISM=false', 'export HF_HUB_OFFLINE=1', 'export TRANSFORMERS_OFFLINE=1', 'cd ' + shlex.quote(str(root))]) + '\n'
    native = str(root / 'envs/native-eagle3/bin/python')
    module = ['/usr/bin/python3', '-B', '-S', '-m', 'taps_rebuild.corpus_parallel']
    def script(name, arguments, cpu=True):
        content = header + ('export CUDA_VISIBLE_DEVICES=\n' if cpu else '') + shlex.join(arguments) + '\n'
        (bundle / (name + '.sh')).write_text(content)
    def generation_script(name, action, extra=(), cpu=True):
        script(name, module + [action, '--directory', str(generation), *extra], cpu)
    generation_script('initialize', 'initialize', ['--root', str(root), '--legacy-plan', str(root / 'outputs/teacher-generation/sharegpt-ultrachat-teacher-20261110-v3/plan.json')])
    generation_script('secondary', 'node', ['--node-rank', '1'], False)
    generation_script('attach', 'attach')
    script('data-qa', [native, '-B', '-m', 'taps_rebuild.native_data_qa', 'audit', '--root', str(root), '--corpus', str(root / 'data/corpora' / (generation.name + '-target-generated')), '--output', str(root / 'outputs/native-data-qa/parallel-full-20261111-v1')])
    base = ['sbatch', '--parsable', '--account=grad-students', '--partition=high-priority', '--qos=spec-priority', '--nodes=1', '--ntasks=1', '--chdir=' + str(root), '--kill-on-invalid-dep=yes']
    def command(name, script_name, dependencies=(), gpu=False, cpus=16, memory='64G', duration='04:00:00', held=False, after='afterok'):
        result = base + ['--nodelist=deep-chungus-' + ('11' if gpu else '6'), '--cpus-per-task=' + str(64 if gpu else cpus), '--time=' + duration, '--job-name=' + name, '--output=' + str(bundle / (script_name + '-%j.log'))]
        result += ['--exclusive', '--gres=gpu:a100:8'] if gpu else ['--mem=' + memory]
        if dependencies:
            result += ['--dependency=' + after + ':' + ':'.join(dependencies)]
        if held:
            result += ['--hold']
        return result + ['--wrap', '/bin/bash ' + shlex.quote(str(bundle / (script_name + '.sh')))]
    secondary = command('taps-parallel-gen', 'secondary', gpu=True, duration='1-00:00:00')
    secondary.insert(-2, '--comment=taps-parallel-20261111-v1')
    write(bundle / 'secondary-schedule.json', {'directory': str(generation), 'comment': 'taps-parallel-20261111-v1', 'command': secondary})
    jobs = {}
    def submit(key, cmd):
        identifier = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        require(identifier.isdigit(), 'Slurm returned a malformed job identifier')
        jobs[key] = {'job_id': identifier, 'command': cmd, 'slurm': job(identifier)}
        write(bundle / 'submission.json', {'code_commit': code_commit, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'jobs': jobs, 'migration_held_until_successor_queue_complete': True})
        return identifier
    initialization = submit('initialize', command('taps-parallel-init', 'initialize', memory='128G'))
    secondary_job = submit('secondary', secondary[:-2] + ['--dependency=afterok:' + initialization] + secondary[-2:])
    generation_script('migration', 'prepare', ['--secondary-job', secondary_job, '--old-coordinator', '2214983'])
    generation_script('finish', 'finish', ['--secondary-job', secondary_job])
    write(bundle / 'bundle.json', {'code_commit': code_commit, 'prior_frozen_bundle_sha256': digest(old / 'bundle.json'), 'migration_review_sha256': digest(review_path), 'files': {str(p.relative_to(bundle)): digest(p) for p in bundle.rglob('*') if p.is_file() and p.name not in {'bundle.json', 'submission.json'} and p.suffix != '.log'}})
    migration = submit('migration', command('taps-parallel-migrate', 'migration', [initialization], memory='128G', duration='08:00:00', held=True))
    attached = submit('protected_node', command('taps-parallel-attach', 'attach', [migration], cpus=1, memory='4G', duration='2-00:00:00'))
    finished = submit('finish', command('taps-parallel-merge', 'finish', [attached], duration='2-00:00:00', after='afterany'))
    qa = submit('data_qa', command('taps-native-data-qa', 'data-qa', [finished], cpus=48, memory='128G', duration='12:00:00'))
    prepared = submit('profile_prepare', command('taps-v2-prepare', 'prepare', [finished]))
    bound = submit('vocabulary_bind', command('taps-v2-bind', 'bind', [qa, prepared]))
    profile = submit('profile', command('taps-v2-profile', 'profile', [bound], gpu=True, duration='08:00:00'))
    full_prepared = submit('full_prepare', command('taps-v2-full-prep', 'full_prepare', [profile]))
    full = submit('full_training', command('taps-full-train', 'full', [full_prepared], gpu=True, duration='2-00:00:00'))
    continuation = command('taps-full-continue', 'continue', [full], after='afterany')
    continuation[-1] += ' --previous-job ' + full + ' --attempt 1'
    submit('conditional_continuation', continuation)
    retired = []
    for identifier, name in reversed(list(old_jobs.items())):
        current = job(identifier)
        require(current['JobState'] == 'PENDING' and current['JobName'] == name and current['UserId'] == live['UserId'] and identifier not in protected, 'Old pending queue changed before retirement')
        retired.append({'job_id': identifier, 'slurm_before_cancel': current})
        subprocess.run(['scancel', identifier], check=True)
    write(bundle / 'retired-pending-jobs.json', retired)
    subprocess.run(['scontrol', 'release', migration], check=True)
    write(bundle / 'migration-released.json', {'job_id': migration, 'all_successors_queued': True, 'protected_jobs_preserved': sorted(protected), 'old_generation_stopped_only_after_all_eight_secondary_gpu_proofs': True})
    print(json.dumps({'bundle': str(bundle), 'generation': str(generation), 'jobs': {key: value['job_id'] for key, value in jobs.items()}, 'running_high_priority_gpus_before_second_node': used, 'maximum_planned_running_high_priority_gpus': used + 8}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    deploy(args.root, args.adapter, args.review, args.code_commit)
