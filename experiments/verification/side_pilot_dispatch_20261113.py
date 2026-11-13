import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


STUDY_ID = 'rq1-pilot-24-20261113-v3'
REPORT_SHA256 = '8bb13bef08ef73121affe85bfcc51f8d56fadeb486b117339f3a1396c3b70783'
MANIFEST_SHA256 = 'b26599d4644a33c054f6a24853eedae9241ee0f3d6473cbd5fb6a3ef3fbcbdd9'
NORMAL_NAME = 'taps-side-pilots'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def fields(text):
    return dict(value.split('=', 1) for value in text.split() if '=' in value)


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def check_study(root, study, producer, review_sha256):
    require(study.resolve() == root.resolve() / 'outputs/pilot-training' / STUDY_ID and digest(study / 'manifest.json') == MANIFEST_SHA256, 'Dispatcher is restricted to the reviewed fresh study')
    manifest = read(study / 'manifest.json')
    require(Path(manifest['common']['source_directory']).resolve() == producer.resolve() / 'src/taps_rebuild', 'Dispatcher producer differs from the registered study')
    bundle = read(producer / 'bundle.json')
    require(bundle['files'] and all(digest(producer / name) == expected for name, expected in bundle['files'].items()), 'Frozen producer bundle changed')
    admission = read(study / 'admission.json')
    require(admission['scientific_training_admitted'] is True and admission['qualification_report']['sha256'] == REPORT_SHA256 and admission['binding']['study_manifest_sha256'] == MANIFEST_SHA256 and admission['independent_review']['sha256'] == review_sha256, 'Dispatcher lacks exact integrated scientific approval')
    require(digest(admission['independent_review']['path']) == review_sha256 and digest(admission['qualification_report']['path']) == REPORT_SHA256, 'Integrated review or report changed')
    return manifest


def submission_environment():
    result = dict(os.environ)
    for name in ('CUDA_VISIBLE_DEVICES', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'WANDB_API_KEY'):
        result.pop(name, None)
    return result


def normal_job_record(job):
    require(str(job).isdigit(), 'Normal pilot job ID must be numeric')
    result = subprocess.run(['scontrol', 'show', 'job', '-o', str(job)], capture_output=True, text=True)
    if result.returncode == 0:
        value = fields(result.stdout)
        require(value.get('UserId', '').startswith('mohamadzbib(') and value.get('Account') == 'grad-students' and value.get('Partition') == 'low-priority' and value.get('QOS') == 'normal' and value.get('JobName') == NORMAL_NAME, 'Normal pilot job ownership or class differs')
        return value
    output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job), '-o', 'JobIDRaw,State%30,ExitCode,User,Account,Partition,QOS,JobName%60'], text=True)
    rows = [line.split('|') for line in output.splitlines() if line.split('|')[0] == str(job)]
    require(len(rows) == 1 and rows[0][3:8] == ['mohamadzbib', 'grad-students', 'low-priority', 'normal', NORMAL_NAME], 'Normal pilot accounting differs')
    return {'JobId': str(job), 'JobState': rows[0][1].split()[0].rstrip('+'), 'ExitCode': rows[0][2]}


def submit_normal(root, study, producer, review_sha256):
    directory = Path(__file__).resolve().parent
    arguments = ['/usr/bin/python3', '-B', '-S', str(Path(__file__).resolve()), 'attached', '--root', str(root), '--study', str(study), '--producer', str(producer), '--review-sha256', review_sha256]
    script = directory / 'normal.sh'
    content = 'set -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\nexport PYTHONNOUSERSITE=1\n' + shlex.join(arguments) + '\n'
    if not script.exists():
        with script.open('x') as stream:
            stream.write(content)
    require(script.read_text() == content, 'Frozen normal-priority batch script changed')
    command = ['sbatch', '--parsable', '--account=grad-students', '--partition=low-priority', '--qos=normal', '--nodes=1', '--exclusive', '--nodelist=deep-chungus-10', '--gres=gpu:a100:8', '--ntasks=1', '--cpus-per-task=48', '--mem=0', '--time=48:00:00', '--no-requeue', '--chdir=' + str(root), '--job-name=' + NORMAL_NAME, '--output=' + str(directory / 'normal-%j.log'), '--wrap', '/bin/bash ' + shlex.quote(str(script))]
    result = subprocess.run(command, env=submission_environment(), capture_output=True, text=True, check=True)
    job = result.stdout.strip().split(';')[0]
    record = normal_job_record(job)
    require(record.get('ReqNodeList') == 'deep-chungus-10' and re.search(r'(?:^|,)gres/gpu=8(?:,|$)', record.get('TRES', '')), 'Normal pilot request is not the full registered node')
    atomic(directory / ('normal-submission-' + job + '.json'), {'job_id': job, 'command': command, 'verified_slurm': record, 'scientific_source_changed': False})
    return job


def attached(root, study, producer, review_sha256):
    check_study(root, study, producer, review_sha256)
    sys.path.insert(0, str(producer / 'src'))
    from taps_rebuild import pool
    from taps_rebuild.pilot_pipeline import train_attached
    hardware = pool.allocation(8)
    job = hardware['slurm']['job']
    require(hardware['node'] == 'deep-chungus-10' and job.get('Account') == 'grad-students' and (job.get('Partition'), job.get('QOS')) in {('high-priority', 'spec-priority'), ('low-priority', 'normal')}, 'Pilot allocation does not match the registered node and authorized scheduling class')
    usage = subprocess.check_output(['squeue', '-h', '-u', 'mohamadzbib', '-p', 'high-priority', '-t', 'RUNNING', '-o', '%b'], text=True)
    require(sum(map(int, re.findall(r'gpu(?::[A-Za-z0-9_.-]+)?:(\d+)', usage))) <= 18, 'Total high-priority GPU allowance exceeded')
    atomic(study / 'side-active-allocation.json', {'job_id': hardware['job_id'], 'node': hardware['node'], 'partition': job['Partition'], 'qos': job['QOS'], 'gpu_count': 8, 'review_sha256': review_sha256})
    train_attached(root, study)


def next_action(high, last_holder, record, progress):
    if record and record['JobState'] not in ACTIVE:
        require(record['JobState'] == 'COMPLETED' and record.get('ExitCode') == '0:0' and progress.get('status') == 'waiting_successor_allocation' and str(progress.get('holder_job_id')) == str(record['JobId']), 'Pilot allocation ended without a clean registered boundary; automatic scientific retry refused')
        return 'clean_boundary'
    if high and str(high) != str(last_holder) and (record is None or record['JobState'] == 'PENDING'):
        return 'high'
    return 'submit' if record is None else 'wait'


def cancel_pending_fallback(job):
    if normal_job_record(job)['JobState'] != 'PENDING':
        return False
    subprocess.run(['scancel', '--state=PENDING', str(job)], check=True)
    stopped = normal_job_record(job)
    if stopped['JobState'] in ACTIVE:
        return False
    require(stopped['JobState'].startswith('CANCELLED'), 'Pending fallback cancellation did not produce a clean queued cancellation')
    return True


def coordinate(root, study, producer, review_sha256):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Pilot dispatcher requires CPU Slurm')
    check_study(root, study, producer, review_sha256)
    sys.path.insert(0, str(producer / 'src'))
    from taps_rebuild import pool
    from taps_rebuild.pilot_queue import holder
    path = study / 'side-priority-dispatch.json'
    with (study / 'side-priority-dispatch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read(path) if path.exists() else {}
        normal = previous.get('normal_job_id')
        last_holder = previous.get('last_holder_job_id')
        while True:
            if (study / 'training-completion.json').exists():
                atomic(path, {'status': 'all24_training_runs_complete_evaluation_required', 'normal_job_id': normal, 'last_holder_job_id': last_holder})
                return
            try:
                high = holder(root)
            except (ValueError, pool.PoolError, subprocess.CalledProcessError):
                high = None
            record = normal_job_record(normal) if normal else None
            progress = read(study / 'progress.json') if (study / 'progress.json').exists() else {}
            action = next_action(high, last_holder, record, progress)
            if action == 'clean_boundary':
                normal = None
                atomic(path, {'status': 'clean_allocation_boundary', 'last_holder_job_id': last_holder, 'normal_job_id': None})
                continue
            if action == 'high':
                if record and record['JobState'] == 'PENDING':
                    if not cancel_pending_fallback(normal):
                        time.sleep(30)
                        continue
                    normal = None
                command = ['srun', '--jobid=' + str(high), '--account=grad-students', '--partition=high-priority', '--qos=spec-priority', '--exclusive', '--nodes=1', '--ntasks=1', '--gres=gpu:a100:8', '--cpus-per-task=48', '--mem=0', '--immediate=60', '--kill-on-bad-exit=1', '/usr/bin/python3', '-B', '-S', str(Path(__file__).resolve()), 'attached', '--root', str(root), '--study', str(study), '--producer', str(producer), '--review-sha256', review_sha256]
                atomic(path, {'status': 'launching_on_protected_high_priority_holder', 'holder_job_id': high, 'normal_job_id': normal, 'last_holder_job_id': last_holder})
                subprocess.run(command, env=submission_environment(), check=True)
                last_holder = str(high)
                require((study / 'training-completion.json').exists() or read(study / 'progress.json').get('status') == 'waiting_successor_allocation', 'High-priority pilot step ended without a clean boundary')
                atomic(path, {'status': 'clean_high_priority_boundary', 'last_holder_job_id': last_holder, 'normal_job_id': normal})
                continue
            if normal is None:
                normal = submit_normal(root, study, producer, review_sha256)
                record = normal_job_record(normal)
            atomic(path, {'status': 'running_normal_priority_pilots' if record['JobState'] == 'RUNNING' else 'waiting_for_registered_gpu_node', 'normal_job_id': normal, 'normal_state': record['JobState'], 'normal_reason': record.get('Reason'), 'last_holder_job_id': last_holder, 'protected_holder_changed': False})
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('coordinate', 'attached'))
    for name in ('root', 'study', 'producer'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID'), 'Pilot dispatch must run inside Slurm')
    (coordinate if args.action == 'coordinate' else attached)(args.root, args.study, args.producer, args.review_sha256)
