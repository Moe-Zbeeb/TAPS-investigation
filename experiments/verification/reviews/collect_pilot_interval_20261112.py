import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(root, submission, output):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Interval accounting requires CPU Slurm')
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root / 'outputs') or output.exists():
        raise ValueError('Choose a fresh interval receipt under outputs/')
    source = json.loads(submission.read_bytes())
    mapping = {'pre_job': source['jobs']['pre']['job_id'], 'qualification_job': '2216698', 'post_job': source['jobs']['post']['job_id']}
    environment = dict(os.environ, TZ='UTC', SLURM_TIME_FORMAT='standard')
    command = ['sacct', '-X', '-n', '-P', '-j', ','.join(mapping.values()), '--format=JobIDRaw,JobName,User,Account,Partition,QOS,State,ExitCode,Start,End,ElapsedRaw']
    expected = {'pre_job': 'taps-runtime-pre', 'qualification_job': 'taps-pilot24-qualify', 'post_job': 'taps-runtime-post'}
    for attempt in range(30):
        raw = subprocess.check_output(command, env=environment, text=True, timeout=30)
        rows = {fields[0]: fields for fields in (line.split('|') for line in raw.splitlines()) if len(fields) >= 11}
        if all(job in rows and rows[job][6] == 'COMPLETED' for job in mapping.values()):
            break
        if attempt == 29:
            raise ValueError('Successful pre/qualification/post accounting is unavailable')
        time.sleep(2)
    jobs = {}
    for name, job in mapping.items():
        row = rows[job]
        if row[1] != expected[name] or row[2:8] != ['mohamadzbib', 'grad-students', 'high-priority', 'spec-priority', 'COMPLETED', '0:0']:
            raise ValueError('Runtime interval Slurm ownership, class or completion differs')
        fields = source['jobs']['pre' if name == 'pre_job' else 'post']['verified_slurm'] if name != 'qualification_job' else source['dependency_overrides']['2216698']['verified_slurm']
        dependency = fields.split('Dependency=', 1)[1].split()[0]
        afterok = []
        for part in dependency.split(','):
            if part.startswith('afterok:'):
                afterok.extend(value.split('(')[0] for value in part.split(':')[1:])
        jobs[name] = {'job_id': job, 'state': row[6], 'exit_code': row[7], 'start_utc': datetime.fromisoformat(row[8]).replace(tzinfo=timezone.utc).isoformat(), 'end_utc': datetime.fromisoformat(row[9]).replace(tzinfo=timezone.utc).isoformat(), 'afterok_jobs': afterok}
    if mapping['pre_job'] not in jobs['qualification_job']['afterok_jobs'] or mapping['qualification_job'] not in jobs['post_job']['afterok_jobs'] or not (jobs['pre_job']['end_utc'] <= jobs['qualification_job']['start_utc'] <= jobs['qualification_job']['end_utc'] <= jobs['post_job']['start_utc']):
        raise ValueError('Runtime envelope does not bracket actual qualification')
    qualifier = root / 'outputs/pilot-qualifications/rq1-pilot-24-20261112-v1'
    launch = json.loads((qualifier / 'launch.json').read_bytes())
    step_command = ['sacct', '-n', '-P', '-j', launch['hardware']['job_id'], '--format=JobIDRaw,JobName,State,ExitCode,Start,End,NodeList,AllocTRES']
    steps = subprocess.check_output(step_command, env=environment, text=True, timeout=30)
    receipt = {**jobs, 'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'timezone_environment': {'TZ': 'UTC', 'SLURM_TIME_FORMAT': 'standard'}, 'raw_sacct': raw, 'sacct_command': command, 'raw_parent_step_accounting': steps, 'step_sacct_command': step_command, 'submission_path': str(submission.resolve()), 'submission_sha256': digest(submission), 'dependency_evidence': source, 'qualification_launch_sha256': digest(qualifier / 'launch.json'), 'collector_source_sha256': digest(__file__), 'scientific_training_admitted': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    return {'status': 'recorded_completed_runtime_envelope', 'output': str(output), 'sha256': digest(output)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'submission', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.submission, args.output)))


if __name__ == '__main__':
    main()
