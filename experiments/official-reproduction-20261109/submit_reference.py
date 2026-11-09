import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from taps_rebuild import attach, pool


SPECS = (
    ('eagle3', 'official-eagle3-20261109-v1', 'deep-chungus-3', 9, 'high-priority', 'spec-priority'),
    ('hass', 'official-hass-20261109-v1', 'deep-chungus-11', 8, 'low-priority', 'normal'),
)
SPEC_BY_METHOD = {specification[0]: specification for specification in SPECS}


class SubmissionError(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def regular(path):
    path = Path(path)
    metadata = path.lstat()
    if path.resolve() != path or not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise SubmissionError('Expected an owned regular file without aliases: ' + str(path))
    return path


def sanitized_environment(environment=None):
    value = dict(os.environ if environment is None else environment)
    exact = {'PYTHONHOME', 'PYTHONPATH', 'LD_PRELOAD', 'VIRTUAL_ENV', 'SSH_AUTH_SOCK', 'SSH_AGENT_PID', 'KRB5CCNAME', 'GPG_AGENT_INFO', 'SLURM_JWT', 'SLURM_GET_USER_ENV', 'SLURM_EXPORT_ENV'}
    for name in list(value):
        upper = name.upper()
        if upper in exact or upper.startswith(('SBATCH_', 'SRUN_', 'AWS_', 'AZURE_')) or upper.endswith(('_TOKEN', '_PASSWORD', '_SECRET', '_API_KEY', '_ACCESS_KEY_ID', '_ACCESS_KEY', '_SESSION_TOKEN')):
            value.pop(name)
    return value


def inspect(arguments):
    result = subprocess.run(arguments, env=sanitized_environment(), capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise SubmissionError('Read-only scheduler query failed: ' + arguments[0])
    return result.stdout


def command_argv(root, specification):
    method, run_id, node, gpus, partition, qos = specification
    return ['env', '-u', 'CUDA_VISIBLE_DEVICES', '-u', 'PYTHONHOME', '-u', 'LD_PRELOAD', 'PATH=/usr/local/bin:/usr/bin:/bin', 'PYTHONPATH=' + str(root / 'src'), 'PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', 'NETRC=/dev/null', '/usr/bin/python3', '-B', '-m', 'taps_rebuild.pool', '--root', str(root), '--config', str(root / 'configs/reproduction.json'), '--run-id', run_id, '--expected-gpus', str(gpus), '--methods', method]


def sbatch_argv(root, specification):
    method, run_id, node, gpus, partition, qos = specification
    return ['sbatch', '--parsable', '--job-name=' + run_id, '--account=grad-students', '--partition=' + partition, '--qos=' + qos, '--nodes=1', '--nodelist=' + node, '--ntasks=1', '--cpus-per-task=64', '--mem=0', '--gres=gpu:a100:' + str(gpus), '--exclusive', '--time=12:00:00', '--requeue', '--open-mode=append', '--output=' + str(root / 'outputs/slurm' / (run_id + '-%j.out')), '--error=' + str(root / 'outputs/slurm' / (run_id + '-%j.err')), '--chdir=' + str(root), '--export=ALL', '--signal=B:USR1@120', '--wrap', shlex.join(['exec'] + command_argv(root, specification))]


def read_plan(root, path, expected_sha256):
    root = Path(root).absolute()
    if root.resolve() != root or not root.is_dir() or root.stat().st_uid != os.getuid():
        raise SubmissionError('Project root must be canonical and owned')
    if not re.fullmatch('[a-f0-9]{64}', expected_sha256):
        raise SubmissionError('An exact expected plan SHA256 is required')
    path = Path(path).absolute()
    contents = regular(path).read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected_sha256:
        raise SubmissionError('Prepared plan SHA256 differs')
    value = json.loads(contents)
    config = regular(root / 'configs/reproduction.json')
    if value.get('schema_version') != 1 or value.get('status') != 'prepared_not_submitted' or value.get('root') != str(root) or value.get('config') != str(config) or value.get('config_sha256_at_preparation') != pool.digest(config):
        raise SubmissionError('Prepared plan root, config or status differs')
    jobs = value.get('jobs', [])
    if len(jobs) != len(SPECS):
        raise SubmissionError('This helper accepts exactly the two reviewed reference jobs')
    for job, specification in zip(jobs, SPECS):
        keys = ('method', 'run_id', 'expected_node', 'expected_gpus', 'expected_partition', 'expected_qos')
        if any(job.get(key) != expected for key, expected in zip(keys, specification)) or job.get('command_argv') != command_argv(root, specification) or job.get('sbatch_argv') != sbatch_argv(root, specification):
            raise SubmissionError('Prepared job arguments differ from the reviewed fixed graph')
    return value


def source_preflight(root, config_path, jobs):
    original = json.loads(config_path.read_bytes())
    selected = pool.effective_config(original, [job['method'] for job in jobs], False, 9)
    prepared = pool.prerequisites(root, config_path, selected)
    if prepared['code'].get('code_dirty') is not False:
        raise SubmissionError('Commit and deploy the final clean source before submission')
    return {'code': prepared['code'], 'authored_source_hashes': prepared['authored_source_hashes'], 'config_sha256': pool.digest(config_path), 'plan_sha256': prepared['plan_sha256'], 'assets_receipt_sha256': prepared['receipts']['assets']['sha256'], 'sources_receipt_sha256': prepared['receipts']['sources']['sha256'], 'submitter_sha256': pool.digest(Path(__file__))}


def no_duplicate_paths(root, jobs):
    for job in jobs:
        run_id = job['run_id']
        for path in (root / 'outputs/runs' / run_id, root / 'outputs/submissions' / (run_id + '.json')):
            if path.exists() or path.is_symlink():
                raise SubmissionError('Existing run or submission receipt blocks duplicate launch: ' + run_id)


def requested_gpus(value):
    counts = re.findall(r'(?:^|,)(?:gres:gpu|gres/gpu)(?::[A-Za-z0-9_.-]+)?:(\d+)(?=\(|,|$)', value)
    if not counts and value not in ('N/A', '(null)', ''):
        raise SubmissionError('Unrecognized queued GPU resource format')
    return sum(map(int, counts))


def live_admission(root, config, jobs, query=inspect, accepted=None, selected_names=None):
    accepted = {} if accepted is None else accepted
    metadata = attach.holder_metadata(Path(config['legacy_runtime_root']) / 'cluster/deep10/current.json')
    holder = attach.verify_job(attach.fields(query(['scontrol', 'show', 'job', '-o', metadata['job_id']])), metadata['job_id'])
    if not metadata['successor']:
        raise SubmissionError('Protected holder successor must be present')
    successor = attach.verify_job(attach.fields(query(['scontrol', 'show', 'job', '-o', metadata['successor']])), metadata['successor'], pending=True, holder=metadata['job_id'])
    queue_text = query(['squeue', '--noheader', '--user', str(os.getuid()), '--format=%i|%j|%T|%P|%q|%N|%b'])
    rows, allocated, queued = [], 0, 0
    names = {job['run_id'] for job in jobs} if selected_names is None else selected_names
    for line in queue_text.splitlines():
        parts = line.split('|')
        if len(parts) != 7 or not parts[0].isdigit():
            raise SubmissionError('Unexpected owned job queue record')
        job_id, name, state, partition, qos, node, gres = parts
        if name in names and accepted.get(name) != job_id:
            raise SubmissionError('A queued job already uses a prepared run ID: ' + name)
        count = requested_gpus(gres)
        rows.append({'job_id': job_id, 'name': name, 'state': state, 'partition': partition, 'qos': qos, 'node': node, 'requested_gpus': count})
        if set(partition.split(',')) & {'high-priority', 'extreme-priority'}:
            if state in ('RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED'):
                allocated += count
            elif state == 'PENDING' and job_id != metadata['successor']:
                queued += count
            elif state != 'PENDING':
                raise SubmissionError('Unknown high-priority allocation state')
    if not any(row['job_id'] == metadata['job_id'] and row['state'] == 'RUNNING' and row['requested_gpus'] == 8 for row in rows):
        raise SubmissionError('Fresh user queue does not contain the protected running eight-GPU holder')
    if not set(accepted.values()) <= {row['job_id'] for row in rows}:
        raise SubmissionError('A previously accepted job is absent from the fresh queue; inspect before continuing')
    additional = sum(job['expected_gpus'] for job in jobs if job['expected_partition'] == 'high-priority')
    if allocated + queued + additional > 18:
        raise SubmissionError('Current and queued high-priority GPUs plus the new job exceed18')
    nodes = {}
    for job in jobs:
        raw = query(['scontrol', 'show', 'node', '-o', job['expected_node']])
        node = attach.fields(raw)
        count = attach.gpu_count(node.get('CfgTRES', ''))
        if node.get('NodeName') != job['expected_node'] or node.get('State') != 'IDLE' or node.get('CPUAlloc') != '0' or count != job['expected_gpus'] or not re.fullmatch('gpu:a100:' + str(job['expected_gpus']) + r'(?:\([^)]*\))?', node.get('Gres', '')) or attach.gpu_count(node.get('AllocTRES', '')) not in (0, None) or int(node.get('CPUTot', '0')) < 64:
            raise SubmissionError('Target node is not idle with its complete configured GPU graph')
        if job['expected_partition'] not in node.get('Partitions', '').split(','):
            raise SubmissionError('Target node does not belong to the reviewed partition')
        nodes[job['expected_node']] = {'record': node, 'raw': raw}
    return {'observed_at_utc': now(), 'holder': holder, 'successor': successor, 'owned_queue': rows, 'queue_raw': queue_text, 'running_or_reserved_high_priority_gpus': allocated, 'other_pending_high_priority_gpus': queued, 'new_high_priority_gpus': additional, 'high_priority_bound': allocated + queued + additional, 'high_priority_allowance': 18, 'nodes': nodes, 'gpu_hardware_not_queried_outside_allocation': True}


def verify_submitted(text, job, job_id):
    value = attach.fields(text)
    owner = re.fullmatch(r'[^()]+\((\d+)\)', value.get('UserId', ''))
    if not owner or int(owner.group(1)) != os.getuid() or value.get('JobId') != job_id:
        raise SubmissionError('Submitted job ownership or identity differs')
    expected = {'JobName': job['run_id'], 'Account': 'grad-students', 'Partition': job['expected_partition'], 'QOS': job['expected_qos'], 'ReqNodeList': job['expected_node'], 'TimeLimit': '12:00:00', 'Requeue': '1'}
    if any(value.get(key) != wanted for key, wanted in expected.items()) or value.get('JobState') not in ('PENDING', 'RUNNING', 'CONFIGURING') or value.get('NumNodes') not in ('1', '1-1') or int(value.get('NumCPUs', '0')) < 64:
        raise SubmissionError('Accepted job resources differ from the reviewed plan')
    if value['JobState'] != 'PENDING' and value.get('NodeList') != job['expected_node']:
        raise SubmissionError('Accepted job is running on another node')
    if attach.gpu_count(value.get('AllocTRES') or value.get('ReqTRES') or value.get('TRES', '')) != job['expected_gpus']:
        raise SubmissionError('Accepted job GPU count differs')
    return value


def durable_json(path, value):
    pool.atomic_json(path, value)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def submission_lock(root):
    directory = root / 'outputs/submissions'
    directory.mkdir(parents=True, exist_ok=True)
    if directory.resolve() != directory or directory.stat().st_uid != os.getuid():
        raise SubmissionError('Submission directory must be owned and canonical')
    path = directory / '.lock'
    if path.is_symlink():
        raise SubmissionError('Submission lock cannot be a symlink')
    with path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SubmissionError('Another reference submission helper is active') from error
        yield


def submit_command(arguments, environment, root):
    return subprocess.run(arguments, env=environment, cwd=root, capture_output=True, text=True, timeout=60)


def run(root, plan_path, expected_sha256, submit=False, query=inspect, prepare=source_preflight, submitter=submit_command, methods=None):
    if os.getuid() != attach.UID:
        raise SubmissionError('Run as the verified cluster UID')
    root = Path(root).absolute()
    plan_path = Path(plan_path).absolute()
    plan = read_plan(root, plan_path, expected_sha256)
    if methods not in (None, 'eagle3', 'hass'):
        raise SubmissionError('Select only eagle3 or hass; omit --methods for both reviewed jobs')
    jobs = [job for job in plan['jobs'] if methods is None or job['method'] == methods]
    names = {job['run_id'] for job in jobs}
    config = json.loads((root / 'configs/reproduction.json').read_bytes())
    no_duplicate_paths(root, jobs)
    deployment = prepare(root, root / 'configs/reproduction.json', jobs)
    admission = live_admission(root, config, jobs, query, selected_names=names)
    if not submit:
        return {'status': 'validated_not_submitted', 'selected_methods': [job['method'] for job in jobs], 'plan_sha256': expected_sha256, 'deployment': deployment, 'admission': admission, 'sbatch_argv': [sbatch_argv(root, SPEC_BY_METHOD[job['method']]) for job in jobs], 'holder_and_successor_modified': False}
    receipts, accepted = [], {}
    with submission_lock(root):
        read_plan(root, plan_path, expected_sha256)
        no_duplicate_paths(root, jobs)
        log_directory = root / 'outputs/slurm'
        log_directory.mkdir(parents=True, exist_ok=True)
        if log_directory.resolve() != log_directory or log_directory.stat().st_uid != os.getuid():
            raise SubmissionError('Persistent log directory must be canonical and owned')
        for index, job in enumerate(jobs):
            read_plan(root, plan_path, expected_sha256)
            current = prepare(root, root / 'configs/reproduction.json', jobs)
            if current != deployment:
                raise SubmissionError('Source or preparation identity changed during sequential submission')
            no_duplicate_paths(root, jobs[index:])
            admission = live_admission(root, config, jobs[index:], query, accepted, names)
            arguments = sbatch_argv(root, SPEC_BY_METHOD[job['method']])
            path = root / 'outputs/submissions' / (job['run_id'] + '.json')
            receipt = {'schema_version': 1, 'status': 'submitting', 'run_id': job['run_id'], 'method': job['method'], 'plan_path': str(plan_path), 'plan_sha256': expected_sha256, 'submission_argv': arguments, 'deployment': deployment, 'resource_admission': admission, 'created_at_utc': now(), 'holder_and_successor_modified': False, 'scientific_result': False, 'submission_implies_success': False}
            durable_json(path, receipt)
            try:
                result = submitter(arguments, sanitized_environment(), root)
                receipt.update(sbatch_stdout=result.stdout, sbatch_stderr=result.stderr, sbatch_returncode=result.returncode)
                if result.returncode != 0:
                    raise SubmissionError('sbatch failed or was inconclusive; inspect the saved receipt and queue before any manual action')
                match = re.fullmatch(r'([1-9][0-9]*)(?:;([A-Za-z0-9_.-]+))?', result.stdout.strip())
                if not match:
                    raise SubmissionError('sbatch response is ambiguous; no automatic retry is allowed')
                job_id, cluster = match.groups()
                accepted[job['run_id']] = job_id
                receipt.update(status='submitted_pending_verification', job_id=job_id, sbatch_cluster=cluster, submitted_at_utc=now())
                durable_json(path, receipt)
                raw = query(['scontrol', 'show', 'job', '-o', job_id])
                receipt.update(status='submitted_verified', live_job=verify_submitted(raw, job, job_id), live_job_raw=raw, verified_at_utc=now(), stdout_path=str(root / 'outputs/slurm' / (job['run_id'] + '-' + job_id + '.out')), stderr_path=str(root / 'outputs/slurm' / (job['run_id'] + '-' + job_id + '.err')))
                durable_json(path, receipt)
                receipts.append({'path': str(path), 'sha256': pool.digest(path), 'job_id': job_id, 'run_id': job['run_id']})
            except BaseException as error:
                receipt.update(status='submitted_verification_failed' if receipt.get('job_id') else 'failed_or_uncertain', failure={'error_type': type(error).__name__, 'message': str(error), 'observed_at_utc': now()}, automatic_retry_performed=False, cancellation_performed=False)
                durable_json(path, receipt)
                raise
    return {'status': 'submitted_verified', 'selected_methods': [job['method'] for job in jobs], 'receipts': receipts, 'holder_and_successor_modified': False}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha', required=True)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--methods', choices=('eagle3', 'hass'))
    args = parser.parse_args(argv)
    if args.root.resolve() != PROJECT:
        raise SubmissionError('Use the project that contains this reviewed submission helper')
    print(json.dumps(run(args.root, args.plan, args.expected_plan_sha, submit=args.submit, methods=args.methods), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
