import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
from datetime import datetime, timezone


OLD_RUN = 'official-eagle3-20261109-v1'
NEW_RUN = 'official-eagle3-20261109-v2'
COMMIT = 'a9855371b638fdd13a1f8125a0041759958aefbf'
SOURCE = '6375933ffe6e688eb4eb032360eabe519efadd01e13233a56869a022bf6f2caa'
HELPER = '7870ffe3f2638a35063fe8ddc22e4399ae901d283fdf834670fdb8fe45be3f44'
CONFIG = '51e82dae1296d16a41bb9cab9ce1042eac4d4b775b97df42866e65ff32e2cd16'
PRIOR = {
    'outputs/submissions/' + OLD_RUN + '.json': '1459182c6aa8eada7a34dcbef1aae829d47f5e15c8b482d856557d328cf4ab1c',
    'outputs/slurm/' + OLD_RUN + '-2212926.out': 'b7d97f4029983c9201141cfca63910de764290fd6ebf618fbe55420eaf689493',
    'outputs/slurm/' + OLD_RUN + '-2212926.err': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def owned(path):
    value = path.lstat()
    if path.resolve() != path or not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid():
        raise ValueError('Expected owned regular evidence without aliases')
    return path


def validate(root, operation, expected):
    if root.resolve() != root or root.stat().st_uid != os.getuid() or os.getuid() != 29562:
        raise ValueError('Use the canonical project as the verified cluster UID')
    if digest(owned(operation)) != expected:
        raise ValueError('Operational payload SHA differs')
    value = json.loads(operation.read_bytes())
    if value['root'] != str(root) or value['new_run_id'] != NEW_RUN:
        raise ValueError('Operational root or run differs')
    if value.get('operational_submitter_sha256') != digest(owned(Path(__file__).absolute())):
        raise ValueError('Operational submitter SHA differs from the reviewed payload')
    path = owned(root / 'experiments/official-reproduction-20261109/submit_reference.py')
    if digest(path) != HELPER or digest(owned(root / 'source-identity.json')) != SOURCE or digest(owned(root / 'configs/reproduction.json')) != CONFIG:
        raise ValueError('Frozen helper, source identity or config changed')
    spec = importlib.util.spec_from_file_location('frozen_reference_submitter', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    for relative, sha in PRIOR.items():
        if digest(owned(root / relative)) != sha:
            raise ValueError('Failed predecessor evidence changed')
    prior = json.loads((root / ('outputs/submissions/' + OLD_RUN + '.json')).read_bytes())
    if prior['job_id'] != '2212926' or prior['deployment']['code']['code_commit'] != COMMIT:
        raise ValueError('Predecessor submission identity differs')
    failure = json.loads((root / ('outputs/slurm/' + OLD_RUN + '-2212926.out')).read_bytes())
    if failure != {'status': 'failed', 'error_type': 'FileNotFoundError', 'error': "[Errno 2] No such file or directory: 'scontrol'"}:
        raise ValueError('Predecessor did not fail in the expected pre-GPU lookup')
    old_directory = root / 'outputs/runs' / OLD_RUN
    if old_directory.exists() or old_directory.is_symlink():
        raise ValueError('Predecessor unexpectedly created frozen inputs or raw output')
    accounting = helper.inspect(['sacct', '-j2212926', '--format=JobIDRaw,State,ExitCode,NodeList,ElapsedRaw,AllocTRES', '-P', '-n'])
    matches = [line.split('|') for line in accounting.splitlines() if line.startswith('2212926|')]
    if len(matches) != 1 or matches[0][1:5] != ['FAILED', '1:0', 'deep-chungus-3', '1']:
        raise ValueError('Predecessor is not the observed terminal failed one-second attempt')
    original = helper.sbatch_argv(root, helper.SPEC_BY_METHOD['eagle3'])
    expected_arguments = [item.replace(OLD_RUN, NEW_RUN).replace('PATH=/usr/bin:/bin', 'PATH=/usr/local/bin:/usr/bin:/bin') for item in original]
    if prior['submission_argv'] != original or value['sbatch_argv'] != expected_arguments or value['command_argv'] != [item.replace(OLD_RUN, NEW_RUN).replace('PATH=/usr/bin:/bin', 'PATH=/usr/local/bin:/usr/bin:/bin') for item in helper.command_argv(root, helper.SPEC_BY_METHOD['eagle3'])] or expected_arguments[-1] != shlex.join(['exec'] + value['command_argv']):
        raise ValueError('Payload must change only the run ID, log names and PATH')
    job = {'method': 'eagle3', 'run_id': NEW_RUN, 'expected_node': 'deep-chungus-3', 'expected_gpus': 9, 'expected_partition': 'high-priority', 'expected_qos': 'spec-priority'}
    helper.no_duplicate_paths(root, [job])
    deployment = helper.source_preflight(root, root / 'configs/reproduction.json', [job])
    if deployment['code']['code_commit'] != COMMIT or deployment['code'].get('deployed_source_identity_sha256') != SOURCE:
        raise ValueError('Frozen deployment changed')
    config = json.loads((root / 'configs/reproduction.json').read_bytes())
    admission = helper.live_admission(root, config, [job])
    return helper, value, job, deployment, admission, accounting


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--operation', type=Path, required=True)
    parser.add_argument('--expected-operation-sha', required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args(argv)
    root, operation = args.root.absolute(), args.operation.absolute()
    helper, value, job, deployment, admission, accounting = validate(root, operation, args.expected_operation_sha)
    if not args.submit:
        print(json.dumps({'status': 'validated_not_submitted', 'sbatch_argv': value['sbatch_argv'], 'deployment': deployment, 'admission': admission, 'prior_accounting': accounting}), flush=True)
        return
    with helper.submission_lock(root):
        helper, value, job, deployment, admission, accounting = validate(root, operation, args.expected_operation_sha)
        path = root / 'outputs/submissions' / (NEW_RUN + '.json')
        receipt = {'status': 'submitting', 'run_id': NEW_RUN, 'method': 'eagle3', 'submission_argv': value['sbatch_argv'], 'operation_sha256': args.expected_operation_sha, 'operational_submitter_sha256': digest(owned(Path(__file__).absolute())), 'deployment': deployment, 'resource_admission': admission, 'prior_failed_job_id': '2212926', 'prior_evidence_sha256': PRIOR, 'prior_accounting': accounting, 'prior_gpu_scope': 'Nine GPUs were allocated for one second; the exact failure precedes GPU inventory/model work and no frozen run or raw output was created.', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'scientific_result': False, 'automatic_retry_performed': False, 'holder_and_successor_modified': False}
        helper.durable_json(path, receipt)
        try:
            archive = root / 'outputs/submissions' / (NEW_RUN + '-predecessor')
            archive.mkdir(exist_ok=False)
            for relative in PRIOR:
                contents = (root / relative).read_bytes()
                with (archive / Path(relative).name).open('xb') as stream:
                    stream.write(contents)
                    stream.flush()
                    os.fsync(stream.fileno())
            result = helper.submit_command(value['sbatch_argv'], helper.sanitized_environment(), root)
            receipt.update(sbatch_stdout=result.stdout, sbatch_stderr=result.stderr, sbatch_returncode=result.returncode)
            match = re.fullmatch(r'([1-9][0-9]*)(?:;([A-Za-z0-9_.-]+))?', result.stdout.strip()) if result.returncode == 0 else None
            if match is None:
                raise ValueError('Submission failed or is ambiguous; inspect without automatic retry')
            job_id, cluster = match.groups()
            receipt.update(status='submitted_pending_verification', job_id=job_id, sbatch_cluster=cluster)
            helper.durable_json(path, receipt)
            raw = helper.inspect(['scontrol', 'show', 'job', '-o', job_id])
            receipt.update(status='submitted_verified', live_job=helper.verify_submitted(raw, job, job_id), live_job_raw=raw, verified_at_utc=datetime.now(timezone.utc).isoformat())
            helper.durable_json(path, receipt)
            print(json.dumps({'status': receipt['status'], 'job_id': job_id, 'receipt': str(path), 'receipt_sha256': digest(path)}), flush=True)
        except BaseException as error:
            receipt.update(status='submitted_verification_failed' if receipt.get('job_id') else 'failed_or_uncertain', failure={'type': type(error).__name__, 'message': str(error)}, cancellation_performed=False)
            helper.durable_json(path, receipt)
            raise


if __name__ == '__main__':
    main()
