import argparse
from contextvars import ContextVar
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from types import ModuleType


if __package__:
    package = __package__
else:
    package = '_taps_full_body_bridge'
    module = ModuleType(package)
    module.__path__ = [str(Path(__file__).parent)]
    sys.modules[package] = module
body = importlib.import_module(package + '.native_full_body')
native = body.native
BASE_GATE = body.gate
BASE_DOCUMENT = body.document
BASE_ROUTER = body.AuditRouter
ACTIVE = ContextVar('validated_native_runtime_interpreter', default=None)
ENTRY = ContextVar('native_v4_entry', default=None)
STAGE = 'corrected-full-v4'


def bridge_binding(root, config):
    bridge = config.get('full_body_v3', {}).get('admission_bridge')
    body.require(isinstance(bridge, dict) and set(bridge) == {'sha256', 'runtime_interpreter'} and bridge['sha256'] == native.digest(__file__), 'Native v4 bridge source binding differs')
    descriptor = bridge['runtime_interpreter']
    body.require(isinstance(descriptor, dict) and set(descriptor) == {'path', 'sha256', 'size_bytes'}, 'Native v4 requires one exact runtime interpreter descriptor')
    actual = (Path(root) / config['runtime_python']).resolve()
    body.require(str(actual) == descriptor['path'] and Path(descriptor['path']).resolve() == actual, 'Native v4 exception is restricted to the resolved declared interpreter')
    validate_interpreter(descriptor)
    return bridge


def validate_interpreter(descriptor):
    path = Path(descriptor['path'])
    body.require(path.is_file() and path.stat().st_size == descriptor['size_bytes'] and native.digest(path) == descriptor['sha256'], 'Native v4 declared interpreter bytes changed')


def validate_report(root, report, bridge):
    body.require(report.get('admission_bridge') == bridge, 'Native v4 original report bridge binding differs')
    descriptor = bridge['runtime_interpreter']
    expected = {key: descriptor[key] for key in ('sha256', 'size_bytes')}
    body.require(report.get('evidence_files', {}).get(descriptor['path']) == expected, 'Native v4 original report lacks the exact observed interpreter')
    allowed = (Path(root).resolve(), (Path(root) / 'models').resolve(), (Path(root) / 'envs/native-eagle3').resolve())
    outside = {name for name in report['evidence_files'] if not any(Path(name).resolve().is_relative_to(base) for base in allowed)}
    body.require(outside == {descriptor['path']}, 'Native v4 rejects any other external artifact')


def document(root, descriptor, label):
    original = BASE_DOCUMENT(root, descriptor, label)
    active = ACTIVE.get()
    if active is None or label not in ('corrected qualification', 'corrected profile'):
        return original
    key = label.removeprefix('corrected ')
    body.require(descriptor == active['reports'][key], 'Native v4 delegated report descriptor differs')
    validate_report(root, original, active['bridge'])
    result = dict(original)
    result['evidence_files'] = dict(original['evidence_files'])
    del result['evidence_files'][active['bridge']['runtime_interpreter']['path']]
    return result


def gate(root, config):
    body.require(config.get('pipeline_stage') == STAGE, 'This admission bridge accepts corrected-full-v4 only')
    bridge = bridge_binding(root, config)
    approval = BASE_DOCUMENT(root, config['full_body_v3']['scientific_admission'], 'native v4 original approval')
    body.require(approval.get('admission_bridge') == bridge, 'Native v4 approval bridge binding differs')
    review = BASE_DOCUMENT(root, approval['independent_review'], 'native v4 original independent review')
    body.require(review.get('scientific_training_admitted') is True and review.get('admission_bridge') == bridge, 'Native v4 exact interpreter exception lacks independent approval')
    reports = {key: approval[key] for key in ('qualification', 'profile')}
    for key, descriptor in reports.items():
        validate_report(root, BASE_DOCUMENT(root, descriptor, 'native v4 original ' + key), bridge)
    token = ACTIVE.set({'bridge': bridge, 'reports': reports})
    try:
        result = BASE_GATE(root, {**config, 'pipeline_stage': 'corrected-full-v3'})
        validate_interpreter(bridge['runtime_interpreter'])
        record_entry(config, bridge, approval)
        return {**result, 'admission_bridge': bridge, 'stage': STAGE, 'external_file_scope': 'only the independently observed resolved declared runtime interpreter; original full inventories retained'}
    finally:
        ACTIVE.reset(token)


def record_entry(config, bridge, approval):
    entry = ENTRY.get()
    if entry is None:
        return
    path = Path(entry['plan']).resolve()
    plan = json.loads(path.read_bytes())
    identity = hashlib.sha256(native.canonical({key: value for key, value in plan.items() if key != 'plan_identity_sha256'})).hexdigest()
    body.require(identity == plan['plan_identity_sha256'] and plan['config'] == config and plan['frozen_files']['inputs/native_full_body_bridge.py'] == bridge['sha256'], 'Native v4 entry plan/bridge identity differs')
    rank = os.environ.get('RANK', 'cpu')
    job = os.environ.get('SLURM_JOB_ID', '0')
    attempt = os.environ.get('TAPS_NATIVE_ATTEMPT_ID', '0')
    body.require((rank == 'cpu' or rank.isdigit()) and re.fullmatch('[0-9]+(?:[._][0-9]+)*', job) and attempt.isdigit(), 'Native v4 entry identifier differs')
    directory = Path(plan['run_dir']) / 'admission-bridge'
    directory.mkdir(exist_ok=True)
    destination = directory / (entry['phase'] + '-' + rank + '-' + job + '-' + attempt + '.json')
    receipt = {'status': 'passed_native_v4_entry_admission_only', 'scientific_training_admitted': True, 'stage': STAGE, 'plan_identity_sha256': identity, 'plan_path': str(path), 'qualified_numerical_source_sha256': body.sources(), 'admission_bridge': bridge, 'qualification': approval['qualification'], 'profile': approval['profile'], 'independent_review': approval['independent_review'], 'rank': rank, 'slurm_job_id': job, 'attempt': attempt, 'scope': 'Separate v4 gate receipt; original seven-file core observations and their v3-only admission field remain unchanged; no execution or quality completion claim'}
    if destination.exists():
        body.require(json.loads(destination.read_bytes()) == receipt, 'Existing native v4 entry receipt differs')
    else:
        native.atomic_json(destination, receipt)


class AuditRouter(BASE_ROUTER):
    def run(self, command, *args, **options):
        if isinstance(command, list) and len(command) >= 3 and Path(str(command[2])).name == 'native_training.py':
            body.require(len(command) == 6 and command[1] == '-B' and command[3:5] == ['audit', '--plan'], 'Unexpected native v4 audit invocation')
            plan = Path(command[5]).resolve()
            body.require(Path(command[2]).resolve() == plan.parent / 'inputs/native_training.py', 'Native v4 audit differs from the frozen run')
            command = [*command[:2], str(plan.parent / 'inputs/native_full_body_bridge.py'), *command[3:]]
            return self.delegate.run(command, *args, **options)
        return super().run(command, *args, **options)


def install():
    body.gate, body.document, body.AuditRouter = gate, document, AuditRouter
    body.install()


def prepare(root, config, train, validation, directory):
    install()
    plan = body.prepare(root, config, train, validation, directory)
    directory = Path(plan['run_dir'])
    path = directory / 'inputs/native_full_body_bridge.py'
    shutil.copyfile(__file__, path)
    plan['frozen_files']['inputs/native_full_body_bridge.py'] = native.digest(path)
    previous = str(directory / 'inputs/native_full_body.py')
    plan['command'][plan['command'].index(previous)] = str(path)
    plan['native_recipe']['admission_bridge'] = plan['config']['full_body_v3']['admission_bridge']
    plan['compatibility_changes'].append('Separately reviewed admission-only v4 bridge validates the sole external declared interpreter; qualified seven-file numerical core is unchanged')
    plan.pop('plan_identity_sha256')
    plan['plan_identity_sha256'] = hashlib.sha256(native.canonical(plan)).hexdigest()
    native.atomic_json(directory / 'plan.json', plan)
    native.atomic_json(Path(plan['trained_models']['resolved_path']) / 'native-plan.json', {'plan_identity_sha256': plan['plan_identity_sha256'], 'plan_path': str(directory / 'plan.json')})
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'audit', 'launch', 'resume', 'worker'))
    for key in ('plan', 'root', 'config', 'train', 'validation', 'run-dir'):
        parser.add_argument('--' + key, type=Path)
    args = parser.parse_args()
    install()
    token = ENTRY.set({'plan': args.plan, 'phase': args.action} if args.plan else None)
    try:
        if args.action == 'prepare':
            print(json.dumps(prepare(args.root, args.config, args.train, args.validation, args.run_dir), indent=2))
        elif args.action == 'audit':
            print(json.dumps(native.audit(args.plan), indent=2))
        elif args.action == 'worker':
            body.worker(args.plan)
        else:
            native.launch(args.plan, resume=args.action == 'resume')
    finally:
        ENTRY.reset(token)


if __name__ == '__main__':
    main()
