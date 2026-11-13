import argparse
import os
from pathlib import Path

from . import native_training as native, pilot_admission as prior, pool
from .pilot_cache import read, require


CHECKS = prior.CHECKS | {'teacher_buffer_policy', 'teacher_gate_prefix208'}
ANALYZER_SHA256 = 'a65a4848ef351860c87a1ed1d4f911bc7c724e58e33411b709fb6e73398ff3b6'
STUDY_ID = 'rq1-pilot-24-20261113-v3'
RECIPE_SHA256 = '2e0e7d0a685d6b29271676cd7ffbcbdc5b42d977df97d8eb398dbf4a73939773'
WORKER_SHA256 = 'f700df226d9bd0ea3391868c24e6780efcfde0be3967bda2dba354e7cc699a4a'


def validate_audit(plan, audit):
    require(plan['study_id'] == STUDY_ID and plan['recipe_sha256'] == RECIPE_SHA256 and plan['source_sha256']['pilot_train.py'] == WORKER_SHA256, 'Pristine teacher compiler requires the exact fresh study and source')
    require(audit['status'] == 'all_checked_evidence_passed_requires_independent_review' and set(audit['checks']) == CHECKS and all(value == 'passed' for value in audit['checks'].values()) and audit['analyzer_v9_sha256'] == ANALYZER_SHA256 and audit['scientific_training_admitted'] is False and audit['automatic_admission_emitted'] is False and audit['project_helpers_imported'] is False, 'Complete fourteen-check independent V9 evidence has not passed')
    proof = audit['teacher_source_and_actual_helper_proof']
    require(proof['status'] == 'passed' and proof['study_id'] == STUDY_ID and proof['source_sha256'] == plan['source_sha256'] and proof['recipe_sha256'] == RECIPE_SHA256 and len(proof['actual_helper_cases']) == 40 and proof['unchanged_schedule_files'] == 96, 'V9 actual pristine teacher helper/source coverage differs')
    prefix = audit['teacher_gate_prefix208']
    require(prefix['status'] == 'passed' and [value['rank'] for value in prefix['ranks']] == list(range(8)) and all(value['committed_updates'] == 208 for value in prefix['ranks']), 'V9 fresh208-update failure-case coverage differs')
    require(audit['teacher_buffer_initializations']['status'] == 'passed' and len(audit['teacher_buffer_initializations']['attempt_ranks']) == 80, 'V9 standard teacher initialization coverage differs')
    require(pool.identity(audit['artifact_inventory']) == audit['artifact_inventory_sha256'] and any(Path(name).name == 'independent_pilot_qualification_audit_v9_20261113.py' and item['sha256'] == ANALYZER_SHA256 for name, item in audit['artifact_inventory'].items()), 'V9 retained inventory or actual analyzer source differs')


def prepare(study, audit_path, teacher_path, teacher_review_path, output):
    native.require_slurm()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Pristine teacher evidence compilation requires CPU Slurm')
    output = Path(output)
    require(not output.exists(), 'Choose a fresh pristine teacher qualification report')
    plan, bound = prior.binding(study)
    audit, teacher, review = map(read, (audit_path, teacher_path, teacher_review_path))
    validate_audit(plan, audit)
    manifest = read(Path(study) / 'manifest.json')
    item = manifest['plans'][0]
    require(pool.digest(item['path']) == item['sha256'] and audit['artifact_inventory'][str(Path(item['path']).resolve())]['sha256'] == item['sha256'] and audit['artifact_inventory'][str((Path(study) / 'manifest.json').resolve())]['sha256'] == bound['study_manifest_sha256'], 'V9 retained study and first plan bytes differ')
    require(teacher['status'] == 'passed_independent_teacher_worker_canonical_native_replay' and teacher['all_bound_input_bytes_stable'] is True and review['decision'] == 'approved' and review['report']['sha256'] == pool.digest(teacher_path) and review['artifact_inventory_sha256'] == teacher['artifact_inventory_sha256'], 'Pristine pilot teacher provenance lacks exact independent approval')
    require({value['population'] for value in teacher['populations']} == {'chat', 'math_code'} and all(value['status'] == 'passed' for value in teacher['populations']), 'Teacher provenance population coverage differs')
    inventory = dict(audit['artifact_inventory'])
    for name, value in teacher['artifact_inventory'].items():
        require(name not in inventory or inventory[name] == value, 'Conflicting pristine pilot evidence identities')
        inventory[name] = value
    for name in (audit_path, teacher_path, teacher_review_path, __file__, prior.__file__):
        path = str(Path(name).resolve())
        inventory[path] = {'sha256': pool.digest(path), 'size_bytes': Path(path).stat().st_size}
    prior.verify_inventory(inventory)
    result = {'admission_schema': 'pristine-teacher-pilot-v1', 'status': 'passed', 'scope': 'Exact fresh24common-ancestor pilots with pristine FP32teacher buffers; no failed-run resume, native scratch, inference evaluation or quality admission', 'binding': bound, 'checks': {key: 'passed' for key in plan['recipe']['admission_required']}, 'independent_check_results': audit['checks'], 'independent_numerical_audit': prior.descriptor(audit_path), 'independent_analyzer_sha256': ANALYZER_SHA256, 'teacher_provenance_audit': prior.descriptor(teacher_path), 'teacher_provenance_review': prior.descriptor(teacher_review_path), 'evidence_files': inventory, 'evidence_inventory_identity_sha256': pool.identity(inventory), 'compiler_sha256': pool.digest(__file__), 'scientific_training_admitted': False, 'independent_review_still_required': True}
    pool.atomic_json(output, result)
    return result


def integrate(study, report_path, review_path):
    report = read(report_path)
    require(report.get('admission_schema') == 'pristine-teacher-pilot-v1' and report.get('independent_analyzer_sha256') == ANALYZER_SHA256 and set(report['independent_check_results']) == CHECKS and all(value == 'passed' for value in report['independent_check_results'].values()), 'Pristine pilot integration requires the exact fourteen-check report')
    plan, bound = prior.binding(study)
    require(plan['study_id'] == STUDY_ID and plan['recipe_sha256'] == RECIPE_SHA256 and plan['source_sha256']['pilot_train.py'] == WORKER_SHA256 and report['binding'] == bound, 'Pristine pilot integration study/source differs')
    return prior.integrate(study, report_path, review_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'integrate'))
    for key in ('study', 'report'):
        parser.add_argument('--' + key, type=Path, required=True)
    for key in ('audit', 'teacher', 'teacher-review', 'review'):
        parser.add_argument('--' + key, type=Path)
    args = parser.parse_args()
    result = prepare(args.study, args.audit, args.teacher, args.teacher_review, args.report) if args.action == 'prepare' else integrate(args.study, args.report, args.review)
    import json
    print(json.dumps({'status': result.get('status', 'independently_admitted'), 'scientific_training_admitted': result['scientific_training_admitted']}))
