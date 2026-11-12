import argparse
from pathlib import Path

from . import native_training as native, pool
from .pilot_cache import read, require
from .pilot_study import verify_identity
from .pilot_train import admission


CHECKS = {'schedules', 'ce_gradients', 'outside_mask_gradients', 'native_step_parity', 'restart_parity', 'suffix_probe', 'raw_zero_state_reconstruction', 'all13_named_hooks_and_corrected_policy', 'rotary_priming', 'all13_named_adam_and_fp32_updates', 'stock_export_compatibility', 'runtime_envelope'}


def descriptor(path):
    return {'path': str(Path(path).resolve()), 'sha256': pool.digest(path)}


def binding(study):
    manifest = verify_identity(read(Path(study) / 'manifest.json'))
    plan = verify_identity(read(manifest['plans'][0]['path']))
    return plan, {'study_manifest_sha256': pool.digest(Path(study) / 'manifest.json'), 'source_sha256': plan['source_sha256'], 'recipe_sha256': plan['recipe_sha256'], 'cache_identity_sha256': plan['cache_identity_sha256'], 'eligibility_identity_sha256': plan['eligibility_identity_sha256'], 'hardware': plan['hardware'], 'vocabulary_identity_sha256': plan['vocabulary_identity_sha256']}


def verify_inventory(inventory):
    require(bool(inventory), 'Evidence has no retained artifact inventory')
    for name, item in inventory.items():
        require(Path(name).stat().st_size == item['size_bytes'] and pool.digest(name) == item['sha256'], 'Retained evidence content changed: ' + name)


def prepare(study, audit_path, teacher_path, teacher_review_path, output):
    native.require_slurm()
    output = Path(output)
    require(not output.exists(), 'Choose a fresh qualification report; reviewed evidence cannot be overwritten')
    plan, bound = binding(study)
    audit, teacher, teacher_review = read(audit_path), read(teacher_path), read(teacher_review_path)
    require(audit['status'] == 'all_checked_evidence_passed_requires_independent_review' and set(audit['checks']) == CHECKS and all(value == 'passed' for value in audit['checks'].values()) and audit['automatic_admission_emitted'] is False and audit['project_helpers_imported'] is False, 'The complete independent numerical/state audit has not passed')
    manifest_path = str((Path(study) / 'manifest.json').resolve())
    require(audit['artifact_inventory'][manifest_path]['sha256'] == bound['study_manifest_sha256'], 'Independent replay belongs to another study')
    require(teacher['status'] == 'passed_independent_teacher_worker_canonical_native_replay' and teacher['all_bound_input_bytes_stable'] is True and teacher_review['decision'] == 'approved' and teacher_review['report']['sha256'] == pool.digest(teacher_path) and teacher_review['artifact_inventory_sha256'] == teacher['artifact_inventory_sha256'], 'Teacher provenance lacks an exact independent evidence review')
    require({value['population'] for value in teacher['populations']} == {'chat', 'math_code'} and all(value['status'] == 'passed' for value in teacher['populations']), 'Teacher provenance population coverage differs')
    inventory = dict(audit['artifact_inventory'])
    for name, item in teacher['artifact_inventory'].items():
        require(name not in inventory or inventory[name] == item, 'Conflicting retained artifact identities')
        inventory[name] = item
    for path in (audit_path, teacher_path, teacher_review_path, __file__):
        path = str(Path(path).resolve())
        inventory[path] = {'sha256': pool.digest(path), 'size_bytes': Path(path).stat().st_size}
    verify_inventory(inventory)
    result = {'status': 'passed', 'scope': 'source-bound24 common-ancestor matched-budget scientific pilots only; no native scratch/generalist or evaluation admission', 'binding': bound, 'checks': {key: 'passed' for key in plan['recipe']['admission_required']}, 'independent_check_results': audit['checks'], 'independent_numerical_audit': descriptor(audit_path), 'teacher_provenance_audit': descriptor(teacher_path), 'teacher_provenance_review': descriptor(teacher_review_path), 'evidence_files': inventory, 'evidence_inventory_identity_sha256': pool.identity(inventory), 'compiler_sha256': pool.digest(__file__), 'scientific_training_admitted': False, 'independent_review_still_required': True}
    pool.atomic_json(output, result)
    return result


def integrate(study, report_path, review_path):
    native.require_slurm()
    study = Path(study)
    require(not (study / 'admission.json').exists(), 'A study admission already exists')
    plan, bound = binding(study)
    report, review = read(report_path), read(review_path)
    require(report['status'] == 'passed' and report['binding'] == bound and set(report['checks']) == set(plan['recipe']['admission_required']) and all(value == 'passed' for value in report['checks'].values()), 'Pilot qualification report differs')
    require(review.get('scientific_training_admitted') is True and review.get('qualification_report_sha256') == pool.digest(report_path) and review.get('binding') == bound, 'Independent review has not admitted the exact report and study')
    verify_inventory(report['evidence_files'])
    receipt = {'scientific_training_admitted': True, 'scope': report['scope'], 'binding': bound, 'qualification_report': descriptor(report_path), 'independent_review': descriptor(review_path)}
    receipt['identity_sha256'] = pool.identity(receipt)
    pool.atomic_json(study / 'admission.json', receipt)
    admission(plan)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'integrate'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--teacher-review', type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    result = prepare(args.study, args.audit, args.teacher, args.teacher_review, args.report) if args.action == 'prepare' else integrate(args.study, args.report, args.review)
    import json
    print(json.dumps({'status': result.get('status', 'independently_admitted'), 'scientific_training_admitted': result['scientific_training_admitted']}))


if __name__ == '__main__':
    main()
