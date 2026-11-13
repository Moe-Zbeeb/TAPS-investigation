import argparse
import hashlib
import json
from pathlib import Path

from . import native_full_body as corrected, native_training as native, pool
from .native_full_body_queue import descriptor
from .pilot_cache import read, require


RAW_CHECKS = {'all13_backward_dynamic_scaler_windows', 'full_native_pre_step_gradient_and_state_parity', 'eight_rank_four_vs_two_plus_two_final_raw_state', 'all13_named_raw_adam', 'maximum_shape_2048_2054', 'complete_full_corpus200_profile', 'all13_profile_master_changes'}
ANALYZERS = {'raw': ('native-full-body-audit-20261112-v2/independent_native_full_body_audit_v2_20261112.py', 'a245225bd0d3da3bf6541a07713e700fe7c470ed0b26f1545fd3f1b457f9d69b'), 'runtime': ('native-runtime-audit-20261112-v1/independent_native_runtime_audit_20261112.py', '67465215bc09b8c23e04c967af9f6caa63f17b8ee5d445489657bc397d52c410'), 'stock': ('native-export-audit-20261112-v1/independent_native_export_audit_20261112.py', 'd41d6b551db0297a104cc257fcb1894b3c0bfe5df8b4d909716cf3a83dcfd2ca')}


def binding(plan):
    policy = plan['config']['full_body_v3']
    return {'source_sha256': corrected.sources(), 'full_data_sha256': policy['full_data_sha256'], 'vocabulary_cache_sha256': policy['vocabulary_cache']['sha256'], 'target_identity_sha256': plan['target']['model_identity_sha256'], 'native_ds_config_sha256': native.UPSTREAM['ds_config.json'], 'runtime_python': plan['config']['runtime_python'], 'required_packages': plan['config']['required_packages'], 'world_size': 8, 'checkpoint_mode': 'non_reentrant', 'rotary_capacity': 2054}


def inventory(reports, extra):
    result = {}
    for report in reports:
        for name, item in report['artifact_inventory'].items():
            value = {'sha256': item['sha256'], 'size_bytes': item['size_bytes']}
            require(name not in result or result[name] == value, 'Native evidence contains conflicting artifact identities')
            result[name] = value
    for name in extra:
        path = str(Path(name).resolve())
        result[path] = {'sha256': native.digest(path), 'size_bytes': Path(path).stat().st_size}
    require(bool(result), 'Native evidence inventory is empty')
    for path, item in result.items():
        require(Path(path).stat().st_size == item['size_bytes'] and native.digest(path) == item['sha256'], 'Native retained evidence changed: ' + path)
    return result


def prepare(preparation, raw_path, runtime_path, stock_path, teacher_path, teacher_review_path, output, bridge_path):
    native.require_slurm()
    output = Path(output)
    require(not output.exists(), 'Choose a fresh native qualification/profile report directory')
    queue = read(Path(preparation) / 'queue-plan.json')
    require(queue['source_sha256'] == corrected.sources(), 'Native admitted core differs from qualified source')
    plan_path = Path(queue['plans']['profile']['path'])
    require(native.digest(plan_path) == queue['plans']['profile']['sha256'], 'Native profile plan changed')
    plan = read(plan_path)
    require(hashlib.sha256(native.canonical({key: value for key, value in plan.items() if key != 'plan_identity_sha256'})).hexdigest() == plan['plan_identity_sha256'], 'Native profile plan identity differs')
    raw, runtime, stock, teacher, teacher_review = map(read, (raw_path, runtime_path, stock_path, teacher_path, teacher_review_path))
    for report in (raw, runtime, stock, teacher):
        require(pool.identity(report['artifact_inventory']) == report['artifact_inventory_sha256'], 'Native independent report inventory identity differs')
    for item in queue['plans'].values():
        path = str(Path(item['path']).resolve())
        require(all(report['artifact_inventory'].get(path, {}).get('sha256') == item['sha256'] for report in (raw, runtime)), 'Native raw/runtime evidence belongs to another registered plan')
    initial_path = Path(queue['plans']['uninterrupted']['path']).resolve()
    require(stock['plan_identity_sha256'] == read(initial_path)['plan_identity_sha256'] and stock['artifact_inventory'].get(str(initial_path), {}).get('sha256') == queue['plans']['uninterrupted']['sha256'], 'Native stock evidence belongs to another registered plan')
    analyzer_paths = []
    for label, report in (('raw', raw), ('runtime', runtime), ('stock', stock)):
        relative, expected = ANALYZERS[label]
        path = Path(plan['project_root']) / 'outputs/preparation' / relative
        require(native.digest(path) == expected == report.get('audit_source_sha256' if label == 'runtime' else 'analyzer_sha256'), 'Native independent analyzer source differs')
        analyzer_paths.append(path)
    require(raw['status'] == 'partial_evidence' and all(raw['checks'].get(key) == 'passed' for key in RAW_CHECKS) and raw['scientific_training_admitted'] is False, 'Complete independent native raw evidence has not passed')
    require(runtime['status'] == 'passed_source_bound_native_runtime_and_profile_interval' and len(runtime['actual_gpu_cases']) == 32 and runtime['profile_started_after_pre_source_order_observation'] is True, 'Independent native runtime interval has not passed')
    require(stock['status'] == 'passed_independent_corrected_native_export_and_stock_compatibility' and stock['source_sha256'] == corrected.sources() and len(stock['ranks']) == 8, 'Independent native stock export replay has not passed')
    require(teacher['status'] == 'passed_independent_teacher_worker_canonical_native_replay' and teacher['all_bound_input_bytes_stable'] is True and teacher_review['decision'] == 'approved' and teacher_review['report']['sha256'] == native.digest(teacher_path) and teacher_review['artifact_inventory_sha256'] == teacher['artifact_inventory_sha256'], 'Native teacher data lacks exact independent provenance approval')
    for label in ('train', 'validation'):
        require(plan[label]['sha256'] == plan['config']['full_body_v3']['full_data_sha256'][label] and any(item['sha256'] == plan[label]['sha256'] for item in teacher['artifact_inventory'].values()), 'Native profile/full data differ from independent teacher replay')
    statuses = [read(plan_path.parent / ('rank-' + str(rank) + '.json')) for rank in range(8)]
    require(all(value['rank'] == rank and value['status'] == 'bounded_complete' and value['successful_optimizer_steps'] == 200 and value['plan_identity_sha256'] == plan['plan_identity_sha256'] for rank, value in enumerate(statuses)), 'Native actual profile rank coverage differs')
    require(read(plan_path.parent / 'launch.json')['status'] == 'complete', 'Native profile launch has not completed')
    bound = binding(plan)
    interpreter = str((Path(plan['project_root']) / plan['config']['runtime_python']).resolve())
    require(interpreter in runtime['artifact_inventory'], 'Native runtime has no exact observed interpreter inventory')
    bridge = {'sha256': native.digest(bridge_path), 'runtime_interpreter': {'path': interpreter, **runtime['artifact_inventory'][interpreter]}}
    paths = (raw_path, runtime_path, stock_path, teacher_path, teacher_review_path, plan_path, Path(preparation) / 'queue-plan.json', __file__, bridge_path, *analyzer_paths)
    files = inventory((raw, runtime, stock, teacher), paths)
    common = {'status': 'passed', 'source_sha256': corrected.sources(), 'binding': bound, 'admission_bridge': bridge, 'qualified_numerical_core_unchanged': True, 'bridge_scope': 'Separately reviewed admission-only gate adapter; original reports and complete inventories retained including the exact external interpreter; no tensor/model/optimizer/loop/recovery changes', 'ranks': list(range(8)), 'evidence_files': files, 'evidence_inventory_identity_sha256': pool.identity(files), 'compiler': descriptor(__file__), 'scientific_training_admitted': False, 'independent_actual_evidence_approval_still_required': True, 'independent_reports': {name: descriptor(path) for name, path in (('raw', raw_path), ('runtime', runtime_path), ('stock', stock_path), ('teacher', teacher_path), ('teacher_review', teacher_review_path))}}
    qualification = {**common, 'scope': 'corrected full-mask native scratch dynamic-scaler GAS2 engineering qualification', 'checks': {key: 'passed' for key in ('full_native_source_bound_numerical_parity', 'eight_rank_four_vs_two_plus_two', 'stock_corrected_native_export', 'all13_named_adam_coverage', 'maximum_shape_2048_2054')}, 'independent_raw_checks': raw['checks']}
    profile = {**common, 'scope': 'corrected full-corpus200-update native engineering profile; no trained-quality result', 'checks': {key: 'passed' for key in ('complete_full_data200_successful_updates', 'all13_named_master_changes', 'runtime_data_vocabulary_and_hardware_inventory')}, 'successful_optimizer_steps_by_rank': [200] * 8, 'skipped_steps_by_rank': [value['skipped_steps'] for value in statuses], 'full_data_sha256': bound['full_data_sha256'], 'plan': descriptor(plan_path)}
    output.mkdir(parents=True)
    native.atomic_json(output / 'qualification.json', qualification)
    native.atomic_json(output / 'profile.json', profile)
    return {'status': 'prepared_actual_native_reports_requires_independent_review', 'qualification': descriptor(output / 'qualification.json'), 'profile': descriptor(output / 'profile.json'), 'binding': bound}


def integrate(output, review_path):
    native.require_slurm()
    output = Path(output)
    require(not (output / 'admission.json').exists(), 'Native scientific admission already exists')
    qualification, profile, review = read(output / 'qualification.json'), read(output / 'profile.json'), read(review_path)
    bound = qualification['binding']
    required = {'qualification': descriptor(output / 'qualification.json'), 'profile': descriptor(output / 'profile.json')}
    require(profile['status'] == qualification['status'] == 'passed' and profile['binding'] == bound and review.get('scientific_training_admitted') is True and review.get('binding') == bound and all(review.get(key) == value for key, value in required.items()), 'Native actual-evidence independent approval differs')
    require(review.get('source_sha256') == corrected.sources() and review.get('full_data_sha256') == bound['full_data_sha256'] and review.get('vocabulary_cache_sha256') == bound['vocabulary_cache_sha256'], 'Native review source/data/vocabulary binding differs')
    require(profile['admission_bridge'] == qualification['admission_bridge'] == review.get('admission_bridge'), 'Native exact-interpreter bridge lacks matching independent approval')
    for report in (qualification, profile):
        for name, item in report['evidence_files'].items():
            require(Path(name).stat().st_size == item['size_bytes'] and native.digest(name) == item['sha256'], 'Native admitted evidence changed')
    receipt = {'scientific_training_admitted': True, 'source_sha256': corrected.sources(), 'full_data_sha256': bound['full_data_sha256'], 'vocabulary_cache_sha256': bound['vocabulary_cache_sha256'], 'binding': bound, 'admission_bridge': qualification['admission_bridge'], **required, 'independent_review': descriptor(review_path)}
    native.atomic_json(output / 'admission.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'integrate'))
    parser.add_argument('--output', type=Path, required=True)
    for key in ('preparation', 'raw', 'runtime', 'stock', 'teacher', 'teacher-review', 'review', 'bridge'):
        parser.add_argument('--' + key, type=Path)
    args = parser.parse_args()
    result = prepare(args.preparation, args.raw, args.runtime, args.stock, args.teacher, args.teacher_review, args.output, args.bridge) if args.action == 'prepare' else integrate(args.output, args.review)
    print(json.dumps(result if args.action == 'prepare' else {'scientific_training_admitted': result['scientific_training_admitted']}))


if __name__ == '__main__':
    main()
