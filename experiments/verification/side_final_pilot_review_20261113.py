import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


EXPECTED = {
    'report': '8bb13bef08ef73121affe85bfcc51f8d56fadeb486b117339f3a1396c3b70783',
    'audit': '7ee35e765c51bdd04ad97edf32e3cc00daadbbeffc83a2466253ae69bfda028f',
    'manifest': 'b26599d4644a33c054f6a24853eedae9241ee0f3d6473cbd5fb6a3ef3fbcbdd9',
    'analyzer': 'a65a4848ef351860c87a1ed1d4f911bc7c724e58e33411b709fb6e73398ff3b6',
    'compiler': '44b8030bba92e71cf30c038481d433ab41b5f38f957807c09ba5d067bf656e12',
    'worker': 'f700df226d9bd0ea3391868c24e6780efcfde0be3967bda2dba354e7cc699a4a',
}
CHECKS = {'schedules', 'ce_gradients', 'outside_mask_gradients', 'native_step_parity', 'restart_parity', 'suffix_probe', 'raw_zero_state_reconstruction', 'all13_named_hooks_and_corrected_policy', 'rotary_priming', 'all13_named_adam_and_fp32_updates', 'stock_export_compatibility', 'runtime_envelope', 'teacher_buffer_policy', 'teacher_gate_prefix208'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def verify_file(item):
    name, expected = item
    path = Path(name)
    before = path.stat()
    require(before.st_size == expected['size_bytes'] and digest(path) == expected['sha256'], 'Retained evidence changed: ' + name)
    after = path.stat()
    require((before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns), 'Evidence changed during final review: ' + name)
    return before.st_size


def review(root, output):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Final evidence rehash requires CPU Slurm')
    study = root / 'outputs/pilot-training/rq1-pilot-24-20261113-v3'
    report_path = study / 'qualification-report.json'
    audit_path = root / 'outputs/pilot-independent-audit/rq1-pilot-24-20261113-v3/full-v9.json'
    manifest_path = study / 'manifest.json'
    require(all(digest(path) == EXPECTED[key] for key, path in [('report', report_path), ('audit', audit_path), ('manifest', manifest_path)]), 'Reviewed report, audit or manifest bytes differ')
    report, audit, manifest = map(read, (report_path, audit_path, manifest_path))
    require(report['status'] == 'passed' and report['scientific_training_admitted'] is False and report['independent_review_still_required'] is True and report['admission_schema'] == 'pristine-teacher-pilot-v1', 'Compiled qualification scope differs')
    require(audit['status'] == 'all_checked_evidence_passed_requires_independent_review' and audit['scientific_training_admitted'] is False and audit['automatic_admission_emitted'] is False and audit['project_helpers_imported'] is False, 'Actual independent audit did not pass')
    require(set(audit['checks']) == CHECKS and report['independent_check_results'] == audit['checks'] and all(value == 'passed' for value in audit['checks'].values()), 'All fourteen independent checks are required')
    require(audit['analyzer_v9_sha256'] == report['independent_analyzer_sha256'] == EXPECTED['analyzer'] and report['compiler_sha256'] == EXPECTED['compiler'], 'Actual analyzer or compiler differs')
    require(report['binding']['study_manifest_sha256'] == EXPECTED['manifest'] and report['binding']['source_sha256'] == manifest['common']['source_sha256'] and report['binding']['source_sha256']['pilot_train.py'] == EXPECTED['worker'], 'Reviewed study source binding differs')
    require(identity(report['evidence_files']) == report['evidence_inventory_identity_sha256'] and identity(audit['artifact_inventory']) == audit['artifact_inventory_sha256'], 'Evidence inventory identities differ')
    require(all(report['evidence_files'].get(name) == value for name, value in audit['artifact_inventory'].items()), 'Compiled report omitted or changed independent evidence')
    require(len(manifest['plans']) == 24, 'Expected twenty-four registered plans')
    conditions = set()
    for descriptor in manifest['plans']:
        require(digest(descriptor['path']) == descriptor['sha256'], 'Registered plan changed')
        plan = read(descriptor['path'])
        require(plan['study_id'] == 'rq1-pilot-24-20261113-v3' and plan['source_sha256'] == report['binding']['source_sha256'] and plan['steps'] == 1536 and plan['quota'] == 64 and plan['initialization'] == 'released_common_ancestor', 'Pilot recipe, budget or initialization differs')
        conditions.add((plan['cohort'], plan['policy'], plan['seed']))
    require(conditions == {(cohort, policy, seed) for cohort in ['chat', 'math', 'code', 'generalist'] for policy in ['original', 'target'] for seed in [0, 1, 2]}, 'Pilot grid differs')
    proof = audit['teacher_source_and_actual_helper_proof']
    require(len(proof['actual_helper_cases']) == 40 and proof['unchanged_schedule_files'] == 96 and len(audit['teacher_buffer_initializations']['attempt_ranks']) == 80, 'Actual teacher proof coverage differs')
    prefix = audit['teacher_gate_prefix208']
    require([value['rank'] for value in prefix['ranks']] == list(range(8)) and all(value['committed_updates'] == 208 and value['all13_finite_nonzero_callbacks_every_update'] is True for value in prefix['ranks']), 'Failure-case engineering replay coverage differs')
    require(audit['raw_states']['status'] == audit['gradients']['status'] == audit['stock']['status'] == audit['runtime']['status'] == 'passed', 'Numerical, raw state, export or runtime evidence is incomplete')
    files = report['evidence_files']
    with ThreadPoolExecutor(max_workers=min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))) as workers:
        total = sum(workers.map(verify_file, sorted(files.items())))
    require(digest(report_path) == EXPECTED['report'] and digest(audit_path) == EXPECTED['audit'], 'Final reviewed reports changed during rehash')
    value = {'decision': 'approved_for_exact_fresh_24_pilot_training', 'scientific_training_admitted': True, 'scope': 'Fresh rq1-pilot-24-20261113-v3 training only; no failed-run resume, native generalist changes, inference quality, speed or full PDF completion admission', 'reviewer': 'independent side-conversation evidence reviewer', 'reviewed_at_utc': datetime.now(timezone.utc).isoformat(), 'qualification_report_sha256': EXPECTED['report'], 'binding': report['binding'], 'independent_audit': {'path': str(audit_path), 'sha256': EXPECTED['audit']}, 'checks_reviewed': audit['checks'], 'evidence_inventory_identity_sha256': report['evidence_inventory_identity_sha256'], 'evidence_artifact_count': len(files), 'evidence_bytes_rehashed': total, 'all_retained_evidence_rehashed': True, 'review_verifier_sha256': digest(__file__), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'cpu_only': True, 'failed_scientific_run_resume_authorized': False, 'conditions': ['Explicit integration must rehash the exact report inventory and enforce its independent review.', 'Run only the unchanged source-frozen producer with the exact registered data, masks, vocabulary, budgets and physical eight-GPU hardware.', 'Normal priority changes scheduling only; no numerical or scientific gate is waived.', 'Record all physical work and stop on any numerical, source, data or recovery failure.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': value['decision'], 'review_sha256': digest(output), 'artifacts': len(files), 'bytes_rehashed': total}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(args.root.resolve(), args.output.resolve())
