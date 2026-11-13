import copy
import unittest

from taps_rebuild import pilot_teacher_admission as admission, pool


class PilotTeacherAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'study_id': admission.STUDY_ID, 'recipe_sha256': admission.RECIPE_SHA256, 'source_sha256': {'pilot_train.py': admission.WORKER_SHA256}}
        inventory = {'/retained/independent_pilot_qualification_audit_v9_20261113.py': {'sha256': admission.ANALYZER_SHA256, 'size_bytes': 123}}
        self.audit = {'status': 'all_checked_evidence_passed_requires_independent_review', 'checks': {name: 'passed' for name in admission.CHECKS}, 'analyzer_v9_sha256': admission.ANALYZER_SHA256, 'scientific_training_admitted': False, 'automatic_admission_emitted': False, 'project_helpers_imported': False, 'teacher_source_and_actual_helper_proof': {'status': 'passed', 'study_id': admission.STUDY_ID, 'source_sha256': copy.deepcopy(self.plan['source_sha256']), 'recipe_sha256': admission.RECIPE_SHA256, 'actual_helper_cases': list(range(40)), 'unchanged_schedule_files': 96}, 'teacher_gate_prefix208': {'status': 'passed', 'ranks': [{'rank': rank, 'committed_updates': 208} for rank in range(8)]}, 'teacher_buffer_initializations': {'status': 'passed', 'attempt_ranks': list(range(80))}, 'artifact_inventory': inventory, 'artifact_inventory_sha256': pool.identity(inventory)}

    def test_exact_fourteen_checks_and_new_evidence_are_accepted_for_compilation(self):
        admission.validate_audit(self.plan, self.audit)

    def test_old_twelve_check_report_is_rejected(self):
        self.audit['checks'].pop('teacher_buffer_policy')
        self.audit['checks'].pop('teacher_gate_prefix208')
        with self.assertRaisesRegex(ValueError, 'fourteen-check'):
            admission.validate_audit(self.plan, self.audit)

    def test_failed_new_teacher_check_is_not_dropped(self):
        self.audit['checks']['teacher_gate_prefix208'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'fourteen-check'):
            admission.validate_audit(self.plan, self.audit)

    def test_old_producer_source_cannot_reuse_new_evidence(self):
        self.plan['source_sha256']['pilot_train.py'] = 'old-source'
        with self.assertRaisesRegex(ValueError, 'fresh study'):
            admission.validate_audit(self.plan, self.audit)

    def test_incomplete_failure_case_prefix_is_rejected(self):
        self.audit['teacher_gate_prefix208']['ranks'][2]['committed_updates'] = 207
        with self.assertRaisesRegex(ValueError, '208-update'):
            admission.validate_audit(self.plan, self.audit)

    def test_stale_inventory_identity_is_rejected(self):
        self.audit['artifact_inventory_sha256'] = 'old-inventory'
        with self.assertRaisesRegex(ValueError, 'inventory'):
            admission.validate_audit(self.plan, self.audit)

    def test_audit_cannot_admit_itself(self):
        self.audit['scientific_training_admitted'] = True
        with self.assertRaisesRegex(ValueError, 'fourteen-check'):
            admission.validate_audit(self.plan, self.audit)


if __name__ == '__main__':
    unittest.main()
