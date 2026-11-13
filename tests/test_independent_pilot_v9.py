import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


DIRECTORY = Path(__file__).resolve().parents[1] / 'experiments/verification/reviews'


def load(directory):
    path = directory / 'independent_pilot_qualification_audit_v9_20261113.py'
    spec = importlib.util.spec_from_file_location('independent_pilot_v9_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndependentPilotV9Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        directory = self.root / 'outputs/analyzers'
        directory.mkdir(parents=True)
        for path in DIRECTORY.glob('independent_pilot_qualification_audit*_2026111*.py'):
            shutil.copyfile(path, directory / path.name)
        path = DIRECTORY / 'pilot_zero_lr_raw_diagnostic_20261112.py'
        shutil.copyfile(path, directory / path.name)
        self.module = load(directory)
        self.study = self.root / 'outputs/pilot-training' / self.module.STUDY_ID / 'manifest.json'
        self.study.parent.mkdir(parents=True)
        self.study.write_text('{"fixture":"new-study"}\n')
        self.report = self.root / 'outputs/pilot-independent-audit' / self.module.STUDY_ID / 'fresh-zero.json'
        self.report.parent.mkdir(parents=True)

    def fact(self, policy):
        names = self.module.v3.NAMES
        return {
            'policy': policy,
            'registered_plan_identity_sha256': 'new-' + policy,
            'all13_named_adam_first_second_nonzero': True,
            'all13_numeric_masters_unchanged': True,
            'all_master_bit_changes_are_signed_zero': True,
            'all_visible_bit_changes_are_signed_zero': True,
            'all_master_casts_bitwise_match_visible': True,
            'visible_trainable_differences': {name: {'numeric_changed': 0, 'all_finite': True, 'bitwise_changed': 0, 'signed_zero_only_bit_changes': 0} for name in names},
            'fp32_master_and_adam_differences': {name: {'numeric_changed': 0, 'all_finite': True, 'bitwise_changed': 0, 'signed_zero_only_bit_changes': 0, 'first_moment_nonzero': 1, 'second_moment_nonzero': 1, 'fp32_master_numeric_delta_l2': 0.0, 'master_cast_visible_bitwise_changes': 0} for name in names},
            'vocabulary': {name: {'dtype_equal': True, 'shape_equal': True, 'tensor_equal': True, 'bitwise_identity_equal': True} for name in ('d2t', 't2d')},
        }

    def proof(self):
        inventory = {str(self.study): {'sha256': self.module.digest(self.study), 'size_bytes': self.study.stat().st_size}}
        return {'status': 'raw_zero_lr_prefix_mechanism_diagnostic_complete_not_admission', 'diagnostic_source_sha256': self.module.v8.ZERO_DIAGNOSTIC_SOURCE_SHA256, 'scientific_training_admitted': False, 'artifact_inventory': inventory, 'artifact_inventory_sha256': self.module.v1.identity(inventory), 'prefixes': [self.fact('original'), self.fact('target')]}

    def audit(self, proof):
        self.report.write_text(json.dumps(proof))
        audit = self.module.Audit(self.root, {'path': str(self.report), 'sha256': self.module.digest(self.report)})
        audit.artifact = self.module.v1.Audit.artifact.__get__(audit)
        audit.study_path = self.study
        audit.study_sha256 = self.module.digest(self.study)
        audit.plans = {policy: {'seed': 0, 'cohort': 'generalist', 'policy': policy, 'identity_sha256': 'new-' + policy} for policy in ('original', 'target')}
        return audit

    def test_fresh_explicit_fact_descriptor_is_bound_to_both_new_plans(self):
        self.assertEqual(set(self.audit(self.proof()).zero_facts()), {'original', 'target'})

    def test_rehashed_old_plan_fact_cannot_be_relabeled_as_new(self):
        proof = self.proof()
        proof['prefixes'][0]['registered_plan_identity_sha256'] = 'old-original'
        with self.assertRaisesRegex(self.module.v1.AuditFailure, 'another_study_or_plan'):
            self.audit(proof).zero_facts()

    def test_rehashed_fact_without_new_manifest_is_rejected(self):
        proof = self.proof()
        proof['artifact_inventory'] = {}
        proof['artifact_inventory_sha256'] = self.module.v1.identity({})
        with self.assertRaisesRegex(self.module.v1.AuditFailure, 'not_bound_to_new_manifest'):
            self.audit(proof).zero_facts()

    def test_numeric_master_difference_is_never_a_signed_zero_exception(self):
        proof = self.proof()
        name = next(iter(self.module.v3.NAMES))
        proof['prefixes'][0]['fp32_master_and_adam_differences'][name]['numeric_changed'] = 1
        with self.assertRaisesRegex(self.module.v1.AuditFailure, 'numeric_or_nonfinite_change'):
            self.audit(proof).zero_facts()

    def test_missing_named_moment_is_rejected(self):
        proof = self.proof()
        name = next(iter(self.module.v3.NAMES))
        proof['prefixes'][0]['fp32_master_and_adam_differences'][name]['first_moment_nonzero'] = 0
        with self.assertRaisesRegex(self.module.v1.AuditFailure, 'moment_or_cast_failure'):
            self.audit(proof).zero_facts()

    def test_stale_explicit_report_hash_is_rejected(self):
        audit = self.audit(self.proof())
        self.report.write_text(self.report.read_text() + ' ')
        with self.assertRaisesRegex(self.module.v1.AuditFailure, 'digest_mismatch'):
            audit.zero_facts()

    def test_teacher_policy_requires_complete_names_original_identity_and_frozen_eval(self):
        expected = 'a' * 64
        valid = {'policy': 'exact_pre_deepspeed_fp32_teacher_buffers', 'buffer_names': self.module.BUFFER_NAMES, 'pristine_buffer_identity_sha256': expected, 'restored_buffer_identity_sha256': expected, 'teacher_parameters_remain_frozen': True, 'teacher_remains_in_evaluation_mode': True}
        self.module.validate_teacher_policy(valid, expected)
        for key, value in [('buffer_names', self.module.BUFFER_NAMES[:-1]), ('restored_buffer_identity_sha256', 'b' * 64), ('teacher_parameters_remain_frozen', False), ('teacher_remains_in_evaluation_mode', False)]:
            bad = copy.deepcopy(valid)
            bad[key] = value
            with self.assertRaises(self.module.v1.AuditFailure):
                self.module.validate_teacher_policy(bad, expected)


if __name__ == '__main__':
    unittest.main()
