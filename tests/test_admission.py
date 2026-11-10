from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from taps_rebuild import admission, native_training, pool, training_queue


class AdmissionTests(unittest.TestCase):
    def test_metadata_alone_cannot_admit_training(self):
        with self.assertRaisesRegex(admission.AdmissionError, 'claimed passed metadata'):
            admission.verify_certificate('.', {'status': 'passed', 'automatic_pipeline_gate_pass': True})

    def test_verified_artifacts_are_invalidated_by_later_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = []
            for index in range(150):
                path = root / str(index)
                path.write_text(str(index))
                artifacts.append(admission.descriptor(root, path))
            report = {'format': 'source-bound-full-model-audit-v1', 'status': 'passed', 'automatic_pipeline_gate_pass': True, 'validator_sha256': admission.digest(admission.__file__), 'driver_sha256': admission.DRIVER_SHA256, 'prepared_sha256': admission.PREPARED_SHA256, 'review': {'decision': 'approved'}, 'checks': {key: True for key in ['greedy', 'conditional', 'sampling', 'complete_registered_phases', 'independent_validator_review']}, 'artifacts': artifacts, 'scope': 'test verifier contract only'}
            report['identity_sha256'] = admission.identity(report)
            self.assertTrue(admission.verify_certificate(root, report)['scientific_training_authorized'])
            (root / '10').write_text('changed')
            with self.assertRaisesRegex(admission.AdmissionError, 'changed after CPU'):
                admission.verify_certificate(root, report)

    def test_pending_review_is_an_explicit_block(self):
        self.assertEqual(admission.review_document(None, 'proof', 'protocol')['decision'], 'pending')

    def test_shared_filesystem_fingerprint_ignores_node_local_device_numbers(self):
        values = {'st_ino': 7, 'st_size': 42, 'st_mtime_ns': 1000}
        with patch.object(Path, 'stat', return_value=SimpleNamespace(st_dev=10, **values)):
            first = admission.stamp('artifact')
        with patch.object(Path, 'stat', return_value=SimpleNamespace(st_dev=20, **values)):
            self.assertEqual(first, admission.stamp('artifact'))

    def test_failed_validation_never_prepares_a_native_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {'run_id': 'test', 'driver': 'driver', 'prepared': 'prepared', 'primary': 'primary', 'screen': 'screen', 'proof': 'proof', 'review': 'review', 'native_template': 'native.json', 'profile_optimizer_steps': 200, 'reproduction_config_sha256': 'a' * 64, 'reproduction_report': {'path': 'failed', 'sha256': 'b' * 64}}
            pool.atomic_json(root / 'queue.json', config)
            pool.atomic_json(root / 'native.json', {})
            with patch.object(admission, 'audit', return_value={'status': 'blocked', 'checks': {'greedy': False}}), patch.object(native_training, 'research_gate', side_effect=native_training.NativeTrainingError('Greedy mismatch')), patch.object(native_training, 'prepare') as prepare:
                with self.assertRaisesRegex(native_training.NativeTrainingError, 'Greedy mismatch'):
                    training_queue.audit(root, root / 'queue.json')
                prepare.assert_not_called()
            status = admission.read(root / 'outputs/training-queue/test/status.json')
            self.assertEqual(status['status'], 'blocked')
            self.assertFalse(status['training_started'])


if __name__ == '__main__':
    unittest.main()
