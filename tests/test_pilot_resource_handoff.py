import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from taps_rebuild import pilot_teacher_admission as compiler


SOURCE = Path(__file__).resolve().parents[1] / 'experiments/verification/release_pilot_when_allocated_20261113.py'
SPEC = importlib.util.spec_from_file_location('pilot_resource_handoff', SOURCE)
HANDOFF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HANDOFF)


class PilotResourceHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.study = self.root / 'outputs/pilot-training/rq1-pilot-24-20261113-v3'
        self.study.mkdir(parents=True)
        self.producer = self.root / 'producer'
        self.producer.mkdir()
        (self.producer / 'marker.txt').write_text('frozen source')
        self.write(self.producer / 'bundle.json', {'files': {'marker.txt': self.digest(self.producer / 'marker.txt')}})
        self.write(self.study / 'manifest.json', {'common': {'source_directory': str(self.producer / 'src/taps_rebuild')}})
        self.binding = {'study_manifest_sha256': self.digest(self.study / 'manifest.json')}
        self.write(self.study / 'qualification-report.json', {'binding': self.binding})
        self.review = self.study / 'independent-evidence-review.json'
        self.write(self.review, {'scientific_training_admitted': True, 'qualification_report_sha256': self.digest(self.study / 'qualification-report.json'), 'binding': self.binding})
        self.review_sha = self.digest(self.review)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def invoke(self):
        HANDOFF.run(self.root, self.study, self.root / 'compiler', self.producer, self.review, self.review_sha)

    def test_admission_waits_for_live_holder_before_invoking_frozen_pipeline(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '100', 'CUDA_VISIBLE_DEVICES': ''}), patch('taps_rebuild.pilot_queue.holder', side_effect=[ValueError('pending'), 'new-holder', 'new-holder']), patch.object(compiler, 'integrate') as integrate, patch.object(HANDOFF.subprocess, 'run') as launch:
            def pending_delay(seconds):
                integrate.assert_not_called()
                launch.assert_not_called()
            with patch.object(HANDOFF.time, 'sleep', side_effect=pending_delay):
                self.invoke()
            integrate.assert_called_once_with(self.study, self.study / 'qualification-report.json', self.review)
            self.assertEqual(launch.call_args.kwargs['env']['PYTHONPATH'], str(self.producer / 'src'))
            self.assertIn('taps_rebuild.pilot_pipeline', launch.call_args.args[0])

    def test_changed_actual_review_is_rejected_before_resource_wait(self):
        self.review.write_text(self.review.read_text() + '\n')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '100', 'CUDA_VISIBLE_DEVICES': ''}), patch.object(compiler, 'integrate') as integrate:
            with self.assertRaisesRegex(ValueError, 'independent actual review'):
                self.invoke()
            integrate.assert_not_called()

    def test_changed_report_during_wait_cannot_be_admitted(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '100', 'CUDA_VISIBLE_DEVICES': ''}), patch('taps_rebuild.pilot_queue.holder', side_effect=[ValueError('pending'), 'new-holder']), patch.object(compiler, 'integrate') as integrate:
            def change_report(seconds):
                path = self.study / 'qualification-report.json'
                path.write_text(path.read_text() + '\n')
            with patch.object(HANDOFF.time, 'sleep', side_effect=change_report):
                with self.assertRaisesRegex(ValueError, 'changed during resource wait'):
                    self.invoke()
            integrate.assert_not_called()

    def test_unavailable_holder_expires_without_writing_admission(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '100', 'CUDA_VISIBLE_DEVICES': ''}), patch('taps_rebuild.pilot_queue.holder', side_effect=ValueError('pending')), patch.object(HANDOFF.time, 'monotonic', side_effect=[0, 172801]), patch.object(compiler, 'integrate') as integrate:
            with self.assertRaisesRegex(ValueError, '48hours'):
                self.invoke()
            integrate.assert_not_called()
            self.assertFalse((self.study / 'admission.json').exists())

    def test_empty_producer_manifest_is_not_an_approved_source_bundle(self):
        self.write(self.producer / 'bundle.json', {'files': {}})
        with patch.dict(os.environ, {'SLURM_JOB_ID': '100', 'CUDA_VISIBLE_DEVICES': ''}), patch.object(compiler, 'integrate') as integrate:
            with self.assertRaisesRegex(ValueError, 'producer bundle'):
                self.invoke()
            integrate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
