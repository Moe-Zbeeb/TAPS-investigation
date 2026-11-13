import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'experiments/verification/side_pilot_dispatch_20261113.py'
SPEC = importlib.util.spec_from_file_location('side_pilot_dispatch', SOURCE)
DISPATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISPATCH)


class SidePilotDispatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.study = self.root / 'outputs/pilot-training' / DISPATCH.STUDY_ID
        self.study.mkdir(parents=True)
        self.producer = self.root / 'producer'
        self.producer.mkdir()
        (self.producer / 'worker.py').write_text('frozen')
        self.write(self.producer / 'bundle.json', {'files': {'worker.py': DISPATCH.digest(self.producer / 'worker.py')}})
        self.write(self.study / 'manifest.json', {'common': {'source_directory': str(self.producer / 'src/taps_rebuild')}})
        self.report = self.study / 'qualification-report.json'
        self.write(self.report, {'status': 'passed'})
        self.review = self.study / 'review.json'
        self.write(self.review, {'scientific_training_admitted': True})
        self.review_sha = DISPATCH.digest(self.review)
        self.write(self.study / 'admission.json', {'scientific_training_admitted': True, 'binding': {'study_manifest_sha256': DISPATCH.digest(self.study / 'manifest.json')}, 'qualification_report': {'path': str(self.report), 'sha256': DISPATCH.digest(self.report)}, 'independent_review': {'path': str(self.review), 'sha256': self.review_sha}})
        for name, value in [('REPORT_SHA256', DISPATCH.digest(self.report)), ('MANIFEST_SHA256', DISPATCH.digest(self.study / 'manifest.json'))]:
            value_patch = patch.object(DISPATCH, name, value)
            value_patch.start()
            self.addCleanup(value_patch.stop)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def check(self):
        return DISPATCH.check_study(self.root, self.study, self.producer, self.review_sha)

    def test_exact_integrated_review_and_frozen_producer_are_required(self):
        self.check()
        self.review.write_text(self.review.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'review or report changed'):
            self.check()

    def test_changed_producer_is_rejected(self):
        (self.producer / 'worker.py').write_text('different')
        with self.assertRaisesRegex(ValueError, 'producer bundle changed'):
            self.check()

    def test_waiting_fallback_can_yield_to_new_high_priority_holder(self):
        self.assertEqual(DISPATCH.next_action('200', '100', {'JobId': '300', 'JobState': 'PENDING'}, {}), 'high')
        self.assertEqual(DISPATCH.next_action('100', '100', {'JobId': '300', 'JobState': 'PENDING'}, {}), 'wait')

    def test_active_normal_job_prevents_a_second_training_launch(self):
        for state in ('RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'):
            self.assertEqual(DISPATCH.next_action('200', None, {'JobId': '300', 'JobState': state}, {}), 'wait')

    def test_failed_normal_job_cannot_be_retried_on_a_high_priority_holder(self):
        for state in ('FAILED', 'PREEMPTED', 'TIMEOUT', 'CANCELLED', 'COMPLETED'):
            with self.assertRaisesRegex(ValueError, 'automatic scientific retry refused'):
                DISPATCH.next_action('200', None, {'JobId': '300', 'JobState': state, 'ExitCode': '0:0'}, {})

    def test_only_matching_clean_boundary_can_continue(self):
        record = {'JobId': '300', 'JobState': 'COMPLETED', 'ExitCode': '0:0'}
        progress = {'status': 'waiting_successor_allocation', 'holder_job_id': '300'}
        self.assertEqual(DISPATCH.next_action(None, None, record, progress), 'clean_boundary')
        progress['holder_job_id'] = '299'
        with self.assertRaises(ValueError):
            DISPATCH.next_action(None, None, record, progress)

    def test_no_holder_submits_normal_priority(self):
        self.assertEqual(DISPATCH.next_action(None, None, None, {}), 'submit')

    def test_submission_requests_exact_reviewed_full_node_and_normal_class(self):
        record = {'JobId': '300', 'JobState': 'PENDING', 'ReqNodeList': 'deep-chungus-10', 'TRES': 'cpu=48,gres/gpu=8'}
        source = self.root / 'dispatch.py'
        with patch.object(DISPATCH, '__file__', str(source)), patch.object(DISPATCH.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '300\n', '')) as submit, patch.object(DISPATCH, 'normal_job_record', return_value=record):
            self.assertEqual(DISPATCH.submit_normal(self.root, self.study, self.producer, self.review_sha), '300')
        command = submit.call_args.args[0]
        for argument in ('--account=grad-students', '--partition=low-priority', '--qos=normal', '--exclusive', '--nodes=1', '--nodelist=deep-chungus-10', '--gres=gpu:a100:8', '--no-requeue'):
            self.assertIn(argument, command)
        self.assertNotIn('CUDA_VISIBLE_DEVICES', submit.call_args.kwargs['env'])

    def test_job_from_other_class_cannot_be_cancelled_or_adopted(self):
        record = 'JobId=300 UserId=other(1) Account=grad-students Partition=low-priority QOS=normal JobName=taps-side-pilots JobState=PENDING'
        with patch.object(DISPATCH.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, record, '')):
            with self.assertRaisesRegex(ValueError, 'ownership or class differs'):
                DISPATCH.normal_job_record('300')

    def test_only_own_pending_fallback_is_cancelled(self):
        with patch.object(DISPATCH, 'normal_job_record', side_effect=[{'JobState': 'PENDING'}, {'JobState': 'CANCELLED'}]), patch.object(DISPATCH.subprocess, 'run') as cancel:
            self.assertTrue(DISPATCH.cancel_pending_fallback('300'))
        cancel.assert_called_once_with(['scancel', '--state=PENDING', '300'], check=True)

    def test_fallback_start_race_cannot_trigger_a_second_writer(self):
        with patch.object(DISPATCH, 'normal_job_record', return_value={'JobState': 'RUNNING'}), patch.object(DISPATCH.subprocess, 'run') as cancel:
            self.assertFalse(DISPATCH.cancel_pending_fallback('300'))
            cancel.assert_not_called()
        with patch.object(DISPATCH, 'normal_job_record', side_effect=[{'JobState': 'PENDING'}, {'JobState': 'RUNNING'}]), patch.object(DISPATCH.subprocess, 'run'):
            self.assertFalse(DISPATCH.cancel_pending_fallback('300'))


if __name__ == '__main__':
    unittest.main()
