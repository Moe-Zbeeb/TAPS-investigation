import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from taps_rebuild import corpus_continue


class ContinuationTests(unittest.TestCase):
    def test_allocation_timeout_and_scheduler_preemption_can_resume(self):
        for holder, coordinator, restarts in [('TIMEOUT', 'FAILED', 0), ('RUNNING', 'PREEMPTED', 0), ('RUNNING', 'FAILED', 1)]:
            self.assertTrue(corpus_continue.allow_resume(holder, coordinator, restarts))

    def test_computation_failure_and_explicit_cancel_stay_blocked(self):
        for holder, coordinator in [('RUNNING', 'FAILED'), ('CANCELLED', 'FAILED'), ('RUNNING', 'CANCELLED'), ('TIMEOUT', 'CANCELLED')]:
            self.assertFalse(corpus_continue.allow_resume(holder, coordinator, 0))
        self.assertFalse(corpus_continue.allow_resume('CANCELLED', 'FAILED', 2))

    def test_requeued_completion_coordinator_resumes_the_existing_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plan = {'directory': str(root), 'identity_sha256': 'plan'}
            (root / 'attempt-20.json').write_text(json.dumps({'plan_identity_sha256': 'plan', 'hardware': {'job_id': '20'}}))
            jobs = {'10': {'JobState': 'FAILED', 'Restarts': '0'}, '20': {'JobState': 'RUNNING', 'Restarts': '0'}, '99': {'JobState': 'RUNNING', 'Restarts': '1'}}
            with patch.dict(os.environ, {'SLURM_JOB_ID': '99', 'CUDA_VISIBLE_DEVICES': ''}), patch.object(corpus_continue.corpus_generate, 'load_plan', return_value=plan), patch.object(corpus_continue, 'completed', side_effect=[None, {'status': 'complete'}]), patch.object(corpus_continue, 'job_record', side_effect=lambda job: jobs[job]), patch.object(corpus_continue.data_queue, 'holder', return_value='20'), patch.object(corpus_continue.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
                result = corpus_continue.run(root, root / 'plan.json', '10')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['attempts']), 1)
            run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
