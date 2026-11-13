from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'experiments/verification/resume_corrected_generalist_20261113.py'
SPEC = importlib.util.spec_from_file_location('corrected_native_continuation', SOURCE)
CONTINUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTINUATION)


class NativeContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.directory = self.root / 'outputs/native-training/sharegpt-ultrachat-corrected-full-20261113-v4'
        self.directory.mkdir(parents=True)
        self.checkpoint = self.root / 'models/fresh/recovery/attempt-1-step-1000'
        self.checkpoint.mkdir(parents=True)
        self.bridge = self.directory / 'inputs/native_full_body_bridge.py'
        self.bridge.parent.mkdir()
        self.bridge.write_text('pass\n')
        self.plan = {'config': {'pipeline_stage': 'corrected-full-v4', 'runtime_python': 'envs/native-eagle3/bin/python', 'full_body_v3': {'source_sha256': {'native_core': 'unchanged'}}}, 'run_dir': str(self.directory), 'project_root': str(self.root), 'plan_identity_sha256': 'new-plan', 'trained_models': {'resolved_path': str(self.root / 'models/fresh')}, 'frozen_files': {'inputs/native_full_body_bridge.py': self.digest(self.bridge)}}
        self.plan_path = self.directory / 'plan.json'
        self.write(self.plan_path, self.plan)
        self.expected_sha = self.digest(self.plan_path)
        self.launch = {'plan_identity_sha256': 'new-plan', 'slurm_job_id': '2216035', 'status': 'running', 'attempt': 1, 'hardware': {'hardware_identity_sha256': 'original-eight-gpus'}}
        self.write(self.directory / 'launch.json', self.launch)
        for rank in range(8):
            self.write(self.directory / ('rank-' + str(rank) + '.json'), {'rank': rank, 'world_size': 8, 'plan_identity_sha256': 'new-plan', 'slurm_job_id': '2216035', 'status': 'running'})
        self.manifest = {'plan_identity_sha256': 'new-plan', 'world_size': 8, 'global_step': 1000, 'hardware_identity_sha256': 'original-eight-gpus'}
        self.write(self.checkpoint / 'manifest.json', self.manifest)
        self.latest = {'path': str(self.checkpoint), 'plan_identity_sha256': 'new-plan', 'manifest_sha256': self.digest(self.checkpoint / 'manifest.json')}
        self.write(self.checkpoint.parent / 'latest.json', self.latest)
        self.end = datetime.now(timezone.utc).replace(microsecond=0)
        self.usage = 'gres:gpu:a100:8\ngres:gpu:a100:8\n'
        self.accounting = '2216035|TIMEOUT|mohamadzbib|taps-deep11-hold|' + self.end.replace(tzinfo=None).isoformat() + '|\n'
        self.control = 'JobState=RUNNING UserId=mohamadzbib(29562) Account=grad-students Partition=high-priority QOS=spec-priority NodeList=deep-chungus-11 TRES=cpu=48,gres/gpu=8'

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def command_output(self, command, **options):
        return {'scontrol': self.control, 'squeue': self.usage, 'sacct': self.accounting}[command[0]]

    def execute(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '900001'}), patch.object(CONTINUATION.subprocess, 'check_output', side_effect=self.command_output), patch.object(CONTINUATION.subprocess, 'run') as resume:
            CONTINUATION.run(self.plan_path, self.expected_sha, '2216035')
            return resume

    def test_timeout_reconciles_only_bound_stale_launch_and_calls_frozen_resume(self):
        resume = self.execute()
        self.assertEqual(json.loads((self.directory / 'launch.json').read_text())['status'], 'failed')
        self.assertEqual(json.loads((self.directory / 'continuation/unclosed-launch-2216035.json').read_text()), self.launch)
        self.assertEqual(resume.call_args.args[0], [str(self.root / 'envs/native-eagle3/bin/python'), '-B', str(self.bridge), 'resume', '--plan', str(self.plan_path)])

    def test_numerical_failure_is_not_retried_even_at_allocation_end(self):
        self.launch.update(status='failed', finished_at=self.end.isoformat(), error_type='NativeTrainingError', error='Native training subprocess failed: 1')
        self.write(self.directory / 'launch.json', self.launch)
        with self.assertRaisesRegex(ValueError, 'noninterruption'):
            self.execute()
        self.assertEqual(json.loads((self.directory / 'launch.json').read_text()), self.launch)

    def test_verified_launcher_interruption_at_expiry_can_resume(self):
        self.launch.update(status='failed', finished_at=self.end.isoformat(), error_type='NativeTrainingError', error='Native launcher interrupted by signal 15')
        self.write(self.directory / 'launch.json', self.launch)
        self.assertTrue(self.execute().called)

    def test_changed_checkpoint_hardware_blocks_resume(self):
        self.manifest['hardware_identity_sha256'] = 'different-eight-gpus'
        self.write(self.checkpoint / 'manifest.json', self.manifest)
        self.latest['manifest_sha256'] = self.digest(self.checkpoint / 'manifest.json')
        self.write(self.checkpoint.parent / 'latest.json', self.latest)
        with self.assertRaisesRegex(ValueError, 'hardware'):
            self.execute()

    def test_high_priority_allowance_is_enforced_across_running_jobs(self):
        self.usage += 'gres:gpu:a100:8\n'
        with self.assertRaisesRegex(ValueError, 'allowance'):
            self.execute()

    def test_already_complete_run_does_not_resume(self):
        self.launch['status'] = 'complete'
        self.write(self.directory / 'launch.json', self.launch)
        self.assertFalse(self.execute().called)

    def test_changed_plan_bytes_block_before_scheduler_access(self):
        self.plan_path.write_text(self.plan_path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'plan bytes'):
            self.execute()

    def test_retired_run_namespace_is_rejected(self):
        self.plan['config']['pipeline_stage'] = 'retired-generalist'
        self.write(self.plan_path, self.plan)
        self.expected_sha = self.digest(self.plan_path)
        with self.assertRaisesRegex(ValueError, 'restricted'):
            self.execute()

    def test_stale_launcher_does_not_hide_worker_numerical_failure(self):
        path = self.directory / 'rank-3.json'
        value = json.loads(path.read_text())
        value.update(status='failed', error_type='FloatingPointError', error='nonfinite loss')
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, 'worker receipt'):
            self.execute()
        self.assertEqual(json.loads((self.directory / 'launch.json').read_text()), self.launch)
        self.assertEqual(json.loads(path.read_text()), value)
        self.assertFalse((self.directory / 'continuation').exists())

    def test_rank_from_another_allocation_blocks_resume(self):
        path = self.directory / 'rank-4.json'
        value = json.loads(path.read_text())
        value['slurm_job_id'] = 'old-invalid-generalist'
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, 'worker receipt'):
            self.execute()

    def test_full_body_failure_is_not_hidden_by_running_root_rank(self):
        path = self.directory / 'attempts/attempt-1/full-body-rank-2.json'
        path.parent.mkdir(parents=True)
        self.write(path, {'rank': 2, 'plan_identity_sha256': 'new-plan', 'status': 'failed', 'error': 'bad gradient'})
        with self.assertRaisesRegex(ValueError, 'full-body worker'):
            self.execute()
        self.assertEqual(json.loads((self.directory / 'launch.json').read_text()), self.launch)

    def test_missing_rank_receipt_blocks_resume(self):
        (self.directory / 'rank-7.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.execute()

    def test_authorized_normal_priority_same_node_continuation_is_accepted(self):
        self.control = self.control.replace('Partition=high-priority QOS=spec-priority', 'Partition=low-priority QOS=normal')
        self.usage = 'gres:gpu:a100:8\n'
        self.assertTrue(self.execute().called)

    def test_unregistered_scheduling_class_is_rejected(self):
        self.control = self.control.replace('QOS=spec-priority', 'QOS=unregistered')
        with self.assertRaisesRegex(ValueError, 'scheduling class'):
            self.execute()


if __name__ == '__main__':
    unittest.main()
