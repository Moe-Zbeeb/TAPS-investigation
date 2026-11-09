from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from taps_rebuild import attach


class HolderAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / 'project'
        self.legacy = self.base / 'legacy'
        self.config = self.root / 'configs/reproduction.json'
        self.config.parent.mkdir(parents=True)
        self.config.write_text(json.dumps({'cluster_root': str(self.root), 'legacy_runtime_root': str(self.legacy), 'scope': 'reference', 'method_order': ['eagle3', 'eagle2', 'hass'], 'methods': {'eagle3': {}, 'eagle2': {}, 'hass': {}}}))
        pool = self.root / 'src/taps_rebuild/pool.py'
        pool.parent.mkdir(parents=True)
        pool.write_text('')
        self.current = self.legacy / 'cluster/deep10/current.json'
        self.current.parent.mkdir(parents=True)
        self.metadata = {'job_id': '2212576', 'successor': '2212580', 'node': attach.NODE, 'gpus': 8}
        self.current.write_text(json.dumps(self.metadata))
        self.holder = 'JobId=2212576 JobName=deep10-hold UserId=mohamadzbib(29562) JobState=RUNNING Account=grad-students Partition=high-priority QOS=spec-priority NodeList=deep-chungus-10 ReqNodeList=deep-chungus-10 NumNodes=1 NumCPUs=64 AllocTRES=cpu=64,gres/gpu=8,gres/gpu:a100=8 TimeLimit=2-00:00:00 RunTime=01:00:00 Dependency=(null)'
        self.successor = 'JobId=2212580 JobName=deep10-hold UserId=mohamadzbib(29562) JobState=PENDING Account=grad-students Partition=high-priority QOS=spec-priority NodeList=(null) ReqNodeList=deep-chungus-10 NumNodes=1 NumCPUs=64 ReqTRES=cpu=64,gres/gpu=8,gres/gpu:a100=8 TimeLimit=2-00:00:00 RunTime=00:00:00 Dependency=afterany:2212576(unfulfilled)'
        self.queries = []
        self.now = datetime(2026, 11, 9, 13, 40, 0, 123456, tzinfo=timezone.utc)
        uid = patch.object(attach.os, 'getuid', return_value=29562)
        uid.start()
        self.addCleanup(uid.stop)
        identity = patch.object(attach, 'process_identity', side_effect=lambda pid: {'pid': pid, 'uid': 29562, 'starttime_ticks': str(pid * 10), 'cmdline_sha256': 'a' * 64})
        identity.start()
        self.addCleanup(identity.stop)
        self.process = Mock(pid=445566)
        self.process.poll.return_value = None
        self.process.wait.return_value = 0

    def query(self, command):
        self.queries.append(command)
        self.assertEqual(command[:4], ['scontrol', 'show', 'job', '-o'])
        return self.holder if command[-1] == self.metadata['job_id'] else self.successor

    def launch(self, **kwargs):
        return attach.launch(self.root, self.config, 'smoke-test', query=self.query, now=self.now, **kwargs)

    def frozen_run(self, smoke=False, methods=None, status='interrupted'):
        directory = self.root / 'outputs/runs/smoke-test'
        directory.mkdir(parents=True)
        config = json.loads(self.config.read_bytes())
        active = [method for method in config['method_order'] if methods is None or method in methods]
        scope = 'smoke' if smoke else 'reference'
        manifest = {'run_id': 'smoke-test', 'root': str(self.root), 'run_dir': str(directory), 'config_sha256': hashlib.sha256(self.config.read_bytes()).hexdigest(), 'scope': scope, 'active_methods': active, 'effective_config': {'scope': scope, 'method_order': active}, 'scientific_identity_sha256': 'a' * 64, 'hardware_identity_sha256': 'b' * 64}
        manifest_path = directory / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        state = {'run_id': 'smoke-test', 'scope': scope, 'active_methods': active, 'status': status, 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(), 'scientific_identity_sha256': 'a' * 64, 'hardware_identity_sha256': 'b' * 64}
        state_path = directory / 'pool.json'
        state_path.write_text(json.dumps(state))
        return directory, manifest_path, state_path

    def test_detach_uses_only_owned_overlapping_step_and_records_identity(self):
        with patch.object(attach.subprocess, 'Popen', return_value=self.process) as popen:
            receipt = self.launch(detach=True, smoke=True, methods='eagle3,hass')
        self.assertEqual(receipt['status'], 'launched')
        self.assertEqual(receipt['launcher_pid'], 445566)
        self.assertEqual(receipt['launcher_process']['starttime_ticks'], '4455660')
        self.assertFalse(receipt['holder_and_successor_modified'])
        self.assertEqual(receipt['step_time_limit'], '12:00:00')
        command = popen.call_args.args[0]
        self.assertEqual(command[0], 'srun')
        for value in ('--jobid=2212576', '--overlap', '--gres=none', '--ntasks=1', '--cpus-per-task=64', '--mem=0', '--time=12:00:00', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', '/usr/bin/python3', 'taps_rebuild.pool', '--smoke'):
            self.assertIn(value, command)
        env_index = command.index('env')
        self.assertEqual(command[env_index + 1:env_index + 3], ['-u', 'CUDA_VISIBLE_DEVICES'])
        self.assertEqual(command[-2:], ['--methods', 'eagle3,hass'])
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertIs(popen.call_args.kwargs['stderr'], subprocess.STDOUT)
        self.assertNotIn('shell', popen.call_args.kwargs)
        self.process.wait.assert_not_called()
        self.assertEqual(len(self.queries), 2)
        self.assertTrue(all(command[0] == 'scontrol' for command in self.queries))
        self.assertEqual(json.loads(Path(receipt['receipt']).read_text()), receipt)
        self.assertTrue(Path(receipt['log']).is_file())
        self.assertEqual(json.loads(self.current.read_text()), self.metadata)
        self.assertFalse((self.root / 'outputs/runs/smoke-test').exists())

    def test_foreground_waits_only_for_its_own_step(self):
        with patch.object(attach.subprocess, 'Popen', return_value=self.process) as popen:
            receipt = self.launch()
        self.assertFalse(popen.call_args.kwargs['start_new_session'])
        self.process.wait.assert_called_once_with()
        self.process.poll.assert_not_called()
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['returncode'], 0)

    def test_pending_successor_single_node_range(self):
        self.successor = self.successor.replace('NumNodes=1 ', 'NumNodes=1-1 ')
        with patch.object(attach.subprocess, 'Popen', return_value=self.process):
            self.assertEqual(self.launch(detach=True)['status'], 'launched')
        for value in ['1-2', '0-1', '2-2']:
            with self.subTest(value=value):
                job = attach.fields(self.successor.replace('NumNodes=1-1 ', 'NumNodes=' + value + ' '))
                with self.assertRaises(attach.AttachError):
                    attach.verify_job(job, '2212580', pending=True, holder='2212576')

    def test_invalid_holder_identity_resources_or_state_prevents_launch(self):
        changes = [('UserId=mohamadzbib(29562)', 'UserId=other(1)'), ('JobState=RUNNING', 'JobState=COMPLETED'), ('JobName=deep10-hold', 'JobName=training'), ('Account=grad-students', 'Account=other'), ('Partition=high-priority', 'Partition=normal'), ('QOS=spec-priority', 'QOS=other'), ('NodeList=deep-chungus-10', 'NodeList=deep-chungus-9'), ('NumNodes=1', 'NumNodes=2'), ('NumCPUs=64', 'NumCPUs=32'), ('gres/gpu=8,gres/gpu:a100=8', 'gres/gpu=7,gres/gpu:a100=7')]
        for old, new in changes:
            with self.subTest(field=old), patch.object(attach.subprocess, 'Popen') as popen:
                original = self.holder
                self.holder = original.replace(old, new)
                with self.assertRaises(attach.AttachError):
                    self.launch(detach=True)
                self.holder = original
                popen.assert_not_called()
        self.assertFalse((self.root / 'outputs/launchers').exists())

    def test_stale_or_running_successor_prevents_launch_without_mutation(self):
        for old, new in [('afterany:2212576(unfulfilled)', 'afterany:1(unfulfilled)'), ('JobState=PENDING', 'JobState=RUNNING'), ('ReqNodeList=deep-chungus-10', 'ReqNodeList=deep-chungus-9')]:
            with self.subTest(field=old), patch.object(attach.subprocess, 'Popen') as popen:
                original = self.successor
                self.successor = original.replace(old, new)
                with self.assertRaises(attach.AttachError):
                    self.launch(detach=True)
                self.successor = original
                popen.assert_not_called()
        self.assertEqual(json.loads(self.current.read_text()), self.metadata)

    def test_optional_successor_and_near_deadline_bounded_time(self):
        self.metadata['successor'] = None
        self.current.write_text(json.dumps(self.metadata))
        self.holder = self.holder.replace('TimeLimit=2-00:00:00', 'TimeLimit=00:04:00').replace('RunTime=01:00:00', 'RunTime=00:00:00')
        with patch.object(attach.subprocess, 'Popen', return_value=self.process):
            receipt = self.launch(detach=True)
        self.assertIsNone(receipt['successor'])
        self.assertEqual(receipt['step_time_limit_seconds'], 180)
        self.assertEqual(len(self.queries), 1)

    def test_duration_parsing_and_holder_safety_reserve(self):
        self.assertEqual(attach.duration_seconds('2-03:04:05'), 183845)
        self.assertEqual(attach.duration_seconds('00:02:00'), 120)
        self.assertEqual(attach.step_seconds({'TimeLimit': '00:03:00', 'RunTime': '00:00:00'}), 120)
        self.assertEqual(attach.step_seconds({'TimeLimit': '2-00:00:00', 'RunTime': '01:00:00'}), 12 * 3600)
        self.assertEqual(attach.step_seconds({'TimeLimit': '12:00:00', 'RunTime': '00:00:00'}), 12 * 3600 - 60)
        self.assertEqual(attach.step_seconds({'TimeLimit': '09:00:00', 'RunTime': '00:30:00'}), 8 * 3600 + 30 * 60 - 60)
        for duration in ('UNLIMITED', '00:60:00', '00:00:60', '-01:00:00', 'bad'):
            with self.subTest(duration=duration), self.assertRaises(attach.AttachError):
                attach.duration_seconds(duration)
        with self.assertRaises(attach.AttachError):
            attach.step_seconds({'TimeLimit': '00:02:59', 'RunTime': '00:00:00'})

    def test_gpu_uid_and_run_id_guards_precede_slurm_queries(self):
        query = Mock()
        for options in ({'expected_gpus': 4}, {'expected_gpus': True}):
            with self.assertRaises(attach.AttachError):
                attach.launch(self.root, self.config, 'run', query=query, **options)
        for run_id in ('../escape', 'with spaces', ';touch', ''):
            with self.subTest(run_id=run_id), self.assertRaises(attach.AttachError):
                attach.launch(self.root, self.config, run_id, query=query)
        with patch.object(attach.os, 'getuid', return_value=0), self.assertRaises(attach.AttachError):
            attach.launch(self.root, self.config, 'run', query=query)
        query.assert_not_called()

    def test_changed_current_identity_is_rejected_after_live_queries(self):
        def query(command):
            result = self.query(command)
            if command[-1] == '2212580':
                self.current.write_text(json.dumps(dict(self.metadata, job_id='9999999')))
            return result

        with patch.object(attach.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(attach.AttachError, 'changed during launch'):
                attach.launch(self.root, self.config, 'run', detach=True, query=query)
            popen.assert_not_called()

    def test_config_root_existing_results_and_method_guards_preserve_files(self):
        run = self.root / 'outputs/runs/smoke-test'
        run.mkdir(parents=True)
        result = run / 'existing.jsonl'
        result.write_bytes(b'preserve-existing-results\n')
        with patch.object(attach.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(attach.AttachError, 'fresh ID'):
                self.launch(detach=True)
            popen.assert_not_called()
        self.assertEqual(result.read_bytes(), b'preserve-existing-results\n')
        for methods in ('unknown', 'eagle3,eagle3', 'eagle3,'):
            with self.subTest(methods=methods), self.assertRaises(attach.AttachError):
                self.launch(methods=methods)
        outside = self.base / 'outside.json'
        outside.write_bytes(self.config.read_bytes())
        with self.assertRaisesRegex(attach.AttachError, 'inside the selected project'):
            attach.launch(self.root, outside, 'fresh', query=Mock())
        value = json.loads(self.config.read_text())
        value['cluster_root'] = str(self.base / 'wrong')
        self.config.write_text(json.dumps(value))
        with self.assertRaisesRegex(attach.AttachError, 'cluster root differs'):
            attach.launch(self.root, self.config, 'fresh', query=Mock())

    def test_launch_receipts_never_overwrite_and_do_not_dump_environment(self):
        with patch.dict(attach.os.environ, {'HF_TOKEN': 'test-marker-never-recorded'}), patch.object(attach.subprocess, 'Popen', return_value=self.process) as popen:
            receipt = self.launch(detach=True)
            before = Path(receipt['receipt']).read_bytes()
            with self.assertRaisesRegex(attach.AttachError, 'refusing overwrite'):
                self.launch(detach=True)
            self.assertEqual(popen.call_count, 1)
        self.assertEqual(Path(receipt['receipt']).read_bytes(), before)
        self.assertNotIn('test-marker-never-recorded', before.decode())

    def test_immediately_failed_own_step_is_reported(self):
        self.process.poll.return_value = 7
        with patch.object(attach.subprocess, 'Popen', return_value=self.process):
            receipt = self.launch(detach=True)
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['returncode'], 7)
        self.assertEqual(json.loads(Path(receipt['receipt']).read_text())['status'], 'failed')

    def test_gpu_tres_generic_and_typed_counts_must_agree(self):
        self.assertEqual(attach.gpu_count('cpu=64,gres/gpu=8,gres/gpu:a100=8'), 8)
        self.assertEqual(attach.gpu_count('cpu=64,gres/gpu:a100=8'), 8)
        with self.assertRaises(attach.AttachError):
            attach.gpu_count('gres/gpu=8,gres/gpu:a100=7')

    def test_resume_reuses_compatible_frozen_run_without_overwriting_metadata(self):
        directory, manifest, state = self.frozen_run(smoke=True, methods=['eagle3', 'hass'])
        before = {path: path.read_bytes() for path in (manifest, state)}
        with patch.object(attach.subprocess, 'Popen', return_value=self.process) as popen:
            receipt = self.launch(detach=True, smoke=True, methods='hass,eagle3', resume=True)
        self.assertTrue(receipt['resume_requested'])
        self.assertEqual(receipt['resume_validation']['active_methods'], ['eagle3', 'hass'])
        self.assertEqual(receipt['resume_validation']['existing_pool_status'], 'interrupted')
        self.assertEqual(receipt['resume_validation']['manifest_sha256'], hashlib.sha256(before[manifest]).hexdigest())
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertNotIn('--resume', popen.call_args.args[0])
        self.assertEqual(popen.call_args.args[0][-2:], ['--methods', 'hass,eagle3'])
        self.assertEqual(json.loads(self.current.read_bytes()), self.metadata)
        self.assertTrue(directory.exists())

    def test_resume_completed_run_is_revalidated_through_existing_pool_logic(self):
        _, manifest, state = self.frozen_run(status='complete')
        before = manifest.read_bytes(), state.read_bytes()
        with patch.object(attach.subprocess, 'Popen', return_value=self.process):
            receipt = self.launch(detach=True, resume=True)
        self.assertEqual(receipt['resume_validation']['existing_pool_status'], 'complete')
        self.assertIn('reuses completed worker cases', receipt['resume_validation']['completed_run_behavior'])
        self.assertEqual((manifest.read_bytes(), state.read_bytes()), before)

    def test_default_still_refuses_compatible_existing_run(self):
        self.frozen_run()
        with patch.object(attach.subprocess, 'Popen') as popen, self.assertRaisesRegex(attach.AttachError, 'explicit --resume'):
            self.launch(detach=True)
        popen.assert_not_called()
        self.assertEqual(self.queries, [])

    def test_resume_requires_both_existing_regular_metadata_files(self):
        with patch.object(attach.subprocess, 'Popen') as popen, self.assertRaisesRegex(attach.AttachError, 'existing run directory'):
            self.launch(resume=True)
        popen.assert_not_called()
        directory, manifest, state = self.frozen_run()
        state.unlink()
        with patch.object(attach.subprocess, 'Popen') as popen, self.assertRaisesRegex(attach.AttachError, 'manifest.json and pool.json'):
            self.launch(resume=True)
        popen.assert_not_called()
        state.symlink_to(manifest)
        with patch.object(attach.subprocess, 'Popen') as popen, self.assertRaisesRegex(attach.AttachError, 'regular files without aliases'):
            self.launch(resume=True)
        popen.assert_not_called()
        state.unlink()
        state.mkdir()
        with patch.object(attach.subprocess, 'Popen') as popen, self.assertRaisesRegex(attach.AttachError, 'regular files without aliases'):
            self.launch(resume=True)
        popen.assert_not_called()
        self.assertTrue(directory.exists())

    def test_resume_mismatched_provenance_never_starts_srun(self):
        _, manifest_path, state_path = self.frozen_run()
        original_manifest = json.loads(manifest_path.read_bytes())
        original_state = json.loads(state_path.read_bytes())
        cases = [('manifest', 'run_id', 'different'), ('manifest', 'root', str(self.base)), ('manifest', 'config_sha256', 'c' * 64), ('manifest', 'scope', 'smoke'), ('manifest', 'active_methods', ['eagle3']), ('pool', 'run_id', 'different'), ('pool', 'scope', 'smoke'), ('pool', 'active_methods', ['eagle3']), ('pool', 'manifest_sha256', 'c' * 64), ('pool', 'scientific_identity_sha256', 'c' * 64), ('pool', 'hardware_identity_sha256', 'c' * 64)]
        for kind, field, replacement in cases:
            with self.subTest(kind=kind, field=field):
                manifest = dict(original_manifest)
                state = dict(original_state)
                (manifest if kind == 'manifest' else state)[field] = replacement
                manifest_path.write_text(json.dumps(manifest))
                if kind == 'manifest':
                    state['manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                state_path.write_text(json.dumps(state))
                with patch.object(attach.subprocess, 'Popen') as popen, self.assertRaises(attach.AttachError):
                    self.launch(resume=True)
                popen.assert_not_called()
        manifest_path.write_text(json.dumps(original_manifest))
        state_path.write_text(json.dumps(original_state))
        for options in ({'smoke': True}, {'methods': 'eagle3'}):
            with patch.object(attach.subprocess, 'Popen') as popen, self.assertRaisesRegex(attach.AttachError, 'scope or methods filter'):
                self.launch(resume=True, **options)
            popen.assert_not_called()
        self.assertEqual(self.queries, [])

    def test_resume_manifest_change_during_live_validation_prevents_launch(self):
        _, manifest, _ = self.frozen_run()
        original = manifest.read_bytes()
        def query(command):
            result = self.query(command)
            if command[-1] == '2212580':
                value = json.loads(manifest.read_bytes())
                value['config_sha256'] = 'c' * 64
                manifest.write_text(json.dumps(value))
            return result
        with patch.object(attach.subprocess, 'Popen') as popen, self.assertRaises(attach.AttachError):
            attach.launch(self.root, self.config, 'smoke-test', detach=True, resume=True, query=query, now=self.now)
        popen.assert_not_called()
        self.assertNotEqual(manifest.read_bytes(), original)

    def test_resume_rejects_parent_directory_aliases(self):
        directory, _, _ = self.frozen_run()
        actual = self.root / 'outside-run'
        directory.rename(actual)
        directory.symlink_to(actual, target_is_directory=True)
        with patch.object(attach.subprocess, 'Popen') as popen, self.assertRaisesRegex(attach.AttachError, 'symlink aliases'):
            self.launch(resume=True)
        popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
