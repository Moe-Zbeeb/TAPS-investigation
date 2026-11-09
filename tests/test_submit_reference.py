import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
FILE = PROJECT / 'experiments/official-reproduction-20261109/submit_reference.py'
SPEC = importlib.util.spec_from_file_location('reference_submission_under_test', FILE)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


class ReferenceSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / 'project'
        self.root.mkdir()
        self.legacy = self.root.parent / 'legacy'
        self.config = self.root / 'configs/reproduction.json'
        self.config.parent.mkdir()
        self.config.write_text(json.dumps({'cluster_root': str(self.root), 'legacy_runtime_root': str(self.legacy)}))
        current = self.legacy / 'cluster/deep10/current.json'
        current.parent.mkdir(parents=True)
        current.write_text(json.dumps({'node': 'deep-chungus-10', 'gpus': 8, 'job_id': '100', 'successor': '101'}))
        self.holder = f'JobId=100 JobName=deep10-hold UserId=user({os.getuid()}) JobState=RUNNING Account=grad-students Partition=high-priority QOS=spec-priority NodeList=deep-chungus-10 ReqNodeList=deep-chungus-10 NumNodes=1 NumCPUs=128 TRES=cpu=128,gres/gpu=8,gres/gpu:a100=8 TimeLimit=2-00:00:00 RunTime=01:00:00'
        self.successor = f'JobId=101 JobName=deep10-hold UserId=user({os.getuid()}) JobState=PENDING Account=grad-students Partition=high-priority QOS=spec-priority NodeList=(null) ReqNodeList=deep-chungus-10 NumNodes=1-1 NumCPUs=128 ReqTRES=cpu=128,gres/gpu=8,gres/gpu:a100=8 Dependency=afterany:100(unfulfilled)'
        self.queue = ['100|deep10-hold|RUNNING|high-priority|spec-priority|deep-chungus-10|gres:gpu:a100:8', '101|deep10-hold|PENDING|high-priority|spec-priority||gres:gpu:a100:8']
        self.nodes = {node: f'NodeName={node} State=IDLE CPUAlloc=0 CPUTot=128 Gres=gpu:a100:{count} CfgTRES=cpu=128,gres/gpu={count},gres/gpu:a100={count} AllocTRES= Partitions=high-priority,low-priority' for node, count in [('deep-chungus-3', 9), ('deep-chungus-11', 8)]}
        self.plan = self.root / 'plan.json'
        self.value = {'schema_version': 1, 'status': 'prepared_not_submitted', 'root': str(self.root), 'config': str(self.config), 'config_sha256_at_preparation': submit.pool.digest(self.config), 'jobs': []}
        for specification in submit.SPECS:
            keys = ('method', 'run_id', 'expected_node', 'expected_gpus', 'expected_partition', 'expected_qos')
            self.value['jobs'].append({**dict(zip(keys, specification)), 'command_argv': submit.command_argv(self.root, specification), 'sbatch_argv': submit.sbatch_argv(self.root, specification)})
        self.write_plan()
        self.calls = []
        self.submissions = []
        self.live_jobs = {}
        self.deployment = {'code': {'code_commit': 'a' * 40, 'code_dirty': False}, 'config_sha256': submit.pool.digest(self.config), 'assets_receipt_sha256': 'b' * 64, 'sources_receipt_sha256': 'c' * 64}
        uid = patch.object(submit.attach, 'UID', os.getuid())
        uid.start()
        self.addCleanup(uid.stop)

    def write_plan(self):
        self.plan.write_text(json.dumps(self.value))
        self.sha = submit.pool.digest(self.plan)

    def query(self, arguments):
        self.calls.append(arguments)
        if arguments[0] == 'squeue':
            return '\n'.join(self.queue) + '\n'
        if arguments[:3] == ['scontrol', 'show', 'node']:
            return self.nodes[arguments[-1]]
        if arguments[-1] == '100':
            return self.holder
        if arguments[-1] == '101':
            return self.successor
        return self.live_jobs[arguments[-1]]

    def prepare(self, *args):
        return copy.deepcopy(self.deployment)

    def invoke(self, **kwargs):
        return submit.run(self.root, self.plan, self.sha, query=self.query, prepare=self.prepare, submitter=self.submitter, **kwargs)

    def submitter(self, arguments, environment, root):
        index = len(self.submissions)
        name = next(argument.split('=', 1)[1] for argument in arguments if argument.startswith('--job-name='))
        job = next(value for value in self.value['jobs'] if value['run_id'] == name)
        job_id = str(200 + index)
        self.submissions.append((arguments, environment, root))
        self.assertTrue((self.root / 'outputs/submissions' / (job['run_id'] + '.json')).exists())
        self.queue.append('|'.join([job_id, job['run_id'], 'PENDING', job['expected_partition'], job['expected_qos'], '', 'gres:gpu:a100:' + str(job['expected_gpus'])]))
        self.live_jobs[job_id] = f"JobId={job_id} JobName={job['run_id']} UserId=user({os.getuid()}) JobState=PENDING Account=grad-students Partition={job['expected_partition']} QOS={job['expected_qos']} ReqNodeList={job['expected_node']} NodeList=(null) NumNodes=1 NumCPUs=64 ReqTRES=cpu=64,gres/gpu={job['expected_gpus']},gres/gpu:a100={job['expected_gpus']} Requeue=1 TimeLimit=12:00:00"
        return subprocess.CompletedProcess(arguments, 0, job_id + ';madry\n', '')

    def test_batch_path_contains_verified_cluster_slurm_directory(self):
        command = submit.command_argv(self.root, submit.SPECS[0])
        paths = [item.removeprefix('PATH=') for item in command if item.startswith('PATH=')]
        self.assertEqual(len(paths), 1)
        self.assertIn('/usr/local/bin', paths[0].split(os.pathsep))

    def test_readonly_default_validates_without_any_files_or_submissions(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'validated_not_submitted')
        self.assertEqual(result['admission']['high_priority_bound'], 17)
        self.assertFalse(self.submissions)
        self.assertFalse((self.root / 'outputs').exists())
        self.assertTrue(all(command[0] in ('scontrol', 'squeue') for command in self.calls))

    def test_explicit_submit_saves_durable_per_job_receipts_and_preserves_home(self):
        with patch.dict(os.environ, {'HOME': '/afs/inherited', 'HF_TOKEN': 'private', 'SBATCH_GET_USER_ENV': '1', 'PYTHONPATH': 'injected'}, clear=True):
            result = self.invoke(submit=True)
        self.assertEqual(result['status'], 'submitted_verified')
        self.assertEqual(len(self.submissions), 2)
        for arguments, environment, root in self.submissions:
            self.assertEqual(environment['HOME'], '/afs/inherited')
            self.assertNotIn('HF_TOKEN', environment)
            self.assertNotIn('SBATCH_GET_USER_ENV', environment)
            self.assertNotIn('PYTHONPATH', environment)
            self.assertIn('--export=ALL', arguments)
            self.assertFalse(any('HOME=' in argument for argument in arguments))
            self.assertNotIn('scancel', arguments)
        for record in result['receipts']:
            path = Path(record['path'])
            receipt = json.loads(path.read_bytes())
            self.assertEqual(receipt['status'], 'submitted_verified')
            self.assertEqual(submit.pool.digest(path), record['sha256'])
            self.assertEqual(receipt['deployment'], self.deployment)
        with self.assertRaises(submit.SubmissionError):
            self.invoke(submit=True)
        self.assertEqual(len(self.submissions), 2)

    def test_hash_root_config_owned_regular_and_unknown_payload_guards(self):
        with self.assertRaises(submit.SubmissionError):
            submit.read_plan(self.root, self.plan, '0' * 64)
        self.value['jobs'][0]['sbatch_argv'][-1] += '; unwanted'
        self.write_plan()
        with self.assertRaises(submit.SubmissionError):
            self.invoke()
        self.value['jobs'][0]['sbatch_argv'] = submit.sbatch_argv(self.root, submit.SPECS[0])
        self.write_plan()
        self.config.write_text('{}')
        with self.assertRaises(submit.SubmissionError):
            self.invoke()
        alias = self.root / 'alias.json'
        alias.symlink_to(self.plan)
        with self.assertRaises(submit.SubmissionError):
            submit.read_plan(self.root, alias, self.sha)
        self.assertFalse(self.submissions)

    def test_stale_holder_successor_duplicate_job_and_high_cap_guards(self):
        for old, new, attribute in [('JobState=RUNNING', 'JobState=PENDING', 'holder'), ('afterany:100', 'afterany:999', 'successor')]:
            before = getattr(self, attribute)
            setattr(self, attribute, before.replace(old, new))
            with self.assertRaises(ValueError):
                self.invoke()
            setattr(self, attribute, before)
        self.queue.append('999|other|RUNNING|high-priority|spec-priority|deep-chungus-2|gres:gpu:a100:2')
        with self.assertRaisesRegex(submit.SubmissionError, 'exceed18'):
            self.invoke()
        self.queue.pop()
        self.queue.append('999|other|PENDING|low-priority,high-priority|spec-priority||gres:gpu:a100:2')
        with self.assertRaisesRegex(submit.SubmissionError, 'exceed18'):
            self.invoke()
        self.queue.pop()
        self.queue.append('999|official-hass-20261109-v1|PENDING|low-priority|normal||gres:gpu:a100:8')
        with self.assertRaisesRegex(submit.SubmissionError, 'already uses'):
            self.invoke()
        self.assertFalse(self.submissions)

    def test_idle_full_node_and_duplicate_run_guards(self):
        node = 'deep-chungus-11'
        before = self.nodes[node]
        for changed in (before.replace('State=IDLE', 'State=MIXED'), before.replace('gpu=8', 'gpu=7').replace('a100=8', 'a100=7')):
            self.nodes[node] = changed
            with self.assertRaises(submit.SubmissionError):
                self.invoke()
        self.nodes[node] = before
        path = self.root / 'outputs/runs/official-eagle3-20261109-v1'
        path.mkdir(parents=True)
        with self.assertRaisesRegex(submit.SubmissionError, 'duplicate'):
            self.invoke()
        self.assertFalse(self.submissions)

    def test_partial_success_is_preserved_and_second_failure_never_retried(self):
        real = self.submitter
        def fail_second(arguments, environment, root):
            if self.submissions:
                return subprocess.CompletedProcess(arguments, 1, '', 'rejected')
            return real(arguments, environment, root)
        with self.assertRaises(submit.SubmissionError):
            submit.run(self.root, self.plan, self.sha, submit=True, query=self.query, prepare=self.prepare, submitter=fail_second)
        directory = self.root / 'outputs/submissions'
        first = json.loads((directory / 'official-eagle3-20261109-v1.json').read_bytes())
        second = json.loads((directory / 'official-hass-20261109-v1.json').read_bytes())
        self.assertEqual(first['status'], 'submitted_verified')
        self.assertEqual(second['status'], 'failed_or_uncertain')
        self.assertFalse(second['cancellation_performed'])
        self.assertEqual(len(self.submissions), 1)
        with self.assertRaises(submit.SubmissionError):
            self.invoke(submit=True)

    def test_accepted_job_is_saved_before_live_verification_failure(self):
        real = self.query
        def changed(arguments):
            text = real(arguments)
            return text.replace('Account=grad-students', 'Account=other') if arguments[-1] == '200' else text
        with self.assertRaises(submit.SubmissionError):
            submit.run(self.root, self.plan, self.sha, submit=True, query=changed, prepare=self.prepare, submitter=self.submitter)
        path = self.root / 'outputs/submissions/official-eagle3-20261109-v1.json'
        receipt = json.loads(path.read_bytes())
        self.assertEqual(receipt['job_id'], '200')
        self.assertEqual(receipt['status'], 'submitted_verification_failed')
        self.assertFalse(receipt['cancellation_performed'])
        self.assertEqual(len(self.submissions), 1)

    def test_ambiguous_submission_blocks_duplicate_retry(self):
        ambiguous = lambda *args: subprocess.CompletedProcess(args[0], 0, 'unexpected output', '')
        with self.assertRaises(submit.SubmissionError):
            submit.run(self.root, self.plan, self.sha, submit=True, query=self.query, prepare=self.prepare, submitter=ambiguous)
        path = self.root / 'outputs/submissions/official-eagle3-20261109-v1.json'
        self.assertEqual(json.loads(path.read_bytes())['status'], 'failed_or_uncertain')
        with self.assertRaises(submit.SubmissionError):
            self.invoke(submit=True)

    def test_sanitization_removes_option_controls_and_credentials_without_home_assignment(self):
        original = {'HOME': '/original', 'PATH': '/bin', 'SLURM_GET_USER_ENV': '1', 'SLURM_EXPORT_ENV': 'NONE', 'SRUN_EXPORT_ENV': 'NONE', 'AWS_REGION': 'private', 'private_secret': 'private', 'PYTHONHOME': 'injected'}
        self.assertEqual(submit.sanitized_environment(original), {'HOME': '/original', 'PATH': '/bin'})
        self.assertEqual(submit.requested_gpus('gres:gpu:a100:9'), 9)
        self.assertEqual(submit.requested_gpus('gres/gpu:a100:8'), 8)
        self.assertEqual(submit.requested_gpus('gres:gpu:a100:8,gres:gpu:h100:4'), 12)

    def test_code_or_config_change_after_first_submission_blocks_second(self):
        real = self.prepare
        def changed(*args):
            value = real(*args)
            if self.submissions:
                value['code']['code_commit'] = 'd' * 40
            return value
        with self.assertRaisesRegex(submit.SubmissionError, 'identity changed'):
            submit.run(self.root, self.plan, self.sha, submit=True, query=self.query, prepare=changed, submitter=self.submitter)
        self.assertEqual(len(self.submissions), 1)
        self.assertFalse((self.root / 'outputs/submissions/official-hass-20261109-v1.json').exists())

    def test_source_preflight_refuses_uncommitted_deployment(self):
        prepared = {'code': {'code_dirty': True}}
        with patch.object(submit.pool, 'effective_config'), patch.object(submit.pool, 'prerequisites', return_value=prepared), self.assertRaisesRegex(submit.SubmissionError, 'final clean'):
            submit.source_preflight(self.root, self.config, self.value['jobs'])

    def test_eagle3_only_ignores_busy_hass_node_and_creates_only_eagle3_receipt(self):
        self.nodes['deep-chungus-11'] = self.nodes['deep-chungus-11'].replace('State=IDLE', 'State=MIXED')
        result = self.invoke(submit=True, methods='eagle3')
        self.assertEqual(result['selected_methods'], ['eagle3'])
        self.assertEqual(len(self.submissions), 1)
        arguments = self.submissions[0][0]
        self.assertIn('--nodelist=deep-chungus-3', arguments)
        self.assertIn('--gres=gpu:a100:9', arguments)
        self.assertFalse((self.root / 'outputs/submissions/official-hass-20261109-v1.json').exists())
        self.assertFalse(any(command[-1] == 'deep-chungus-11' for command in self.calls))

    def test_hass_only_uses_hass_spec_and_ignores_unselected_eagle3_paths(self):
        directory = self.root / 'outputs/runs/official-eagle3-20261109-v1'
        directory.mkdir(parents=True)
        receipts = self.root / 'outputs/submissions'
        receipts.mkdir()
        (receipts / 'official-eagle3-20261109-v1.json').write_text('{}')
        self.queue.append('299|official-eagle3-20261109-v1|RUNNING|high-priority|spec-priority|deep-chungus-3|gres:gpu:a100:9')
        self.nodes['deep-chungus-3'] = self.nodes['deep-chungus-3'].replace('State=IDLE', 'State=ALLOCATED')
        result = self.invoke(submit=True, methods='hass')
        self.assertEqual(result['selected_methods'], ['hass'])
        arguments = self.submissions[0][0]
        self.assertIn('--nodelist=deep-chungus-11', arguments)
        self.assertIn('--gres=gpu:a100:8', arguments)
        self.assertIn('--partition=low-priority', arguments)
        self.assertIn('--qos=normal', arguments)
        self.assertEqual(result['receipts'][0]['run_id'], 'official-hass-20261109-v1')
        receipt = json.loads(Path(result['receipts'][0]['path']).read_bytes())
        self.assertEqual(receipt['resource_admission']['high_priority_bound'], 17)
        self.assertFalse(any(command[-1] == 'deep-chungus-3' for command in self.calls))

    def test_hass_only_still_requires_idle_full_node_and_global_high_cap(self):
        before = self.nodes['deep-chungus-11']
        self.nodes['deep-chungus-11'] = before.replace('State=IDLE', 'State=MIXED')
        with self.assertRaises(submit.SubmissionError):
            self.invoke(methods='hass')
        self.nodes['deep-chungus-11'] = before
        self.queue.append('299|official-eagle3-20261109-v1|RUNNING|high-priority|spec-priority|deep-chungus-3|gres:gpu:a100:11')
        with self.assertRaisesRegex(submit.SubmissionError, 'exceed18'):
            self.invoke(methods='hass')
        self.assertFalse(self.submissions)

    def test_subset_selection_does_not_bypass_full_original_plan_validation(self):
        self.value['jobs'][1]['sbatch_argv'][-1] += '; unwanted'
        self.write_plan()
        with self.assertRaises(submit.SubmissionError):
            self.invoke(methods='eagle3')
        with self.assertRaises(submit.SubmissionError):
            self.invoke(methods='eagle2')
        self.assertFalse(self.submissions)


if __name__ == '__main__':
    unittest.main()
