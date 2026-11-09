import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from taps_rebuild import assets, runtime, serving


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = (Path(self.temporary.name) / 'project').resolve()
        self.root.mkdir()
        self.legacy = (Path(self.temporary.name) / 'legacy').resolve()
        self.python = self.legacy / 'python/cpython-3.10/bin/python3.10'
        self.uv = self.legacy / 'tools/uv'
        for path in (self.python, self.uv):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('executable')
            path.chmod(0o700)
        (self.root / 'configs').mkdir()
        self.requirements = self.root / 'configs/runtime-vllm.txt'
        self.requirements.write_bytes((Path(__file__).parents[1] / 'configs/runtime-vllm.txt').read_bytes())
        self.config = json.loads((Path(__file__).parents[1] / 'configs/serving.json').read_bytes())
        self.config.update(cluster_root=str(self.root), legacy_runtime_root=str(self.legacy))
        (self.root / 'configs/serving.json').write_text(json.dumps(self.config))
        (self.root / 'src/taps_rebuild').mkdir(parents=True)
        (self.root / 'src/taps_rebuild/serving.py').write_text('source')
        self.calls = []

    def probe(self, python=None):
        path = python or self.root / 'envs/serving-vllm/bin/python'
        return {'python': [3, 10, 17], 'prefix': str(path.parent.parent), 'base_prefix': str(self.python.parent.parent), 'packages': dict(runtime.EXPECTED), 'torch_cuda': '12.4'}

    def runner(self, command, environment, log, timeout=7200, capture=False):
        self.calls.append((command, dict(environment)))
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')
        if command[:2] == [str(self.uv), 'venv']:
            path = self.root / 'envs/serving-vllm/bin/python'
            path.parent.mkdir(parents=True)
            path.write_text('virtualenv')
            path.chmod(0o700)
            (path.parent.parent / 'pyvenv.cfg').write_text('home = ' + str(self.python.parent))
        if command[:2] == [str(self.uv), '--version']:
            return 'uv 0.9.0\n'
        if command[:3] == [str(self.uv), 'pip', 'freeze']:
            return 'vllm==0.8.5.post1\ntorch==2.6.0+cu124\n'
        if command[0] == str(self.python):
            return json.dumps({'version': [3, 10], 'prefix': str(self.python.parent.parent), 'base_prefix': str(self.python.parent.parent)})
        if '-c' in command:
            return json.dumps(self.probe())
        if '-m' in command and command[command.index('-m') + 1] == 'taps_rebuild.serving':
            name = command[command.index('--method') + 1]
            checkpoints = self.checkpoints(self.root, self.config['methods'][name])[0]
            _, _, fingerprint = serving.server_command(self.root, self.config, name, checkpoints)
            return json.dumps({'server_identity_sha256': fingerprint})
        return '' if capture else None

    def checkpoints(self, root, method, verify=False):
        names = [method['target_asset']] + ([method['draft_asset']] if method['mode'] == 'eagle3' else [])
        return {name: {'path': str(root / 'models' / name), 'model_identity_sha256': 'a' * 64, 'source': {}} for name in names}, 'b' * 64

    def patches(self):
        return patch.multiple(runtime.platform, system=lambda: 'Linux', machine=lambda: 'x86_64')

    def test_cpu_guard_prevents_any_preparation(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(assets.PreparationError):
            runtime.prepare(self.root, self.python, self.uv, runner=self.runner)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '1', 'CUDA_VISIBLE_DEVICES': '0'}, clear=True), self.assertRaises(assets.PreparationError):
            runtime.prepare(self.root, self.python, self.uv, runner=self.runner)
        self.assertFalse(self.calls)
        self.assertFalse((self.root / 'outputs').exists())

    def test_pinned_wheels_are_hashed_and_builder_never_invokes_gpu_or_slurm(self):
        requirements = self.requirements.read_text()
        self.assertEqual(requirements.count('#sha256='), 4)
        self.assertIn('torch-2.6.0%2Bcu124-cp310-cp310-linux_x86_64.whl', requirements)
        commands = runtime.install_commands(self.uv, self.python, self.root / 'envs/serving-vllm', self.requirements, False)
        self.assertEqual(commands[0][:2], [str(self.uv), 'venv'])
        self.assertIn(runtime.EXCLUDE_NEWER, commands[1])
        self.assertIn('vllm,torch,torchvision,torchaudio,xformers', commands[1])
        self.assertFalse(any(value in ('sbatch', 'srun', 'scancel', 'nvidia-smi') for command in commands for value in command))
        self.assertEqual(len(runtime.install_commands(self.uv, self.python, self.root / 'envs/serving-vllm', self.requirements, True)), 1)

    def test_environment_ignores_credentials_and_legacy_configuration(self):
        with patch.dict(os.environ, {'UV_INDEX_URL': 'private', 'PIP_EXTRA_INDEX_URL': 'secret', 'HF_TOKEN': 'secret', 'PYTHONPATH': 'old', 'LD_PRELOAD': 'old'}, clear=True):
            value = runtime.clean_environment(self.root, self.root / '.cache')
        self.assertEqual(value['UV_LINK_MODE'], 'copy')
        self.assertEqual(value['UV_NO_CONFIG'], '1')
        self.assertEqual(value['UV_PYTHON_DOWNLOADS'], 'never')
        self.assertEqual(value['UV_HTTP_TIMEOUT'], '300')
        self.assertEqual(value['UV_CONCURRENT_DOWNLOADS'], '4')
        self.assertEqual(value['NETRC'], '/dev/null')
        self.assertEqual(value['TAPS_RUNTIME_WORKDIR'], str(self.root))
        for key in ('HF_TOKEN', 'UV_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'PYTHONPATH', 'LD_PRELOAD'):
            self.assertNotIn(key, value)

    def test_path_validation_rejects_other_python_or_legacy_virtualenv(self):
        outside = self.root / 'python'
        outside.write_text('executable')
        outside.chmod(0o700)
        with self.assertRaisesRegex(runtime.RuntimeError, 'managed Python'):
            runtime.validate_paths(self.root, outside, self.uv, self.requirements, 'configs/serving.json')
        self.config['engines']['vllm']['serving_python'] = str(self.legacy / 'envs/taps-eagle/bin/python')
        (self.root / 'configs/serving.json').write_text(json.dumps(self.config))
        with self.assertRaisesRegex(runtime.RuntimeError, 'serving configuration|Serving configuration'):
            runtime.validate_paths(self.root, self.python, self.uv, self.requirements, 'configs/serving.json')

    def test_probe_requires_exact_local_cuda_build_without_gpu_execution(self):
        python = self.root / 'envs/serving-vllm/bin/python'
        python.parent.mkdir(parents=True)
        python.write_text('virtualenv')
        python.chmod(0o700)
        (python.parent.parent / 'pyvenv.cfg').write_text('home = ' + str(self.python.parent))
        value = runtime.package_probe(python, {'CUDA_VISIBLE_DEVICES': ''}, self.root / 'log', self.runner)
        self.assertFalse(value['gpu_execution_tested'])
        script = self.calls[-1][0][-1]
        self.assertIn('import vllm.v1.worker.gpu_worker', script)
        self.assertIn('triton', script)
        self.assertIn("'setuptools'", script)
        wrong = self.probe()
        wrong['packages']['torch'] = '2.6.0'
        with self.assertRaises(runtime.RuntimeError):
            runtime.package_probe(python, {}, self.root / 'log', lambda *args, **kwargs: json.dumps(wrong))

    def test_full_mocked_preparation_records_freeze_and_stock_preflights_then_resumes(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': '', 'SLURMD_NODENAME': 'node'}, clear=True), self.patches(), patch.object(serving, 'checkpoint_records', side_effect=self.checkpoints), patch.object(serving, 'runtime_check', return_value={'runtime_verified': True}):
            path, receipt = runtime.prepare(self.root, self.python, self.uv, runner=self.runner)
            self.assertEqual(receipt['status'], 'complete')
            self.assertFalse(receipt['gpu_execution_tested'])
            self.assertFalse(receipt['model_loading_tested'])
            self.assertEqual(set(receipt['module_command_preflights']), {'vllm_ar', 'vllm_eagle3'})
            self.assertTrue(Path(receipt['freeze_path']).is_file())
            self.assertEqual(json.loads(path.read_bytes()), receipt)
            self.calls.clear()
            runtime.prepare(self.root, self.python, self.uv, runner=self.runner)
            self.assertFalse(any(command[:2] == [str(self.uv), 'venv'] or command[:3] == [str(self.uv), 'pip', 'install'] for command, _ in self.calls))
            with patch.object(runtime, 'package_probe', side_effect=runtime.RuntimeError('changed runtime')), self.assertRaises(runtime.RuntimeError):
                runtime.prepare(self.root, self.python, self.uv, runner=self.runner)
            self.assertEqual(json.loads(path.read_bytes())['status'], 'failed')
            self.requirements.write_text(self.requirements.read_text() + '\n')
            with self.assertRaisesRegex(runtime.RuntimeError, 'provenance differs'):
                runtime.prepare(self.root, self.python, self.uv, runner=self.runner)

    def test_unknown_environment_is_never_modified(self):
        (self.root / 'envs/serving-vllm').mkdir(parents=True)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, clear=True), self.patches(), self.assertRaisesRegex(runtime.RuntimeError, 'unrecognized'):
            runtime.prepare(self.root, self.python, self.uv, runner=self.runner)
        self.assertFalse(self.calls)

    def test_package_failure_cannot_create_complete_receipt(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, clear=True), self.patches(), patch.object(runtime, 'package_probe', side_effect=runtime.RuntimeError('wrong package')), self.assertRaises(runtime.RuntimeError):
            runtime.prepare(self.root, self.python, self.uv, runner=self.runner)
        self.assertEqual(json.loads((self.root / 'outputs/preparation/runtime-vllm/runtime.json').read_bytes())['status'], 'failed')
        self.assertTrue((self.root / 'outputs/preparation/runtime-vllm/intent.json').exists())

    def test_node_local_cache_requires_owned_live_job_current_node_and_taps_name(self):
        path = Path('/tmp') / ('taps-rebuild-vllm-' + str(os.getuid()) + '-123-node-test')
        job = 'UserId=user(' + str(os.getuid()) + ') JobState=RUNNING NodeList=node-test NumNodes=1'
        mount = '1 0 8:1 / / rw - ext4 /dev/nvme0n1p1 rw\n'
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'node-test', 'CUDA_VISIBLE_DEVICES': ''}, clear=True), patch.object(runtime.socket, 'gethostname', return_value='node-test.example'):
            actual, record = runtime.cache_directory(self.root, path, query=lambda command: job, mountinfo=mount)
            self.assertEqual(actual, path)
            self.assertEqual(record['kind'], 'node_local_tmp')
            for invalid in (Path('/tmp/unrelated-cache'), self.root / path.name):
                with self.assertRaises(runtime.RuntimeError):
                    runtime.cache_directory(self.root, invalid, query=lambda command: job, mountinfo=mount)
            with self.assertRaises(runtime.RuntimeError):
                runtime.cache_directory(self.root, path, query=lambda command: job.replace('RUNNING', 'PENDING'), mountinfo=mount)
            with self.assertRaises(runtime.RuntimeError):
                runtime.cache_directory(self.root, path, query=lambda command: job, mountinfo=mount.replace('ext4', 'nfs4'))
            with patch.object(runtime.socket, 'gethostname', return_value='login'), self.assertRaises(runtime.RuntimeError):
                runtime.cache_directory(self.root, path, query=lambda command: job, mountinfo=mount)

    def test_failed_source_only_retry_preserves_exact_attempt_and_rejects_changed_config(self):
        import hashlib
        output = self.root / 'old'
        output.mkdir()
        intent, receipt = output / 'intent.json', output / 'runtime.json'
        old = {'source_sha256': 'a' * 64, 'requirements_sha256': 'b' * 64, 'serving_config_sha256': 'c' * 64, 'environment': str(self.root / 'envs/serving-vllm'), 'expected_packages': runtime.EXPECTED}
        intent.write_text(json.dumps({'identity': old}, indent=3))
        receipt.write_text(json.dumps({'status': 'failed', 'identity': old, 'recovery_observation': {'reason': 'Measured NFS extraction stall'}}))
        (output / 'build.log').write_bytes(b'old-attempt\n')
        before, receipt_before = intent.read_bytes(), receipt.read_bytes()
        digest = hashlib.sha256(before).hexdigest()
        current = dict(old, source_sha256='d' * 64)
        for identity, expected in ((dict(current, serving_config_sha256='e' * 64), digest), (current, '0' * 64)):
            with self.assertRaises(runtime.RuntimeError):
                runtime.retry_intent(intent, receipt, identity, expected)
        self.assertFalse((output / 'retries').exists())
        migration = runtime.retry_intent(intent, receipt, current, digest)
        archive = Path(migration['archive'])
        self.assertEqual((archive / 'intent.json').read_bytes(), before)
        self.assertEqual((archive / 'runtime.json').read_bytes(), receipt_before)
        self.assertEqual((archive / 'build.log').read_bytes(), b'old-attempt\n')
        self.assertEqual(json.loads(intent.read_bytes())['identity'], current)

    def test_base_interpreter_failure_is_recorded_without_installing(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, clear=True), self.patches(), self.assertRaises(runtime.RuntimeError):
            runtime.prepare(self.root, self.python, self.uv, runner=lambda *args, **kwargs: (_ for _ in ()).throw(runtime.RuntimeError('base failed')))
        value = json.loads((self.root / 'outputs/preparation/runtime-vllm/runtime.json').read_bytes())
        self.assertEqual(value['status'], 'failed')
        self.assertEqual(value['stage'], 'base-interpreter')

    def test_retry_migration_preserves_partial_environment_and_reinstalls_pinned_dependencies(self):
        import hashlib
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': '', 'SLURMD_NODENAME': 'node'}, clear=True), self.patches(), patch.object(serving, 'checkpoint_records', side_effect=self.checkpoints), patch.object(serving, 'runtime_check', return_value={'runtime_verified': True}):
            with patch.object(runtime, 'package_probe', side_effect=runtime.RuntimeError('interrupted')), self.assertRaises(runtime.RuntimeError):
                runtime.prepare(self.root, self.python, self.uv, runner=self.runner)
            directory = self.root / 'outputs/preparation/runtime-vllm'
            intent, receipt = directory / 'intent.json', directory / 'runtime.json'
            old = json.loads(intent.read_bytes())
            old['identity']['source_sha256'] = 'a' * 64
            intent.write_text(json.dumps(old))
            failed = json.loads(receipt.read_bytes())
            failed['identity'] = old['identity']
            receipt.write_text(json.dumps(failed))
            expected = hashlib.sha256(intent.read_bytes()).hexdigest()
            self.calls.clear()
            _, result = runtime.prepare(self.root, self.python, self.uv, runner=self.runner, retry_from=expected)
            self.assertEqual(result['status'], 'complete')
            self.assertFalse(any(command[:2] == [str(self.uv), 'venv'] for command, _ in self.calls))
            self.assertTrue(any(command[:3] == [str(self.uv), 'pip', 'install'] for command, _ in self.calls))
            self.assertEqual(len(list((directory / 'retries').glob('*/migration.json'))), 1)

    def test_physical_prefix_retains_venv_instead_of_resolving_python_to_base(self):
        actual = self.root / 'physical-env'
        (actual / 'bin').mkdir(parents=True)
        (actual / 'bin/python').symlink_to(self.python)
        (actual / 'pyvenv.cfg').write_text('home = ' + str(self.python.parent))
        alias = self.root / 'alias-env'
        alias.symlink_to(actual, target_is_directory=True)
        self.assertEqual(native := runtime.physical_python(alias / 'bin/python'), actual / 'bin/python')
        self.assertNotEqual(native, (alias / 'bin/python').resolve())
        with self.assertRaises(runtime.RuntimeError):
            runtime.physical_python(self.python)

    def test_complete_receipt_separates_configured_and_executed_command_identities(self):
        physical = self.root / 'physical-envs'
        physical.mkdir()
        (self.root / 'envs').symlink_to(physical, target_is_directory=True)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, clear=True), self.patches(), patch.object(serving, 'checkpoint_records', side_effect=self.checkpoints), patch.object(serving, 'runtime_check', return_value={'runtime_verified': True}):
            _, receipt = runtime.prepare(self.root, self.python, self.uv, runner=self.runner)
        self.assertEqual(receipt['packages']['executed_python'], str(physical / 'serving-vllm/bin/python'))
        for check in receipt['stock_command_preflights'].values():
            self.assertEqual(check['command'][0], str(physical / 'serving-vllm/bin/python'))
            self.assertEqual(check['configured_command'][0], str(self.root / 'envs/serving-vllm/bin/python'))
            self.assertNotEqual(check['server_identity_sha256'], check['configured_server_identity_sha256'])

    def prerequisite_fixture(self):
        import hashlib
        output = self.root / 'outputs/preparation/runtime-vllm'
        output.mkdir(parents=True)
        old_requirements = self.requirements.read_bytes().removesuffix(b'setuptools==77.0.3\n')
        prior_requirements = output / 'requirements-before.txt'
        prior_requirements.write_bytes(old_requirements)
        old_packages = {key: value for key, value in runtime.EXPECTED.items() if key != 'setuptools'}
        old = {'root': str(self.root), 'source_sha256': 'a' * 64, 'requirements_sha256': hashlib.sha256(old_requirements).hexdigest(), 'serving_config_sha256': 'c' * 64, 'environment': str(self.root / 'envs/serving-vllm'), 'expected_packages': old_packages}
        current = dict(old, source_sha256='d' * 64, requirements_sha256=hashlib.sha256(self.requirements.read_bytes()).hexdigest(), expected_packages=dict(runtime.EXPECTED))
        intent, receipt = output / 'intent.json', output / 'runtime.json'
        intent.write_text(json.dumps({'identity': old}))
        receipt.write_text(json.dumps({'status': 'complete', 'identity': old}))
        (output / 'build.log').write_bytes(b'cpu complete\n')
        (output / 'freeze.txt').write_bytes(b'vllm==0.8.5.post1\n')
        failure = self.root / 'outputs/serving-pools/v2'
        failure.mkdir(parents=True)
        (failure / 'state.json').write_text(json.dumps({'status': 'failed', 'cleanup': {'status': 'complete'}}))
        (failure / 'manifest.json').write_text(json.dumps({'runtime_receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()}))
        (failure / 'completion.json').write_text(json.dumps({'controller_state_sha256': hashlib.sha256((failure / 'state.json').read_bytes()).hexdigest(), 'compute_processes': []}))
        logs = self.root / 'outputs/serving/servers'
        logs.mkdir(parents=True)
        for rank in range(8):
            (logs / ('smoke-v2-gpu' + str(rank) + '.log')).write_text("ModuleNotFoundError: No module named 'setuptools'\n")
        return intent, receipt, current, hashlib.sha256(intent.read_bytes()).hexdigest(), prior_requirements, failure

    def test_explicit_prerequisite_transition_retains_cpu_success_and_actual_gpu_failure(self):
        intent, receipt, identity, digest, prior, failure = self.prerequisite_fixture()
        before = receipt.read_bytes()
        result = runtime.prerequisite_intent(intent, receipt, identity, digest, self.requirements, prior, failure)
        archive = Path(result['archive'])
        self.assertEqual((archive / 'runtime.json').read_bytes(), before)
        self.assertEqual(json.loads(before)['status'], 'complete')
        self.assertEqual(len(list(archive.glob('server-gpu-*.log'))), 8)
        self.assertEqual(json.loads((archive / 'expected-after.json').read_text()), runtime.EXPECTED)
        self.assertEqual(json.loads(intent.read_bytes())['identity'], identity)
        self.assertTrue((archive / 'requirements-before.txt').is_file())
        self.assertTrue((archive / 'requirements-after.txt').is_file())

    def test_prerequisite_transition_rejects_unrelated_changes_and_unproven_failure(self):
        intent, receipt, identity, digest, prior, failure = self.prerequisite_fixture()
        for changed in (dict(identity, serving_config_sha256='e' * 64), dict(identity, expected_packages=dict(runtime.EXPECTED, torch='2.7.0'))):
            with self.assertRaises(runtime.RuntimeError):
                runtime.prerequisite_intent(intent, receipt, changed, digest, self.requirements, prior, failure)
        path = next((self.root / 'outputs/serving/servers').glob('*.log'))
        path.write_text('some unrelated failure')
        with self.assertRaises(runtime.RuntimeError):
            runtime.prerequisite_intent(intent, receipt, identity, digest, self.requirements, prior, failure)
        self.assertFalse((intent.parent / 'prerequisite-transitions').exists())


if __name__ == '__main__':
    unittest.main()
