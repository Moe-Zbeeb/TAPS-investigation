import hashlib
import base64
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from taps_rebuild import assets, native_runtime


class NativeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = (Path(self.temporary.name) / 'project').resolve()
        self.root.mkdir()
        self.legacy = self.root.parent / 'legacy'
        self.python = self.legacy / 'python/cpython-3.10/bin/python3.10'
        self.uv = self.legacy / 'tools/uv'
        for path in (self.python, self.uv):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('executable')
            path.chmod(0o700)
        (self.root / 'configs').mkdir()
        self.requirements = self.root / 'configs/runtime-native.txt'
        self.requirements.write_bytes((Path(__file__).parents[1] / 'configs/runtime-native.txt').read_bytes())
        self.config = json.loads((Path(__file__).parents[1] / 'configs/native-eagle3-training.json').read_bytes())
        (self.root / 'configs/native-eagle3-training.json').write_text(json.dumps(self.config))
        self.source = {'directory': str(self.root / 'repos/EAGLE/eagle/traineagle3'), 'revision': native_runtime.REVISION, 'files_sha256': {}, 'source_receipt_sha256': 'a' * 64}
        self.calls = []
        self.events = []
        self.hardware = {'job_id': '123', 'node': 'deep-chungus-10', 'device_bindings': [{'shard_index': index, 'token': str(index), 'uuid': 'GPU-' + str(index)} for index in range(8)], 'slurm': {'job': {'JobName': 'deep10-hold', 'Account': 'grad-students', 'Partition': 'high-priority', 'QOS': 'spec-priority'}}}
        self.real_gpu_guard = native_runtime.gpu_metadata_environment
        def guarded(root, environment):
            self.events.append('idle-guard')
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')
            result = dict(environment, CUDA_VISIBLE_DEVICES=','.join('GPU-' + str(index) for index in range(8)))
            return result, {'hardware': self.hardware, 'cuda_visible_devices': result['CUDA_VISIBLE_DEVICES'].split(',')}
        self.guard = patch.object(native_runtime, 'gpu_metadata_environment', side_effect=guarded)
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def value(self):
        return {'python': [3, 10, 17], 'prefix': str(self.root / 'envs/native-eagle3'), 'base_prefix': str(self.python.parent.parent), 'packages': dict(native_runtime.EXPECTED), 'torch_cuda': '12.4', 'cuda_context_initialized': True, 'gpu_checks': [{'logical_index': index, 'uuid_binding': 'GPU-' + str(index), 'name': 'Actual mocked A100', 'capability': [8, 0], 'total_memory_bytes': 80 * 1024**3, 'arithmetic_sum_squares': 1240.0} for index in range(8)], 'deepspeed_build_accelerator': 'cuda', 'deepspeed_runtime_accelerator': 'cuda', 'dataset_fixture_rows': 2, 'native_model_module_imported': 'cnets', 'deepspeed_installed_ops': {'fused_adam': False}}

    def runner(self, command, environment, log, timeout=7200, capture=False):
        self.calls.append(command)
        gpu = '--no-build-isolation' in command or ('-c' in command and command[0] == str(self.root / 'envs/native-eagle3/bin/python'))
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], ','.join('GPU-' + str(index) for index in range(8)) if gpu else '')
        if '--no-build-isolation' in command:
            self.events.append('deepspeed-build')
        elif '-c' in command and gpu:
            self.events.append('all-eight-arithmetic' if 'actual_all_eight_small_cuda_arithmetic_only' in command[-1] else 'import-probe')
        elif command[:3] == [str(self.uv), 'pip', 'check']:
            self.events.append('cpu-dependency-check')
        self.assertEqual(environment['NETRC'], '/dev/null')
        self.assertEqual(environment['DS_BUILD_OPS'], '0')
        self.assertEqual(environment['UV_HTTP_TIMEOUT'], '300')
        self.assertEqual(environment['UV_CONCURRENT_DOWNLOADS'], '4')
        self.assertEqual(environment['TAPS_RUNTIME_WORKDIR'], str(self.root))
        if command[:2] == [str(self.uv), 'venv']:
            path = self.root / 'envs/native-eagle3/bin/python'
            path.parent.mkdir(parents=True)
            path.write_text('virtualenv')
            path.chmod(0o700)
            (path.parent.parent / 'pyvenv.cfg').write_text('home = managed-python\n')
        if command[0] == str(self.python):
            return json.dumps({'python': [3, 10], 'prefix': str(self.python.parent.parent), 'base_prefix': str(self.python.parent.parent)})
        if command[:2] == [str(self.uv), '--version']:
            return 'uv 0.9.0\n'
        if command[:3] == [str(self.uv), 'pip', 'freeze']:
            return 'torch==2.6.0+cu124\ndeepspeed==0.17.1\n'
        if '-c' in command:
            return json.dumps(self.value())
        return '' if capture else None

    def platform(self):
        return patch.multiple(native_runtime.platform, system=lambda: 'Linux', machine=lambda: 'x86_64')

    def test_cpu_slurm_guard_runs_before_mutation(self):
        for environment in ({}, {'SLURM_JOB_ID': '1', 'CUDA_VISIBLE_DEVICES': '0'}):
            with patch.dict(os.environ, environment, clear=True), self.assertRaises(assets.PreparationError):
                native_runtime.prepare(self.root, self.python, self.uv, runner=self.runner, gpu_metadata=True)
        self.assertFalse((self.root / 'outputs').exists())
        self.assertFalse(self.calls)


    def test_default_mode_fails_before_mutation_or_deepspeed_import(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, clear=True):
            with self.assertRaisesRegex(native_runtime.NativeRuntimeError, '--gpu-metadata'):
                native_runtime.prepare(self.root, self.python, self.uv, runner=self.runner)
        self.assertFalse(self.calls)
        self.assertFalse((self.root / 'outputs').exists())

    def holder(self, job_id='123'):
        path = self.legacy / 'cluster/deep10/current.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'job_id': job_id, 'node': 'deep-chungus-10'}))
        return path

    def test_gpu_transition_verifies_holder_idle_nvml_and_all_uuid_bindings(self):
        self.holder()
        inspected = []
        def query(command):
            inspected.append(command)
            return '\n'.join('GPU-' + str(index) + ',0,0' for index in range(8)) if '--query-gpu=uuid,memory.used,utilization.gpu' in command else ''
        environment = native_runtime.environment_for(self.root, self.root / '.cache/task')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, clear=True), patch.object(native_runtime, 'LEGACY_ROOT', self.legacy), patch.object(native_runtime.pool, 'allocation', return_value=self.hardware) as allocated:
            result, proof = self.real_gpu_guard(self.root, environment, query=query)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], ','.join('GPU-' + str(index) for index in range(8)))
        self.assertNotIn('CUDA_VISIBLE_DEVICES', allocated.call_args.kwargs['environment'])
        self.assertEqual(allocated.call_args.args[0], 8)
        self.assertEqual(len(proof['idle_gpu_counters']), 8)
        self.assertEqual(proof['compute_processes'], [])
        for key in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR'):
            self.assertTrue(Path(result[key]).is_relative_to(self.root / '.cache/task'))

    def test_gpu_transition_rejects_busy_counters_processes_or_stale_holder(self):
        self.holder()
        environment = native_runtime.environment_for(self.root, self.root / '.cache/task')
        def idle():
            return '\n'.join('GPU-' + str(index) + ',0,0' for index in range(8))
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, clear=True), patch.object(native_runtime, 'LEGACY_ROOT', self.legacy), patch.object(native_runtime.pool, 'allocation', return_value=self.hardware):
            for memory, processes in [(idle().replace('GPU-7,0,0', 'GPU-7,32,0'), ''), (idle(), 'GPU-7,456,128')]:
                with self.subTest(processes=processes), self.assertRaises(native_runtime.NativeRuntimeError):
                    self.real_gpu_guard(self.root, environment, query=lambda command: memory if '--query-gpu=uuid,memory.used,utilization.gpu' in command else processes)
            self.holder(job_id='999')
            with self.assertRaisesRegex(native_runtime.NativeRuntimeError, 'holder allocation differs'):
                self.real_gpu_guard(self.root, environment, query=lambda command: self.fail('Idle GPU query must not bypass stale holder'))

    def test_busy_gpu_guard_prevents_any_deepspeed_build_or_import(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, clear=True), self.platform(), patch.object(native_runtime, 'LEGACY_ROOT', self.legacy), patch.object(native_runtime, 'source_record', return_value=self.source), patch.object(native_runtime, 'gpu_metadata_environment', side_effect=native_runtime.NativeRuntimeError('GPU remains busy')):
            with self.assertRaisesRegex(native_runtime.NativeRuntimeError, 'GPU remains busy'):
                native_runtime.prepare(self.root, self.python, self.uv, runner=self.runner, gpu_metadata=True)
        self.assertFalse(any('--no-build-isolation' in command for command in self.calls))
        self.assertFalse(any(command[0] == str(self.root / 'envs/native-eagle3/bin/python') and '-c' in command for command in self.calls))
        receipt = json.loads((self.root / 'outputs/preparation/runtime-native/runtime.json').read_bytes())
        self.assertEqual(receipt['status'], 'failed')
        self.assertFalse(receipt['gpu_execution_tested'])

    def test_environment_disables_netrc_home_config_and_cuda_builds_without_changing_home(self):
        with patch.dict(os.environ, {'HOME': '/correct/home', 'DS_BUILD_FUSED_ADAM': '1', 'UV_INDEX_URL': 'private', 'HF_TOKEN': 'secret'}, clear=True):
            value = native_runtime.environment_for(self.root, self.root / '.cache/task')
        self.assertEqual(value['HOME'], '/correct/home')
        self.assertEqual(value['NETRC'], '/dev/null')
        self.assertEqual(value['DS_BUILD_OPS'], '0')
        self.assertEqual(value['DS_ACCELERATOR'], 'cuda')
        self.assertEqual(value['UV_HTTP_TIMEOUT'], '300')
        self.assertEqual(value['UV_CONCURRENT_DOWNLOADS'], '4')
        self.assertNotIn('DS_BUILD_FUSED_ADAM', value)
        self.assertNotIn('HF_TOKEN', value)
        self.assertNotIn('UV_INDEX_URL', value)
        for name in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME', 'UV_CREDENTIALS_DIR'):
            self.assertTrue(value[name].startswith(str(self.root / '.cache/task')))
        selected = self.root / 'shared-uv-cache'
        value = native_runtime.environment_for(self.root, self.root / 'native-state', selected)
        self.assertEqual(value['UV_CACHE_DIR'], str(selected))
        self.assertEqual(value['TMPDIR'], str(self.root / 'native-state/tmp'))

    def test_bootstrap_installs_torch_before_nonisolated_deepspeed_source_build(self):
        contents = self.requirements.read_bytes()
        base = native_runtime.bootstrap(contents)
        self.assertNotIn(b'deepspeed @', base)
        self.assertIn(b'torch @', base)
        self.assertIn(b'setuptools==80.9.0', base)
        values = native_runtime.commands(self.uv, self.python, self.root / 'envs/native-eagle3', self.requirements, self.root / 'bootstrap.txt', False)
        self.assertEqual(values[0][:2], [str(self.uv), 'venv'])
        self.assertNotIn('--no-build-isolation', values[1])
        self.assertIn('--no-build-isolation', values[2])
        self.assertIn(native_runtime.EXCLUDE_NEWER, values[1])
        self.assertNotIn('sbatch', str(values))
        with self.assertRaises(native_runtime.NativeRuntimeError):
            native_runtime.bootstrap(base)

    def test_paths_require_managed_python_and_separate_native_environment(self):
        with patch.object(native_runtime, 'LEGACY_ROOT', self.legacy):
            paths = native_runtime.paths(self.root, self.python, self.uv, 'configs/runtime-native.txt', 'configs/native-eagle3-training.json')
            self.assertEqual(paths[5], self.root / 'envs/native-eagle3')
            self.config['runtime_python'] = 'envs/taps-eagle/bin/python'
            (self.root / 'configs/native-eagle3-training.json').write_text(json.dumps(self.config))
            with self.assertRaises(native_runtime.NativeRuntimeError):
                native_runtime.paths(self.root, self.python, self.uv, 'configs/runtime-native.txt', 'configs/native-eagle3-training.json')

    def test_source_metadata_and_actual_module_bytes_are_verified(self):
        source = self.root / 'repos/EAGLE'
        directory = source / 'eagle/traineagle3'
        directory.mkdir(parents=True)
        expected = {}
        for name in ('cnets.py', 'configs.py', 'modeling_llama_kv.py'):
            data = name.encode()
            (directory / name).write_bytes(data)
            expected[name] = hashlib.sha256(data).hexdigest()
        receipt = self.root / 'outputs/preparation/sources.json'
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({'status': 'complete', 'sources': {'EAGLE': {'path': str(source), 'revision': native_runtime.REVISION}}}))
        with patch.object(native_runtime, 'UPSTREAM', expected):
            result = native_runtime.source_record(self.root)
            self.assertEqual(result['files_sha256'], expected)
            (directory / 'cnets.py').write_text('changed')
            with self.assertRaises(native_runtime.NativeRuntimeError):
                native_runtime.source_record(self.root)

    def test_probe_requires_gpu_permission_and_validates_actual_all_eight_results(self):
        python = self.root / 'envs/native-eagle3/bin/python'
        environment = {'CUDA_VISIBLE_DEVICES': ','.join('GPU-' + str(index) for index in range(8))}
        python.parent.mkdir(parents=True)
        python.write_text('virtualenv')
        python.chmod(0o700)
        (python.parent.parent / 'pyvenv.cfg').write_text('home = managed-python\n')
        with self.assertRaisesRegex(native_runtime.NativeRuntimeError, '--gpu-metadata'):
            native_runtime.probe(python, environment, self.root / 'log', self.source, self.config, self.root / 'fixture.jsonl', lambda *args, **kwargs: self.fail('Probe must not execute without permission'))
        value = native_runtime.probe(python, environment, self.root / 'log', self.source, self.config, self.root / 'fixture.jsonl', lambda *args, **kwargs: json.dumps(self.value()), gpu_metadata=True)
        self.assertTrue(value['cuda_context_initialized'])
        for key, replacement in [('torch_cuda', '12.6'), ('cuda_context_initialized', False), ('deepspeed_build_accelerator', 'cpu')]:
            value = self.value()
            value[key] = replacement
            with self.assertRaises(native_runtime.NativeRuntimeError):
                native_runtime.probe(python, environment, self.root / 'log', self.source, self.config, self.root / 'fixture.jsonl', lambda *args, **kwargs: json.dumps(value), gpu_metadata=True)
        for key, replacement in [('capability', [9, 0]), ('arithmetic_sum_squares', 0.0), ('uuid_binding', 'GPU-0')]:
            value = self.value()
            value['gpu_checks'][7][key] = replacement
            with self.assertRaises(native_runtime.NativeRuntimeError):
                native_runtime.validate_gpu_checks(value, environment)

    def test_execute_uses_explicit_project_workdir_and_logs_only_command_output(self):
        with patch.object(native_runtime.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=b'public-output')) as run:
            output = native_runtime.execute(['python', '--version'], {'TAPS_RUNTIME_WORKDIR': str(self.root)}, self.root / 'build.log', capture=True)
        self.assertEqual(output, 'public-output')
        self.assertEqual(run.call_args.kwargs['cwd'], str(self.root))
        self.assertEqual((self.root / 'build.log').read_bytes(), b'public-output')

    def test_mocked_preparation_freezes_actual_environment_and_resumes_without_install(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, clear=True), self.platform(), patch.object(native_runtime, 'LEGACY_ROOT', self.legacy), patch.object(native_runtime, 'source_record', return_value=self.source), patch.object(native_runtime.shutil, 'which', return_value=None):
            path, receipt = native_runtime.prepare(self.root, self.python, self.uv, runner=self.runner, gpu_metadata=True)
            self.assertEqual(receipt['status'], 'complete')
            self.assertFalse(receipt['fused_adam_cuda_jit_tested'])
            self.assertFalse(receipt['optimizer_execution_tested'])
            self.assertTrue(receipt['gpu_execution_tested'])
            self.assertTrue(receipt['gpu_basic_arithmetic_tested'])
            self.assertNotIn('gpu_metadata', receipt['identity'])
            self.assertEqual(self.events, ['idle-guard', 'all-eight-arithmetic', 'idle-guard', 'deepspeed-build', 'idle-guard', 'import-probe', 'idle-guard', 'cpu-dependency-check'])
            self.assertEqual(receipt['toolchain']['nvcc']['available'], False)
            self.assertEqual(json.loads(path.read_bytes()), receipt)
            self.assertTrue(Path(receipt['freeze_path']).is_file())
            self.calls.clear()
            native_runtime.prepare(self.root, self.python, self.uv, runner=self.runner, gpu_metadata=True)
            self.assertFalse(any(command[:2] == [str(self.uv), 'venv'] or command[:3] == [str(self.uv), 'pip', 'install'] for command in self.calls))
            with patch.object(native_runtime, 'probe', side_effect=native_runtime.NativeRuntimeError('changed runtime')), self.assertRaises(native_runtime.NativeRuntimeError):
                native_runtime.prepare(self.root, self.python, self.uv, runner=self.runner, gpu_metadata=True)
            failed = json.loads(path.read_bytes())
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['failure']['recorded_by'], 'native_runtime')

    def test_existing_unknown_environment_and_changed_provenance_are_preserved(self):
        (self.root / 'envs/native-eagle3').mkdir(parents=True)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, clear=True), self.platform(), patch.object(native_runtime, 'LEGACY_ROOT', self.legacy), patch.object(native_runtime, 'source_record', return_value=self.source), self.assertRaisesRegex(native_runtime.NativeRuntimeError, 'unrecognized'):
            native_runtime.prepare(self.root, self.python, self.uv, runner=self.runner, gpu_metadata=True)
        self.assertFalse(self.calls)

    def failed_intent(self, source='a' * 64, status='failed'):
        output = self.root / 'retry'
        output.mkdir()
        old = {'source_sha256': source, 'root': str(self.root), 'requirements_sha256': 'b' * 64, 'environment': str(self.root / 'envs/native-eagle3'), 'expected_packages': dict(native_runtime.EXPECTED)}
        intent = output / 'intent.json'
        receipt = output / 'runtime.json'
        intent.write_bytes(json.dumps({'identity': old}, indent=3).encode() + b'\n')
        receipt.write_bytes(json.dumps({'identity': old, 'status': status, 'recovery_observation': {'reason': 'Observed failed launcher', 'build_log_sha256': 'c' * 64}}, indent=2).encode() + b'\n')
        (output / 'build.log').write_bytes(b'failed-download\n')
        return intent, receipt, old

    def test_explicit_retry_archives_exact_failed_attempt_and_only_changes_source_identity(self):
        intent, receipt, old = self.failed_intent()
        intent_bytes, receipt_bytes = intent.read_bytes(), receipt.read_bytes()
        intent_sha = hashlib.sha256(intent_bytes).hexdigest()
        new = dict(old, source_sha256='d' * 64)
        migration = native_runtime.retry_intent(intent, receipt, new, intent_sha)
        archive = Path(migration['archive'])
        self.assertEqual((archive / 'intent.json').read_bytes(), intent_bytes)
        self.assertEqual((archive / 'runtime.json').read_bytes(), receipt_bytes)
        self.assertEqual((archive / 'build.log').read_bytes(), b'failed-download\n')
        self.assertEqual(receipt.read_bytes(), receipt_bytes)
        self.assertEqual(json.loads(intent.read_bytes())['identity'], new)
        self.assertEqual(migration['prior_failure_record']['recovery_observation']['reason'], 'Observed failed launcher')
        self.assertEqual(migration['retained_files_sha256']['intent.json'], intent_sha)

    def test_retry_rejects_wrong_hash_nonfailed_attempt_or_changed_dependencies(self):
        intent, receipt, old = self.failed_intent(status='checking')
        original = intent.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        new = dict(old, source_sha256='d' * 64)
        with self.assertRaises(native_runtime.NativeRuntimeError):
            native_runtime.retry_intent(intent, receipt, new, digest)
        value = json.loads(receipt.read_bytes())
        value['status'] = 'failed'
        receipt.write_text(json.dumps(value))
        for candidate, claimed in ((new, '0' * 64), (dict(new, requirements_sha256='e' * 64), digest), (old, digest)):
            with self.assertRaises(native_runtime.NativeRuntimeError):
                native_runtime.retry_intent(intent, receipt, candidate, claimed)
        self.assertEqual(intent.read_bytes(), original)
        self.assertFalse((intent.parent / 'retries').exists())

    def test_prepare_retry_reuses_only_its_known_partial_environment_after_failed_migration(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, clear=True), self.platform(), patch.object(native_runtime, 'LEGACY_ROOT', self.legacy), patch.object(native_runtime, 'source_record', return_value=self.source), patch.object(native_runtime.shutil, 'which', return_value=None):
            with patch.object(native_runtime, 'probe', side_effect=native_runtime.NativeRuntimeError('download failed')), self.assertRaises(native_runtime.NativeRuntimeError):
                native_runtime.prepare(self.root, self.python, self.uv, runner=self.runner, gpu_metadata=True)
            output = self.root / 'outputs/preparation/runtime-native'
            intent_path, receipt_path = output / 'intent.json', output / 'runtime.json'
            old_intent = json.loads(intent_path.read_bytes())
            old_intent['identity']['source_sha256'] = 'a' * 64
            intent_path.write_text(json.dumps(old_intent))
            old_receipt = json.loads(receipt_path.read_bytes())
            old_receipt['identity'] = old_intent['identity']
            receipt_path.write_text(json.dumps(old_receipt))
            digest = hashlib.sha256(intent_path.read_bytes()).hexdigest()
            self.calls.clear()
            _, receipt = native_runtime.prepare(self.root, self.python, self.uv, runner=self.runner, retry_from=digest, gpu_metadata=True)
            self.assertEqual(receipt['status'], 'complete')
            self.assertFalse(any(command[:2] == [str(self.uv), 'venv'] for command in self.calls))
            self.assertTrue(any(command[:3] == [str(self.uv), 'pip', 'install'] for command in self.calls))
            self.assertEqual(len(list((output / 'retries').glob('*/migration.json'))), 1)


    def additive_attempt(self):
        intent, receipt, old = self.failed_intent()
        after = self.requirements.read_bytes()
        before = after[:-len(native_runtime.NVIDIA_ML_REQUIREMENT.encode())]
        old['expected_packages'].pop('nvidia-ml-py')
        old['requirements_sha256'] = hashlib.sha256(before).hexdigest()
        intent.write_text(json.dumps({'identity': old}))
        receipt.write_text(json.dumps({'identity': old, 'status': 'failed', 'stage': 'dependency-check'}))
        (intent.parent / 'build.log').write_bytes(b"The package `deepspeed` requires `nvidia-ml-py`, but it's not installed\n")
        metadata = b'Metadata-Version: 2.4\nName: deepspeed\nVersion: 0.17.1\nRequires-Dist: nvidia-ml-py\n'
        evidence = {'deepspeed_version': '0.17.1', 'requires_dist': ['nvidia-ml-py'], 'nvidia_ml_installed_version': None, 'metadata_base64': base64.b64encode(metadata).decode(), 'metadata_sha256': hashlib.sha256(metadata).hexdigest()}
        new = {**old, 'source_sha256': 'd' * 64, 'requirements_sha256': hashlib.sha256(after).hexdigest(), 'expected_packages': dict(native_runtime.EXPECTED)}
        return intent, receipt, old, new, before, after, evidence

    def test_explicit_additive_retry_archives_exact_requirement_and_declared_metadata_bytes(self):
        intent, receipt, old, new, before, after, evidence = self.additive_attempt()
        claimed = hashlib.sha256(intent.read_bytes()).hexdigest()
        with self.assertRaises(native_runtime.NativeRuntimeError):
            native_runtime.retry_intent(intent, receipt, new, claimed)
        migration = native_runtime.retry_intent(intent, receipt, new, claimed, add_nvidia_ml_py=True, requirements=self.requirements, dependency_evidence=evidence)
        archive = Path(migration['archive'])
        self.assertEqual((archive / 'requirements-before.txt').read_bytes(), before)
        self.assertEqual((archive / 'requirements-after.txt').read_bytes(), after)
        self.assertEqual((archive / 'installed-deepspeed-METADATA.txt').read_bytes(), base64.b64decode(evidence['metadata_base64']))
        self.assertEqual(migration['allowed_dependency_addition'], {'name': 'nvidia-ml-py', 'version': '12.575.51'})
        self.assertEqual(json.loads(intent.read_bytes())['identity'], new)
        self.assertTrue((archive / 'nvidia-primary-evidence.json').is_file())

    def test_additive_retry_rejects_any_other_pin_config_or_unproven_requirement(self):
        intent, receipt, old, new, before, after, evidence = self.additive_attempt()
        claimed = hashlib.sha256(intent.read_bytes()).hexdigest()
        for candidate, proof in [(dict(new, expected_packages=dict(new['expected_packages'], torch='other')), evidence), (dict(new, root='other'), evidence), (new, dict(evidence, requires_dist=[]))]:
            with self.subTest(candidate=candidate), self.assertRaises(native_runtime.NativeRuntimeError):
                native_runtime.retry_intent(intent, receipt, candidate, claimed, add_nvidia_ml_py=True, requirements=self.requirements, dependency_evidence=proof)
        changed = after.replace(b'torchvision-0.21.0', b'torchvision-0.22.0')
        self.requirements.write_bytes(changed)
        with self.assertRaisesRegex(native_runtime.NativeRuntimeError, 'previous requirement bytes'):
            native_runtime.retry_intent(intent, receipt, dict(new, requirements_sha256=hashlib.sha256(changed).hexdigest()), claimed, add_nvidia_ml_py=True, requirements=self.requirements, dependency_evidence=evidence)
        self.assertFalse((intent.parent / 'retries').exists())

    def test_physical_prefix_execution_keeps_venv_instead_of_resolving_binary_to_base(self):
        physical = self.root / 'physical-native'
        (physical / 'bin').mkdir(parents=True)
        (physical / 'pyvenv.cfg').write_text('home = managed-python\n')
        (physical / 'bin/python').symlink_to(self.python)
        alias = self.root / 'envs/native-eagle3'
        alias.parent.mkdir()
        alias.symlink_to(physical, target_is_directory=True)
        configured = alias / 'bin/python'
        actual = native_runtime.runtime.physical_python(configured)
        self.assertEqual(actual, physical / 'bin/python')
        self.assertNotEqual(actual, configured.resolve())
        command = [str(self.uv), 'pip', 'install', '--python', str(configured)]
        self.assertEqual(native_runtime.operational_command(command, alias)[-1], str(actual))
        self.assertEqual(native_runtime.operational_command([str(self.uv), 'venv', '--python', str(self.python)], alias)[-1], str(self.python))

    def test_explicit_operational_cache_uses_shared_uv_cas_with_separate_native_state(self):
        selected = self.root / 'shared-uv-cache'
        record = {'kind': 'node_local_tmp', 'node': 'node', 'path': str(selected), 'filesystem_type': 'zfs'}
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, clear=True), self.platform(), patch.object(native_runtime, 'LEGACY_ROOT', self.legacy), patch.object(native_runtime, 'source_record', return_value=self.source), patch.object(native_runtime.shutil, 'which', return_value=None), patch.object(native_runtime.runtime, 'cache_directory', return_value=(selected, record)) as guarded:
            _, receipt = native_runtime.prepare(self.root, self.python, self.uv, runner=self.runner, cache_dir=selected, gpu_metadata=True)
        guarded.assert_called_once_with(self.root, selected)
        self.assertEqual(receipt['operational_cache'], record)
        self.assertEqual(receipt['state_directory'], str(selected / 'native-state'))
        self.assertNotIn('operational_cache', receipt['identity'])
        self.assertTrue((selected / 'native-state/tmp').is_dir())


if __name__ == '__main__':
    unittest.main()
