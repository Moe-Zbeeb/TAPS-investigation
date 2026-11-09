import copy
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from taps_rebuild import pool


def hardware(count=8, driver='580.95.05'):
    inventory = [{'index': str(i), 'uuid': 'GPU-fixture-' + str(i), 'name': 'NVIDIA A100', 'memory_total_mib': 81920.0, 'driver_version': driver} for i in range(count)]
    bindings = [{'shard_index': i, 'token': str(i), 'uuid': item['uuid'], 'index': str(i)} for i, item in enumerate(inventory)]
    return {'gpu_inventory': inventory, 'device_bindings': bindings, 'hardware_identity_sha256': pool.identity({'gpu_inventory': inventory, 'device_bindings': bindings}), 'job_id': '123', 'node': 'fixture-node', 'slurm': {'job': {'JobId': '123'}, 'node': {'NodeName': 'fixture-node'}}}


class Fixture:
    def __init__(self, path):
        self.root = Path(path).resolve()
        source = Path(pool.__file__).resolve().parent
        for file in source.rglob('*.py'):
            destination = self.root / 'src/taps_rebuild' / file.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, destination)
        self.legacy = self.root / 'legacy'
        for profile in ('eagle', 'hass'):
            python = self.legacy / 'envs' / ('taps-' + profile) / 'bin/python'
            python.parent.mkdir(parents=True)
            python.write_text('fixture interpreter')
            python.chmod(0o755)
        self.root.joinpath('configs').mkdir()
        self.root.joinpath('TAPS_rebuild_plan.pdf').write_bytes(b'synthetic test plan')
        methods = {name: {'source': 'HASS' if name == 'hass' else 'EAGLE', 'target': 'target', 'drafter': 'draft-' + name, 'profile': 'hass' if name == 'hass' else 'eagle', 'benchmarks': ['mt_bench']} for name in ('eagle3', 'eagle2', 'hass')}
        self.config = {'cluster_root': str(self.root), 'legacy_runtime_root': str(self.legacy), 'plan_sha256': pool.digest(self.root / 'TAPS_rebuild_plan.pdf'), 'scope': 'reference', 'method_order': list(methods), 'methods': methods, 'temperatures': [0, 1], 'timing_repeats': 5, 'warmup_generations_per_mode': 3, 'max_new_tokens': 512, 'max_length': 2048, 'base_seed': 17, 'bootstrap_samples': 16, 'relative_acceptance_tolerance': 0.05, 'reference_cells': [{'method': name, 'benchmark': 'mt_bench', 'temperature': temperature, 'published_tau': 3.0, 'source_url': 'https://example.test/synthetic-reference', 'expected_questions': 80, 'expected_turns': 2, 'timing_repeats': 5} for name in methods for temperature in (0, 1)]}
        self.config_path = self.root / 'configs/reproduction.json'
        self.config_path.write_text(json.dumps(self.config))
        preparation = self.root / 'outputs/preparation'
        preparation.mkdir(parents=True)
        for name in ('assets', 'sources'):
            registry = self.root / 'configs' / (name + '.json')
            registry.write_text(json.dumps({'fixture_registry': name}))
            receipt = {'status': 'complete', 'root': str(self.root), 'registry': str(registry), 'registry_sha256': pool.digest(registry), 'source_sha256': pool.digest(self.root / 'src/taps_rebuild' / (name + '.py'))}
            if name == 'assets':
                receipt['models'] = {identifier: {'status': 'verified'} for identifier in ['target', *['draft-' + method for method in methods]]}
            else:
                receipt.update(sources={'EAGLE': {}, 'HASS': {}}, benchmarks={'mt_bench': {'rows': 80, 'turns_per_row': 2, 'question_ids': list(range(80))}})
                self.sources = receipt
            (preparation / (name + '.json')).write_text(json.dumps(receipt))
        hashes = {str(path.relative_to(self.root)): pool.digest(path) for path in (self.root / 'src/taps_rebuild').rglob('*.py')}
        (self.root / 'source-identity.json').write_text(json.dumps({'git_commit': 'a' * 40, 'tracked_hashes': hashes, 'plan_sha256': self.config['plan_sha256']}))

    def freeze(self, count=8, smoke=True, run_id='fixture'):
        config = pool.effective_config(self.config, None, smoke, count)
        prepared = pool.prerequisites(self.root, self.config_path, config)
        run_dir = self.root / 'outputs/runs' / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = pool.freeze_run(self.root, run_dir, run_id, config, prepared, hardware(count))
        return run_dir, manifest, prepared


class CompletedProcess:
    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


def worker_factory(spawns, mismatch=False):
    def spawn(command, **kwargs):
        manifest_path = Path(command[command.index('--manifest') + 1])
        run_dir = Path(command[command.index('--run-dir') + 1])
        method = command[command.index('--method') + 1]
        shard = int(command[command.index('--shard-index') + 1])
        count = int(command[command.index('--shard-count') + 1])
        manifest = json.loads(manifest_path.read_text())
        spawns.append({'method': method, 'shard': shard, 'count': count, 'command': command, 'kwargs': kwargs})
        rows = []
        for turn in (0, 1):
            for mode in ('speculative', 'target_only'):
                tokens = [1, 2, 3]
                if mismatch and shard == 0 and turn == 0 and mode == 'target_only':
                    tokens = [1, 9, 3]
                emitted = [3] if mode == 'speculative' else [1, 1, 1]
                rows.append({'method': method, 'generation_mode': mode, 'benchmark': 'mt_bench', 'temperature': 0, 'question_id': shard, 'turn_id': turn, 'repeat': 0, 'seed': 17 + turn, 'cycles': [{'accepted_draft_tokens': value - 1, 'emitted_tokens': value, 'drafting_ms': 0.25, 'verification_ms': 1.0, 'verifier_nodes': value, 'proposed_nodes': value - 1} for value in emitted], 'new_token_count': 3, 'cycle_count': len(emitted), 'wall_ms': 10 if mode == 'speculative' else 20, 'output_token_ids': tokens, 'comparison_token_ids': tokens, 'warmup': False, 'case_id': f'{method}-{shard}-{turn}-{mode}', 'status': 'completed', 'run_id': manifest['run_id'], 'scope': manifest['scope'], 'scientific_identity_sha256': manifest['scientific_identity_sha256'], 'manifest_sha256': pool.digest(manifest_path), 'shard_index': shard, 'shard_count': count, 'unused_final_drafting_ms': 0})
        raw = run_dir / 'raw' / f'{method}-shard{shard}.jsonl'
        raw.parent.mkdir(exist_ok=True)
        content = '\n'.join(json.dumps(row) for row in rows) + '\n'
        if raw.exists():
            if raw.read_text() != content:
                raise AssertionError('Fixture resume would change raw measurements')
        else:
            raw.write_text(content)
        receipt = {'status': 'complete', 'method': method, 'shard_index': shard, 'shard_count': count, 'cuda_visible_devices': kwargs['env']['CUDA_VISIBLE_DEVICES'], 'scientific_identity_sha256': manifest['scientific_identity_sha256'], 'manifest_sha256': pool.digest(manifest_path), 'raw_path': str(raw), 'planned_cases': 4, 'completed_cases': 4}
        receipt_path = run_dir / 'workers' / f'{method}-{shard}.json'
        receipt_path.parent.mkdir(exist_ok=True)
        receipt_path.write_text(json.dumps(receipt))
        return CompletedProcess(90000 + len(spawns))
    return spawn


class PoolTests(unittest.TestCase):
    def command(self, calls, configured=8, allocated=8, uid=None):
        def run(arguments):
            calls.append(arguments)
            if arguments[:3] == ['scontrol', 'show', 'job']:
                return f'JobId=123 JobState=RUNNING UserId=fixture({os.getuid() if uid is None else uid}) NumNodes=1 NodeList=fixture-node TRES=cpu=64,gres/gpu={allocated},gres/gpu:a100={allocated}'
            if arguments[:3] == ['scontrol', 'show', 'node']:
                return f'NodeName=fixture-node CfgTRES=cpu=64,gres/gpu={configured},gres/gpu:a100={configured} Gres=gpu:a100:{configured}'
            if arguments[0] == 'nvidia-smi':
                return '\n'.join(f'{i}, GPU-fixture-{i}, NVIDIA A100, 81920, 580.95.05' for i in range(configured))
            raise AssertionError(arguments)
        return run

    def test_full_gpu_mapping_preserves_allocated_token_order(self):
        calls = []
        environment = {'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'fixture-node', 'CUDA_VISIBLE_DEVICES': '3,1,7,0,2,4,6,5'}
        value = pool.allocation(8, environment, self.command(calls))
        self.assertEqual([x['index'] for x in value['device_bindings']], ['3', '1', '7', '0', '2', '4', '6', '5'])
        self.assertEqual([call[0] for call in calls], ['scontrol', 'scontrol', 'nvidia-smi'])
        self.assertEqual(len({x['uuid'] for x in value['device_bindings']}), 8)

    def test_unset_cvd_is_inferred_only_after_full_node_proof(self):
        environment = {'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'fixture-node'}
        calls = []
        value = pool.allocation(8, environment, self.command(calls))
        self.assertEqual([x['token'] for x in value['device_bindings']], list(map(str, range(8))))
        calls = []
        with self.assertRaisesRegex(pool.PoolError, 'all configured GPUs'):
            pool.allocation(8, environment, self.command(calls, configured=9))
        self.assertNotIn('nvidia-smi', [call[0] for call in calls])

    def test_explicit_empty_cvd_never_infers_gpu_access(self):
        calls = []
        with self.assertRaisesRegex(pool.PoolError, 'Explicitly empty'):
            pool.allocation(8, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}, self.command(calls))
        self.assertEqual(calls, [])

    def test_foreign_allocation_and_duplicate_physical_aliases_rejected(self):
        environment = {'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'fixture-node', 'CUDA_VISIBLE_DEVICES': ','.join(map(str, range(8)))}
        with self.assertRaisesRegex(pool.PoolError, 'owned by this UID'):
            pool.allocation(8, environment, self.command([], uid=os.getuid() + 1))
        environment['CUDA_VISIBLE_DEVICES'] = '0,GPU-fixture-0,2,3,4,5,6,7'
        with self.assertRaisesRegex(pool.PoolError, 'same physical GPU'):
            pool.allocation(8, environment, self.command([]))

    def test_smoke_overrides_cover_all_eight_shards_and_filter_preserves_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            smoke = pool.effective_config(fixture.config, 'hass,eagle3', True, 8)
            self.assertEqual(smoke['method_order'], ['eagle3', 'hass'])
            self.assertEqual(smoke['temperatures'], [0])
            self.assertEqual((smoke['timing_repeats'], smoke['warmup_generations_per_mode'], smoke['max_new_tokens'], smoke['question_limit']), (1, 1, 32, 8))
            self.assertEqual({cell['method'] for cell in smoke['reference_cells']}, {'eagle3', 'hass'})
            self.assertEqual([pool._expected_cases('eagle3', i, 8, smoke, fixture.sources) for i in range(8)], [4] * 8)
            reference = pool.effective_config(fixture.config, None, False, 8)
            self.assertEqual(reference, fixture.config)

    def test_child_environment_is_offline_frozen_and_has_no_authentication_secrets(self):
        original = {'HF_TOKEN': 'fixture', 'GITHUB_TOKEN': 'fixture', 'AWS_SECRET_ACCESS_KEY': 'fixture', 'SSH_AUTH_SOCK': '/fixture', 'PATH': '/usr/bin', 'CUDA_VISIBLE_DEVICES': '0,1'}
        value = pool.worker_environment('7', Path('/run/inputs/source'), Path('/project'), Path('/legacy/envs/taps-eagle/bin/python'), original)
        self.assertEqual(value['CUDA_VISIBLE_DEVICES'], '7')
        self.assertEqual(value['PYTHONPATH'], '/run/inputs/source/src')
        self.assertEqual(value['HF_HUB_OFFLINE'], '1')
        self.assertEqual(value['HF_HOME'], '/project/cache/hf')
        self.assertFalse({'HF_TOKEN', 'GITHUB_TOKEN', 'AWS_SECRET_ACCESS_KEY', 'SSH_AUTH_SOCK'} & value.keys())
        self.assertEqual(original['HF_TOKEN'], 'fixture')

    def test_freeze_resume_keeps_manifest_bytes_and_uses_registry_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            run_dir, manifest, prepared = fixture.freeze()
            before = (run_dir / 'manifest.json').read_bytes()
            again = pool.freeze_run(fixture.root, run_dir, 'fixture', manifest['effective_config'], prepared, hardware())
            self.assertEqual(before, (run_dir / 'manifest.json').read_bytes())
            self.assertEqual(again, manifest)
            self.assertEqual(pool.digest(run_dir / 'inputs/config.json'), manifest['config_sha256'])
            self.assertEqual(Path(manifest['asset_registry_path']).read_bytes(), (fixture.root / 'configs/assets.json').read_bytes())
            self.assertEqual(Path(manifest['asset_receipt_path']).read_bytes(), (fixture.root / 'outputs/preparation/assets.json').read_bytes())

    def test_scientific_or_driver_mismatch_never_overwrites_existing_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            run_dir, manifest, prepared = fixture.freeze()
            before = (run_dir / 'manifest.json').read_bytes()
            changed = copy.deepcopy(manifest['effective_config'])
            changed['max_new_tokens'] = 64
            with self.assertRaisesRegex(pool.PoolError, 'Scientific identity differs'):
                pool.freeze_run(fixture.root, run_dir, 'fixture', changed, prepared, hardware())
            with self.assertRaisesRegex(pool.PoolError, 'Hardware inventory'):
                pool.freeze_run(fixture.root, run_dir, 'fixture', manifest['effective_config'], prepared, hardware(driver='581.0'))
            self.assertEqual(before, (run_dir / 'manifest.json').read_bytes())

    def test_deployed_identity_and_preparation_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            config = pool.effective_config(fixture.config, None, True, 8)
            (fixture.root / 'src/taps_rebuild/metrics.py').write_text('changed')
            with self.assertRaisesRegex(pool.PoolError, 'identity differs'):
                pool.prerequisites(fixture.root, fixture.config_path, config)

    def test_manifest_body_cannot_be_changed_behind_its_identity_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            run_dir, manifest, prepared = fixture.freeze()
            altered = copy.deepcopy(manifest)
            altered['effective_config']['max_new_tokens'] = 64
            (run_dir / 'manifest.json').write_text(json.dumps(altered))
            with self.assertRaisesRegex(pool.PoolError, 'scientific content was changed'):
                pool.freeze_run(fixture.root, run_dir, 'fixture', manifest['effective_config'], prepared, hardware())

    def test_every_gpu_is_started_once_per_method_even_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            spawns = []
            with patch.object(pool, 'allocation', return_value=hardware()), patch.object(pool.subprocess, 'Popen', side_effect=worker_factory(spawns)), patch.object(pool, 'terminate_children') as terminate:
                value = pool.run_pool(fixture.root, fixture.config_path, 'smoke', 8, smoke=True)
                self.assertEqual(value['status'], 'complete')
                self.assertTrue(value['smoke_validation_passed'])
                self.assertFalse(value['reproduction_gate_passed'])
                self.assertEqual(value['reproduction_gate_status'], 'ineligible')
                self.assertEqual(len(spawns), 24)
                for index, method in enumerate(fixture.config['method_order']):
                    wave = spawns[index * 8:(index + 1) * 8]
                    self.assertEqual([x['method'] for x in wave], [method] * 8)
                    self.assertEqual([x['kwargs']['env']['CUDA_VISIBLE_DEVICES'] for x in wave], list(map(str, range(8))))
                    self.assertTrue(all(x['kwargs']['start_new_session'] and x['kwargs']['pass_fds'] for x in wave))
                    self.assertTrue(all(x['kwargs']['env']['OMP_NUM_THREADS'] == x['kwargs']['env']['MKL_NUM_THREADS'] == '1' for x in wave))
                    self.assertTrue(all('/inputs/source/src' in x['kwargs']['env']['PYTHONPATH'] for x in wave))
                before = (fixture.root / 'outputs/runs/smoke/manifest.json').read_bytes()
                pool.run_pool(fixture.root, fixture.config_path, 'smoke', 8, smoke=True)
                self.assertEqual(len(spawns), 48)
                self.assertEqual(before, (fixture.root / 'outputs/runs/smoke/manifest.json').read_bytes())
                terminate.assert_not_called()

    def test_smoke_runtime_success_does_not_hide_t0_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            with patch.object(pool, 'allocation', return_value=hardware()), patch.object(pool.subprocess, 'Popen', side_effect=worker_factory([], mismatch=True)), patch.object(pool, 'terminate_children'):
                value = pool.run_pool(fixture.root, fixture.config_path, 'mismatch', 8, smoke=True, methods='eagle3')
            self.assertEqual(value['status'], 'complete')
            self.assertFalse(value['smoke_validation_passed'])
            self.assertFalse(value['reproduction_gate_passed'])

    def test_missing_completed_raw_cases_are_runtime_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            with patch.object(pool, 'allocation', return_value=hardware()), patch.object(pool.subprocess, 'Popen', side_effect=worker_factory([])), patch.object(pool, 'terminate_children'):
                pool.run_pool(fixture.root, fixture.config_path, 'corrupt', 8, smoke=True, methods='eagle3')
            run_dir = fixture.root / 'outputs/runs/corrupt'
            path = run_dir / 'raw/eagle3-shard0.jsonl'
            path.write_text('\n'.join(path.read_text().splitlines()[2:]) + '\n')
            manifest = json.loads((run_dir / 'manifest.json').read_text())
            with self.assertRaisesRegex(pool.PoolError, 'entire assigned panel'):
                pool.summarize_run(fixture.root, run_dir, manifest)

    def test_method_runtime_failure_prevents_later_waves(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            with patch.object(pool, 'allocation', return_value=hardware()), patch.object(pool, 'run_wave', side_effect=pool.PoolError('fixture worker failure')) as wave:
                with self.assertRaisesRegex(pool.PoolError, 'fixture worker failure'):
                    pool.run_pool(fixture.root, fixture.config_path, 'failed', 8, smoke=True)
            self.assertEqual(wave.call_count, 1)
            state = json.loads((fixture.root / 'outputs/runs/failed/pool.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertFalse(state['smoke_validation_passed'])
            self.assertFalse((fixture.root / 'outputs/runs/failed/report.json').exists())

    def test_run_lock_blocks_duplicate_controller_before_spawning(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            run_dir = fixture.root / 'outputs/runs/locked'
            run_dir.mkdir(parents=True)
            with (run_dir / 'pool.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(pool, 'allocation', return_value=hardware()), patch.object(pool.subprocess, 'Popen') as spawn:
                    with self.assertRaisesRegex(pool.PoolError, 'holds this run lock'):
                        pool.run_pool(fixture.root, fixture.config_path, 'locked', 8, smoke=True)
                    spawn.assert_not_called()

    def test_worker_failure_kills_own_sibling_and_grandchild_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            run_dir, manifest, prepared = fixture.freeze(count=2)
            marker = fixture.root / 'grandchild.pid'
            original_popen = subprocess.Popen
            children = []

            def launch(command, **kwargs):
                shard = int(command[command.index('--shard-index') + 1])
                code = 'import time;time.sleep(0.25);raise SystemExit(9)' if shard == 0 else 'import pathlib,subprocess,sys,time;p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);pathlib.Path(' + repr(str(marker)) + ').write_text(str(p.pid));time.sleep(60)'
                process = original_popen([sys.executable, '-B', '-c', code], **kwargs)
                children.append(process)
                return process

            with (run_dir / 'pool.lock').open('a+') as lock:
                with patch.object(pool.subprocess, 'Popen', side_effect=launch):
                    with self.assertRaisesRegex(pool.PoolError, 'Worker failed'):
                        pool.run_wave(fixture.root, run_dir, manifest, hardware(2), 'eagle3', prepared['receipts']['sources']['value'], lock.fileno())
            self.assertEqual(len(children), 2)
            self.assertTrue(all(child.poll() is not None for child in children))
            self.assertTrue(marker.is_file())
            pid = int(marker.read_text())
            deadline = time.monotonic() + 2
            alive = True
            while time.monotonic() < deadline:
                result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True, timeout=2)
                alive = bool(result.stdout.strip()) and not result.stdout.strip().startswith('Z')
                if not alive:
                    break
                time.sleep(0.02)
            self.assertFalse(alive, 'A worker grandchild survived process-group cleanup')


if __name__ == '__main__':
    unittest.main()
