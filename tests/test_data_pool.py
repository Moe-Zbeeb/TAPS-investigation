import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from taps_rebuild import data, data_pool, learning, pool


class DataPoolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / 'project'
        self.root.mkdir()
        source = self.root / 'src/taps_rebuild'
        source.mkdir(parents=True)
        real_source = Path(__file__).resolve().parents[1] / 'src/taps_rebuild'
        for name in data_pool.MODULES:
            (source / name).write_bytes((real_source / name).read_bytes())
        target = self.root / 'models/target'
        target.mkdir(parents=True)
        configuration = {'num_hidden_layers': 32, 'hidden_size': 4096, 'vocab_size': 128256, 'max_position_embeddings': 2048}
        (target / 'config.json').write_text(json.dumps(configuration))
        (target / 'model.safetensors').write_bytes(b'synthetic unit-test weight fixture')
        files = {name: {'size_bytes': (target / name).stat().st_size, 'sha256': pool.digest(target / name)} for name in ['config.json', 'model.safetensors']}
        asset = {'status': 'verified', 'path': str(target), 'files': files, 'source': {'repo_id': 'test-target', 'revision': 'a' * 40}, 'model_identity_sha256': hashlib.sha256(data_pool.native_training.canonical(files)).hexdigest()}
        self.assets = {'status': 'complete', 'root': str(self.root), 'models': {'llama31_instruct_8b': asset}}
        registry = {'runtime_python': sys.executable, 'assets': [{'id': 'llama31_instruct_8b', 'source': asset['source']}]}
        pool.atomic_json(self.root / 'configs/assets.json', registry)
        self.assets['registry_sha256'] = pool.digest(self.root / 'configs/assets.json')
        pool.atomic_json(self.root / 'outputs/preparation/assets.json', self.assets)
        self.runtime = {'prefix': sys.prefix, 'prefix_resolved': str(Path(sys.prefix).resolve()), 'versions': {'torch': '2.6.0+cu124', 'transformers': '4.53.1'}}
        self.hardware = {'job_id': '123', 'node': 'deep-chungus-4', 'device_bindings': [{'shard_index': index, 'token': str(index), 'uuid': 'GPU-' + str(index)} for index in range(8)], 'slurm': {'job': {'JobName': 'test'}}}

    def freeze(self, operation='features', **changes):
        config = {'operation': operation, **changes}
        directory = data_pool.output_directory(self.root, 'run')
        with patch.object(data_pool, 'runtime_identity', return_value=self.runtime):
            return data_pool.freeze(self.root, directory, 'run', config, data_pool.fixture_rows(), self.hardware)

    def make_operation(self, plan, assignment, operation):
        destination = Path(assignment['output_root']) / operation
        destination.mkdir(parents=True)
        input_path = assignment['input'] if operation == 'regenerate' or plan['operation'] == 'features' else str(Path(assignment['output_root']) / 'regenerate/records.jsonl')
        originals = data_pool.complete_records(input_path)
        records, entries = [], []
        for ordinal, original in enumerate(originals):
            if operation == 'regenerate':
                value = copy.deepcopy(original)
                value['messages'][-1]['content'] = 'Synthetic target response.'
                value['original_messages_sha256'] = data.identity(original['messages'])
                value['response_regime'] = 'target_regenerated_all_assistant_turns'
                value['target_model_identity_sha256'] = plan['target']['model_identity_sha256']
                value['generation'] = {'seed': int(data.identity([plan['config']['seed'], original['prompt_id'], plan['config']['target_id']])[:8], 16), 'temperature': plan['config']['temperature'], 'max_new_tokens': plan['config']['max_new_tokens'], 'max_context': plan['config']['max_context']}
                records.append(value)
                entries.append({'example_id': value['example_id'], 'prompt_id': value['prompt_id'], 'messages_sha256': data.identity(value['messages'])})
            else:
                path = destination / (str(ordinal).zfill(8) + '.pt')
                path.write_bytes(('unit tensor fixture ' + str(assignment['shard_index']) + ' ' + str(ordinal)).encode())
                item = {'path': str(path), 'sha256': pool.digest(path), **{key: original[key] for key in ['example_id', 'prompt_id', 'domain', 'split', 'response_regime']}, 'tokens': 3, 'loss_tokens': 2, 'messages_sha256': data.identity(original['messages'])}
                entries.append(item)
                records.append(item)
        data_pool.write_records(destination / 'records.jsonl', records)
        manifest = {'status': 'complete', 'command': operation, 'source_sha256': plan['source_hashes']['data.py'], 'input_sha256': pool.digest(input_path), 'asset_receipt_sha256': plan['asset_receipt_sha256'], 'target_model_identity_sha256': plan['target']['model_identity_sha256'], 'target_model_id': plan['config']['target_id'], 'processing_config': {'root': plan['runtime_root'], 'command': operation, 'input': input_path, 'output': str(destination), 'target_id': plan['config']['target_id'], 'limit': len(assignment['example_ids']), 'max_context': plan['config']['max_context']}, 'feature_layers': plan['feature_layers'], 'alignment': data_pool.ALIGNMENT, 'include_teacher_logits': plan['config']['include_teacher_logits'], 'examples': entries}
        pool.atomic_json(destination / 'manifest.json', manifest)
        return data_pool.validate_operation(plan, assignment, operation)

    def make_workers(self, plan):
        for assignment in plan['assignments']:
            operations = {}
            for operation in ('regenerate', 'features') if plan['operation'] == 'regenerate-features' else (plan['operation'],):
                result = self.make_operation(plan, assignment, operation)
                operations[operation] = {key: value for key, value in result.items() if key not in ('records', 'examples')}
            receipt = {'status': 'complete', 'shard_index': assignment['shard_index'], 'example_ids': assignment['example_ids'], 'plan_identity_sha256': plan['plan_identity_sha256'], 'physical_gpu_uuid': assignment['binding']['uuid'], 'cuda_visible_devices': assignment['binding']['token'], 'slurm_job_id': '123', 'package_versions': self.runtime['versions'], 'operations': operations}
            pool.atomic_json(Path(plan['run_dir']) / 'workers' / (str(assignment['shard_index']) + '.json'), receipt)


    def test_runtime_probe_has_bounded_cold_filesystem_timeout_and_isolated_environment(self):
        environment = self.root / 'venv'
        environment.mkdir()
        python = environment / 'bin/python'
        result = {'prefix': str(environment), 'versions': {'torch': '2.6.0+cu124', 'transformers': '4.53.1'}}
        with patch.dict(os.environ, {'PYTHONPATH': '/irrelevant/source', 'PYTHONHOME': '/irrelevant/python', 'PYTHONNOUSERSITE': '0', 'CUDA_VISIBLE_DEVICES': '0,1'}), patch.object(data_pool.subprocess, 'check_output', return_value=json.dumps(result)) as probe:
            actual = data_pool.runtime_identity(python)
        command = probe.call_args.args[0]
        settings = probe.call_args.kwargs
        self.assertEqual(command[:3], [str(python), '-B', '-c'])
        self.assertEqual(settings['timeout'], 300)
        self.assertEqual(settings['cwd'], environment)
        self.assertEqual(settings['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(settings['env']['PYTHONNOUSERSITE'], '1')
        self.assertEqual(settings['env']['PYTHONDONTWRITEBYTECODE'], '1')
        self.assertNotIn('PYTHONPATH', settings['env'])
        self.assertNotIn('PYTHONHOME', settings['env'])
        self.assertEqual(actual['versions'], result['versions'])
        self.assertEqual(actual['prefix_resolved'], str(environment.resolve()))

    def test_uneven_last_shard_preserves_every_id_and_split(self):
        rows = [{'example_id': str(index), 'split': ['train', 'dev', 'test'][index % 3]} for index in range(17)]
        shards = data_pool.shard_rows(rows)
        self.assertEqual([ordinal for ordinal, _ in shards[7]], [7, 15])
        self.assertEqual([ordinal for ordinal, _ in shards[0]], [0, 8, 16])
        recovered = sorted((ordinal, row) for shard in shards for ordinal, row in shard)
        self.assertEqual([row for _, row in recovered], rows)
        with self.assertRaisesRegex(data_pool.DataPoolError, 'eight'):
            data_pool.shard_rows(rows[:7])

    def test_freeze_uses_unchanged_receipt_and_actual_target_layers(self):
        plan = self.freeze()
        self.assertEqual(plan['feature_layers'], [2, 16, 29])
        self.assertEqual(plan['feature_dim'], 12288)
        self.assertEqual(Path(plan['runtime_root'], 'outputs/preparation/assets.json').read_bytes(), (self.root / 'outputs/preparation/assets.json').read_bytes())
        self.assertEqual(plan['asset_receipt_original_root'], str(self.root))
        self.assertEqual(data_pool.load_plan(Path(plan['run_dir']) / 'plan.json')['plan_identity_sha256'], plan['plan_identity_sha256'])
        for assignment in plan['assignments']:
            output = Path(assignment['output_root']) / 'features'
            self.assertEqual(data.output_path(Path(plan['runtime_root']), output), output)

    def test_all_shards_use_same_base_seed_and_exact_one_gpu_assignment(self):
        plan = self.freeze('regenerate')
        ids = []
        for assignment in plan['assignments']:
            command = data_pool.command_for(plan, assignment, 'regenerate')
            self.assertEqual(command[command.index('--seed') + 1], '20261109')
            self.assertEqual(command[command.index('--limit') + 1], '1')
            ids += assignment['example_ids']
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual({item['binding']['token'] for item in plan['assignments']}, set(map(str, range(8))))

    def test_merged_feature_manifest_passes_actual_learning_inspector(self):
        plan = self.freeze()
        self.make_workers(plan)
        result = data_pool.merge(plan)
        manifest = Path(plan['run_dir']) / 'manifest.json'
        inspected = learning.inspect_feature_manifest(self.root, manifest, split='train')
        self.assertEqual(inspected['exported_examples'], 8)
        self.assertEqual(len(inspected['examples']), 3)
        self.assertEqual(result['actual_exported_tokens'], 24)
        self.assertEqual(result['actual_exported_loss_tokens'], 16)
        self.assertEqual([entry['global_ordinal'] for entry in result['examples']], list(range(8)))
        self.assertTrue(all(Path(entry['path']).is_relative_to(manifest.parent) for entry in result['examples']))

    def test_pipeline_preserves_all_splits_and_merges_regenerated_text_in_input_order(self):
        plan = self.freeze('regenerate-features', merge_regenerated=True)
        self.make_workers(plan)
        result = data_pool.merge(plan)
        rows = data_pool.complete_records(result['merged_regenerated_canonical']['path'])
        self.assertEqual([row['example_id'] for row in rows], [row['example_id'] for row in data_pool.fixture_rows()])
        self.assertEqual([row['split'] for row in rows], [row['split'] for row in data_pool.fixture_rows()])
        self.assertTrue(all(row['response_regime'] == 'target_regenerated_all_assistant_turns' for row in rows))
        self.assertTrue(all(entry['response_regime'] == 'target_regenerated_all_assistant_turns' for entry in result['examples']))

    def test_output_path_traversal_and_child_symlinks_are_rejected(self):
        for identifier in ['../escape', '/absolute', '..', 'a/b']:
            with self.subTest(identifier=identifier), self.assertRaises(data_pool.DataPoolError):
                data_pool.output_directory(self.root, identifier)
        outside = self.root / 'outside'
        outside.mkdir()
        (self.root / 'data').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(data_pool.DataPoolError):
            data_pool.output_directory(self.root, 'run')

    def test_feature_index_cannot_escape_export_directory(self):
        plan = self.freeze()
        assignment = plan['assignments'][0]
        self.make_operation(plan, assignment, 'features')
        destination = Path(assignment['output_root']) / 'features'
        path = self.root / 'outside.pt'
        path.write_bytes(b'unrelated')
        manifest = json.loads((destination / 'manifest.json').read_bytes())
        manifest['examples'][0].update(path=str(path), sha256=pool.digest(path))
        (destination / 'records.jsonl').unlink()
        data_pool.write_records(destination / 'records.jsonl', manifest['examples'])
        pool.atomic_json(destination / 'manifest.json', manifest)
        with self.assertRaisesRegex(data_pool.DataPoolError, 'escapes'):
            data_pool.validate_operation(plan, assignment, 'features')
        inside = destination / '00000000.pt'
        with self.assertRaisesRegex(data_pool.DataPoolError, 'escapes'):
            data_pool.bounded_path(str(destination / '../features/00000000.pt'), destination)

    def test_source_and_completed_output_tampering_are_rejected(self):
        plan = self.freeze()
        data_path = Path(plan['run_dir']) / 'inputs/source/src/taps_rebuild/data.py'
        data_path.write_text('changed')
        with self.assertRaisesRegex(data_pool.DataPoolError, 'Frozen data input/source'):
            data_pool.load_plan(Path(plan['run_dir']) / 'plan.json')

    def test_changed_regeneration_seed_or_split_cannot_be_merged(self):
        plan = self.freeze('regenerate')
        assignment = plan['assignments'][0]
        self.make_operation(plan, assignment, 'regenerate')
        destination = Path(assignment['output_root']) / 'regenerate'
        rows = data_pool.complete_records(destination / 'records.jsonl')
        rows[0]['generation']['seed'] += 1
        (destination / 'records.jsonl').unlink()
        data_pool.write_records(destination / 'records.jsonl', rows)
        with self.assertRaisesRegex(data_pool.DataPoolError, 'deterministic seed'):
            data_pool.validate_operation(plan, assignment, 'regenerate')

    def test_unterminated_completed_output_never_passes_validation(self):
        plan = self.freeze()
        assignment = plan['assignments'][0]
        self.make_operation(plan, assignment, 'features')
        path = Path(assignment['output_root']) / 'features/records.jsonl'
        path.write_bytes(path.read_bytes().rstrip(b'\n'))
        with self.assertRaisesRegex(data_pool.DataPoolError, 'terminated'):
            data_pool.validate_operation(plan, assignment, 'features')

    def test_failed_wave_preserves_partial_artifacts_and_refuses_training_admission(self):
        def failed(plan, lock):
            destination = Path(plan['assignments'][0]['output_root']) / 'features'
            destination.mkdir(parents=True)
            pool.atomic_json(destination / 'manifest.json', {'status': 'failed', 'completed_examples': []})
            raise data_pool.DataPoolError('injected worker failure')
        with patch.object(data_pool.pool, 'allocation', return_value=self.hardware), patch.object(data_pool, 'runtime_identity', return_value=self.runtime), patch.object(data_pool, 'run_wave', side_effect=failed):
            with self.assertRaisesRegex(data_pool.DataPoolError, 'worker failure'):
                data_pool.controller(self.root, 'run', {'operation': 'features'})
        directory = self.root / 'data/exports/run'
        manifest = json.loads((directory / 'manifest.json').read_bytes())
        self.assertEqual(manifest['status'], 'failed')
        self.assertTrue(manifest['partial_outputs_preserved'])
        self.assertTrue((directory / 'runtime-view/data/shards/0/features/manifest.json').exists())
        with self.assertRaisesRegex(learning.LearningError, 'Only completed'):
            learning.inspect_feature_manifest(self.root, directory, split='train')

    def test_research_missing_gate_and_fake_engineering_data_fail_before_gpu_allocation(self):
        with patch.object(data_pool.pool, 'allocation') as allocation:
            with self.assertRaisesRegex(data_pool.DataPoolError, 'gate evidence'):
                data_pool.controller(self.root, 'run', {'operation': 'regenerate', 'scope': 'research', 'input': 'real.jsonl'})
            allocation.assert_not_called()
        with self.assertRaisesRegex(data_pool.DataPoolError, 'Engineering is restricted'):
            data_pool.options({'operation': 'features', 'input': 'real.jsonl'})


    def test_repointed_model_alias_is_rejected_before_worker_load(self):
        target = self.root / 'models/target'
        alias = self.root / 'models/alias'
        alias.symlink_to(target, target_is_directory=True)
        self.assets['models']['llama31_instruct_8b']['path'] = str(alias)
        pool.atomic_json(self.root / 'outputs/preparation/assets.json', self.assets)
        plan = self.freeze()
        other = self.root / 'models/other'
        other.mkdir()
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(data_pool.DataPoolError, 'target alias changed'):
            data_pool.load_plan(Path(plan['run_dir']) / 'plan.json')

    def test_worker_output_hash_receipt_is_binding_at_merge(self):
        plan = self.freeze()
        self.make_workers(plan)
        path = Path(plan['run_dir']) / 'workers/0.json'
        receipt = json.loads(path.read_bytes())
        receipt['operations']['features']['manifest_sha256'] = '0' * 64
        pool.atomic_json(path, receipt)
        with self.assertRaisesRegex(data_pool.DataPoolError, 'output hashes changed'):
            data_pool.merge(plan)
        self.assertNotEqual(json.loads((Path(plan['run_dir']) / 'manifest.json').read_bytes())['status'], 'complete')

    def test_worker_checks_inherited_lock_and_revalidates_cli_outputs(self):
        plan = self.freeze()
        directory = Path(plan['run_dir'])
        assignment = plan['assignments'][0]
        calls = []
        def run(command, **kwargs):
            calls.append((command, kwargs))
            self.make_operation(plan, assignment, 'features')
        with (directory / 'controller.lock').open('a+') as lock, patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': '0'}), patch.object(data_pool.importlib.metadata, 'version', side_effect=lambda name: self.runtime['versions'][name]), patch.object(data_pool.subprocess, 'run', side_effect=run):
            data_pool.worker(directory / 'plan.json', 0, lock.fileno())
            self.assertEqual(calls[0][1]['pass_fds'], (lock.fileno(),))
        receipt = json.loads((directory / 'workers/0.json').read_bytes())
        self.assertEqual(receipt['status'], 'complete')
        self.assertEqual(receipt['example_ids'], assignment['example_ids'])
        self.assertIn('features', receipt['operations'])


if __name__ == '__main__':
    unittest.main()
