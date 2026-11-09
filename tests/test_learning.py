import ast
import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from taps_rebuild import learning


def configuration():
    return {'scope': 'engineering', 'experiment': 'expert-tests', 'run_id': 'tiny', 'seed': 37, 'model': {'feature_dim': 4, 'vocab_size': 16, 'hidden_size': 8, 'num_attention_heads': 2, 'num_hidden_layers': 1, 'num_experts': 2, 'lora_rank': 2, 'max_sequence_length': 8}, 'data': {'synthetic': True, 'examples': 3, 'sequence_length': 4, 'include_teacher_logits': True}, 'loss': {'objective': 'hard_ce', 'feature_mse_weight': 0.1, 'domain_gate_weight': 0.1}, 'training': {'optimizer_steps': 2, 'batch_size': 1, 'gradient_accumulation_steps': 2, 'checkpoint_every_steps': 1}, 'budget': {'optimizer_steps': 2, 'loss_tokens': 16, 'nonpadding_tokens': 16, 'input_slots': 16}}


class LearningMetadataTests(unittest.TestCase):
    def test_config_defaults_and_engineering_recipe(self):
        config = learning.normalize_config(configuration())
        self.assertEqual(config['model']['expert_mode'], 'separate_heads')
        self.assertEqual(config['loss']['feature_mse_target'], 'next_features')
        path = Path(__file__).resolve().parents[1] / 'configs/engineering-training.json'
        default = learning.normalize_config(json.loads(path.read_text()))
        expected = default['training']['optimizer_steps'] * default['training']['batch_size'] * default['training']['gradient_accumulation_steps'] * default['data']['sequence_length']
        self.assertEqual(default['budget']['loss_tokens'], expected)
        self.assertEqual(default['scope'], 'engineering')

    def test_invalid_model_and_data_contracts(self):
        for field, value in [('num_hidden_layers', 3), ('hidden_size', 7), ('lora_rank', 17), ('expert_mode', 'native_eagle3'), ('dropout', 1.0)]:
            with self.subTest(field=field):
                config = configuration()
                config['model'][field] = value
                with self.assertRaises(learning.LearningError):
                    learning.normalize_config(config)
        config = configuration()
        config['scope'] = 'research'
        with self.assertRaises(learning.LearningError):
            learning.normalize_config(config)
        config['data'] = {'path': 'features.pt'}
        config['loss']['feature_mse_target'] = 'input_reconstruction'
        with self.assertRaises(learning.LearningError):
            learning.normalize_config(config)
        config['loss']['feature_mse_target'] = 'next_features'
        self.assertEqual(learning.normalize_config(config)['data']['split'], 'train')

    def test_torch_import_is_lazy_and_native_recipe_not_claimed(self):
        tree = ast.parse(Path(learning.__file__).read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(value.name == 'torch' for value in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, 'torch')
        self.assertFalse(learning.capabilities()['native_official_eagle3_retraining'])
        self.assertFalse(learning.capabilities()['benchmark_scores_produced'])

    def test_budget_eligibility_requires_observed_exact_tokens_and_steps(self):
        observed = {'loss_tokens': 16, 'optimizer_steps': 2, 'nonpadding_tokens': 20, 'input_slots': 24}
        self.assertTrue(learning.budget_eligibility(observed, {'loss_tokens': 16, 'optimizer_steps': 2})['eligible'])
        for declared in ({}, {'loss_tokens': 16}, {'loss_tokens': 17, 'optimizer_steps': 2}, {'loss_tokens': 16, 'optimizer_steps': 2, 'input_slots': 20}):
            self.assertFalse(learning.budget_eligibility(observed, declared)['eligible'])
        left = {'counters': observed, 'declared_budget': {'loss_tokens': 16, 'optimizer_steps': 2}}
        self.assertTrue(learning.compare_budgets(left, left)['eligible'])
        right = {'counters': {**observed, 'loss_tokens': 8}, 'declared_budget': {'loss_tokens': 8, 'optimizer_steps': 2}}
        self.assertFalse(learning.compare_budgets(left, right)['eligible'])

    def test_gpu_guard_and_explicit_injected_cpu(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(learning._device('cpu', True), 'cpu')
            with self.assertRaises(learning.LearningError):
                learning._device('cpu', False)
            with self.assertRaises(learning.LearningError):
                learning._device(None, False)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '1', 'CUDA_VISIBLE_DEVICES': '0,1'}, clear=True):
            with self.assertRaises(learning.LearningError):
                learning._device(None, False)

    def test_trained_directory_allows_models_storage_alias_but_rejects_child_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = configuration()
            path = learning.trained_directory(root, config)
            self.assertTrue(path.is_relative_to(root.resolve() / 'models/trained'))
            config['run_id'] = '../input'
            with self.assertRaises(learning.LearningError):
                learning.trained_directory(root, config)
            config['run_id'] = 'tiny'
            storage = root / 'foreign'
            storage.mkdir()
            (storage / 'pretrained.pt').write_bytes(b'preserved input weights')
            (root / 'models').symlink_to(storage)
            path = learning.trained_directory(root, config)
            self.assertEqual(path, storage.resolve() / 'trained/expert-tests/tiny')
            self.assertEqual((storage / 'pretrained.pt').read_bytes(), b'preserved input weights')
            (storage / 'trained').symlink_to(root / 'original-model')
            with self.assertRaises(learning.LearningError):
                learning.trained_directory(root, config)

    def test_checkpoint_sha_identity_and_path_guard_without_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'checkpoint-step-000001.pt'
            checkpoint.write_bytes(b'weights fixture')
            metadata = {'config_sha256': 'config', 'data_identity_sha256': 'data'}
            receipt = {'status': 'committed', 'checkpoint_sha256': learning.file_sha256(checkpoint), 'metadata': metadata}
            checkpoint.with_suffix('.json').write_text(json.dumps(receipt))
            self.assertEqual(learning.verify_checkpoint(checkpoint, root, metadata)['metadata'], metadata)
            with self.assertRaises(learning.LearningError):
                learning.verify_checkpoint(checkpoint, root, {'config_sha256': 'changed'})
            with self.assertRaises(learning.LearningError):
                learning.verify_checkpoint(checkpoint, root / 'other', metadata)
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(learning.LearningError):
                learning.verify_checkpoint(checkpoint, root, metadata)

    def _manifest(self, root):
        entries = []
        for split in ('train', 'validation'):
            path = root / (split + '.pt')
            path.write_bytes(split.encode())
            entries.append({'path': str(path), 'sha256': learning.file_sha256(path), 'split': split, 'tokens': 4, 'loss_tokens': 4})
        path = root / 'manifest.json'
        path.write_text(json.dumps({'status': 'complete', 'command': 'features', 'alignment': 'explicit next-feature alignment', 'examples': entries}))
        return path

    def test_completed_feature_manifest_directory_and_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._manifest(root)
            result = learning.inspect_feature_manifest(root, root)
            self.assertEqual(result['path'], str(path.resolve()))
            self.assertEqual(result['exported_examples'], 2)
            self.assertEqual(len(result['examples']), 1)
            self.assertEqual(result['examples'][0]['split'], 'train')
            other = learning.inspect_feature_manifest(root, path, 'validation')
            self.assertEqual(other['examples'][0]['split'], 'validation')
            with self.assertRaises(learning.LearningError):
                learning.inspect_feature_manifest(root, path, 'test')

    def test_feature_manifest_rejects_partial_sha_duplicate_and_escape(self):
        for failure in ('partial', 'sha', 'duplicate', 'escape'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cache = root / 'cache'
                cache.mkdir()
                path = self._manifest(cache)
                manifest = json.loads(path.read_text())
                if failure == 'partial':
                    manifest['status'] = 'failed'
                elif failure == 'sha':
                    (cache / 'validation.pt').write_bytes(b'corrupt heldout record')
                elif failure == 'duplicate':
                    manifest['examples'].append(manifest['examples'][0])
                else:
                    outside = root / 'outside.pt'
                    outside.write_bytes(b'outside')
                    manifest['examples'][0].update(path=str(outside), sha256=learning.file_sha256(outside))
                path.write_text(json.dumps(manifest))
                with self.assertRaises(learning.LearningError):
                    learning.inspect_feature_manifest(root, path)

    def test_counter_invariants(self):
        config = learning.normalize_config(configuration())
        good = {'optimizer_steps': 1, 'micro_batches': 2, 'loss_tokens': 8, 'nonpadding_tokens': 8, 'input_slots': 8, 'examples_consumed': 2}
        learning._counters_valid(good, config)
        for name, value in [('micro_batches', 1), ('examples_consumed', 1), ('loss_tokens', 9), ('optimizer_steps', 3)]:
            with self.subTest(name=name), self.assertRaises(learning.LearningError):
                learning._counters_valid({**good, name: value}, config)

    def test_orphan_checkpoint_quarantine_requires_run_identity_and_keeps_committed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            expected = {'config_sha256': 'config', 'data_identity_sha256': 'data'}
            (root / 'run.json').write_text(json.dumps({'identity': expected}))
            orphan = root / 'checkpoint-step-000002.pt'
            orphan.write_bytes(b'interrupted checkpoint bytes')
            committed = root / 'checkpoint-step-000001.pt'
            committed.write_bytes(b'committed checkpoint bytes')
            receipt = committed.with_suffix('.json')
            receipt.write_text(json.dumps({'status': 'committed'}))
            with (root / 'training.lock').open('a+') as lock:
                recovered = learning.recover_uncommitted_checkpoints(root, expected, lock)
            self.assertEqual(len(recovered), 1)
            item = recovered[0]
            self.assertEqual(item['status'], 'quarantined')
            self.assertFalse(orphan.exists())
            quarantined = Path(item['quarantined_path'])
            self.assertEqual(quarantined.read_bytes(), b'interrupted checkpoint bytes')
            self.assertEqual(learning.file_sha256(quarantined), item['sha256'])
            self.assertTrue(quarantined.is_relative_to(root / 'recovery'))
            self.assertEqual(committed.read_bytes(), b'committed checkpoint bytes')
            self.assertEqual(json.loads(receipt.read_text()), {'status': 'committed'})
            with (root / 'training.lock').open('a+') as lock, self.assertRaises(learning.LearningError):
                learning.recover_uncommitted_checkpoints(root, {'config_sha256': 'changed'}, lock)

    def test_history_torn_tail_and_uncommitted_rows_are_preserved_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            history = root / 'training.jsonl'
            first = b'{ "counters": {"optimizer_steps": 1} }\n'
            second = b'{"counters":{"optimizer_steps":2}}\n'
            tail = b'{"counters":{"optimizer_steps":3'
            original = first + second + tail
            history.write_bytes(original)
            recovery = Path(learning._history(history, 1))
            self.assertEqual(history.read_bytes(), first)
            self.assertEqual((recovery / 'original.jsonl').read_bytes(), original)
            self.assertEqual((recovery / 'trailing-tail.bin').read_bytes(), tail)
            receipt = json.loads((recovery / 'receipt.json').read_text())
            self.assertEqual(receipt['original_sha256'], learning.file_sha256(recovery / 'original.jsonl'))
            self.assertEqual(receipt['trailing_tail_bytes'], len(tail))

    def test_history_malformed_complete_row_fails_without_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            history = root / 'training.jsonl'
            original = b'{"counters":{"optimizer_steps":1}}\n{broken full row}\npartial'
            history.write_bytes(original)
            with self.assertRaises(learning.LearningError):
                learning._history(history, 1)
            self.assertEqual(history.read_bytes(), original)
            self.assertFalse((root / 'recovery').exists())


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Torch absent locally; tensor checks run in the existing GPU environment')
class LearningTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = learning._torch()
        cls.old_threads = cls.torch.get_num_threads()
        cls.old_deterministic = cls.torch.are_deterministic_algorithms_enabled()
        cls.torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)
        cls.torch.use_deterministic_algorithms(cls.old_deterministic)

    def batch(self, config=None):
        config = learning.normalize_config(config or configuration())
        examples, _ = learning.load_examples('.', config)
        return learning.collate_examples(examples[:2], config['model'], config['loss'], config['scope'])

    def test_output_shapes_normalization_shared_one_pass_and_layers(self):
        torch = self.torch
        for mode in ('separate_heads', 'lora_output'):
            for layers in (1, 2, 4):
                with self.subTest(mode=mode, layers=layers):
                    config = configuration()
                    config['model'].update(expert_mode=mode, num_hidden_layers=layers)
                    batch = self.batch(config)
                    model = learning.build_model(config['model']).eval()
                    output = model(batch['input_ids'], batch['features'], batch['attention_mask'])
                    self.assertEqual(tuple(output['expert_logits'].shape), (2, 4, 2, 16))
                    self.assertEqual(tuple(output['gate_logits'].shape), (2, 4, 2))
                    self.assertEqual(tuple(output['feature_predictions'].shape), (2, 4, 4))
                    self.assertEqual(model.body.forward_calls, 1)
                    torch.testing.assert_close(output['mixture_log_probs'].exp().sum(-1), torch.ones(2, 4))

    def test_causal_body_and_padding(self):
        torch = self.torch
        config = configuration()
        batch = self.batch(config)
        model = learning.build_model(config['model']).eval()
        initial = model(batch['input_ids'], batch['features'], batch['attention_mask'])
        features = batch['features'].clone()
        features[:, 2:] += 10
        ids = batch['input_ids'].clone()
        ids[:, 2:] = (ids[:, 2:] + 1) % 16
        changed = model(ids, features, batch['attention_mask'])
        torch.testing.assert_close(initial['mixture_log_probs'][:, :2], changed['mixture_log_probs'][:, :2])
        attention = torch.tensor([[False, True, True, True], [False, False, False, False]])
        padded = model(ids, features, attention)
        self.assertTrue(bool(torch.isfinite(padded['mixture_log_probs']).all()))

    def test_lora_heads_and_trainable_gate_finite_gradients(self):
        torch = self.torch
        config = configuration()
        config['model']['expert_mode'] = 'lora_output'
        model = learning.build_model(config['model'])
        batch = self.batch(config)
        output = model(batch['input_ids'], batch['features'], batch['attention_mask'])
        self.assertEqual(tuple(model.expert_A.shape), (2, 8, 2))
        self.assertEqual(tuple(model.expert_B.shape), (2, 2, 16))
        torch.testing.assert_close(output['expert_logits'][:, :, 0], output['expert_logits'][:, :, 1])
        result = learning.compute_loss(output, batch, config['loss'])
        result['loss'].backward()
        self.assertIsNotNone(model.gate.weight.grad)
        self.assertGreater(float(model.gate.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.expert_B.grad.abs().sum()), 0)
        self.assertTrue(all(bool(torch.isfinite(value.grad).all()) for value in model.parameters() if value.grad is not None))

    def test_loss_mask_labels_are_not_shifted_and_teacher_detached(self):
        torch = self.torch
        logits = torch.tensor([[[5.0, -5.0], [-5.0, 5.0], [0.0, 0.0]]], requires_grad=True)
        log_probs = torch.log_softmax(logits, -1)
        batch = {'labels': torch.tensor([[0, 1, 0]]), 'attention_mask': torch.ones(1, 3, dtype=torch.bool), 'loss_mask': torch.tensor([[True, True, False]])}
        result = learning.compute_loss({'mixture_log_probs': log_probs}, batch)
        torch.testing.assert_close(result['loss'], -log_probs[0, torch.arange(2), torch.tensor([0, 1])].mean())
        result['loss'].backward()
        torch.testing.assert_close(logits.grad[0, 2], torch.zeros(2))
        batch['loss_mask'].zero_()
        with self.assertRaises(learning.LearningError):
            learning.compute_loss({'mixture_log_probs': log_probs}, batch)
        config = configuration()
        config['loss']['objective'] = 'teacher_kl'
        model = learning.build_model(config['model'])
        batch = self.batch(config)
        batch['teacher_logits'].requires_grad_()
        result = learning.compute_loss(model(batch['input_ids'], batch['features'], batch['attention_mask']), batch, config['loss'])
        result['loss'].backward()
        self.assertIsNone(batch['teacher_logits'].grad)

    def test_missing_feature_target_empty_mask_and_shape_fail(self):
        config = learning.normalize_config(configuration())
        values, _ = learning.load_examples('.', config)
        value = dict(values[0])
        del value['feature_targets']
        with self.assertRaises(learning.LearningError):
            learning.validate_example(value, config['model'], config['loss'], 'research')
        loss = {**config['loss'], 'feature_mse_target': 'input_reconstruction'}
        learning.validate_example(value, config['model'], loss, 'engineering')
        with self.assertRaises(learning.LearningError):
            learning.validate_example(value, config['model'], loss, 'research')
        value = dict(values[0])
        value['loss_mask'] = value['loss_mask'].clone().zero_()
        with self.assertRaises(learning.LearningError):
            learning.validate_example(value, config['model'], config['loss'])
        value = {**values[0], 'features': values[0]['features'][:-1]}
        with self.assertRaises(learning.LearningError):
            learning.validate_example(value, config['model'], config['loss'])

    def test_manifest_tensor_loader_and_no_silent_truncation(self):
        torch = self.torch
        config = learning.normalize_config(configuration())
        values, _ = learning.load_examples('.', config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for index, split in enumerate(('train', 'validation')):
                path = root / (str(index) + '.pt')
                torch.save(values[index], path)
                entries.append({'path': str(path), 'sha256': learning.file_sha256(path), 'split': split, 'tokens': 4, 'loss_tokens': 4})
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'status': 'complete', 'command': 'features', 'alignment': 'already aligned next-feature targets', 'examples': entries}))
            config['data'] = {'path': str(root), 'split': 'train'}
            config['scope'] = 'research'
            loaded, details = learning.load_examples(root, config)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(details['split'], 'train')
            torch.testing.assert_close(loaded[0]['labels'], values[0]['labels'])
            entries[0]['tokens'] = 3
            manifest.write_text(json.dumps({'status': 'complete', 'command': 'features', 'alignment': 'aligned', 'examples': entries}))
            with self.assertRaises(learning.LearningError):
                learning.load_examples(root, config)

    def test_training_accumulation_and_checkpoint_resume_exact(self):
        torch = self.torch
        config = configuration()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = learning.train(config, root, device='cpu', allow_cpu=True, stop_after_steps=1)
            self.assertEqual(first['status'], 'partial')
            self.assertEqual(first['counters']['loss_tokens'], 8)
            self.assertFalse(first['budget_eligibility']['eligible'])
            resumed = learning.train(config, root, device='cpu', allow_cpu=True, resume='latest')
            self.assertEqual(resumed['status'], 'complete')
            self.assertTrue(resumed['budget_eligibility']['eligible'])
            self.assertEqual(resumed['counters']['optimizer_steps'], 2)
            self.assertEqual(resumed['counters']['micro_batches'], 4)
            self.assertEqual(resumed['body_forward_calls_this_attempt'], 2)
            other = copy.deepcopy(config)
            other['run_id'] = 'uninterrupted'
            full = learning.train(other, root, device='cpu', allow_cpu=True)
            a = torch.load(resumed['checkpoint'], map_location='cpu', weights_only=True)
            b = torch.load(full['checkpoint'], map_location='cpu', weights_only=True)
            for name in a['model']:
                torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
            Path(resumed['checkpoint']).with_suffix('.json').unlink()
            history = root / 'outputs/learning' / config['experiment'] / config['run_id'] / 'training.jsonl'
            with history.open('ab') as stream:
                stream.write(b'{torn-final-line')
            recovered = learning.train(config, root, device='cpu', allow_cpu=True, resume='latest')
            self.assertEqual(len(recovered['checkpoint_recoveries']), 1)
            self.assertIsNotNone(recovered['history_recovery'])
            packed = torch.load(recovered['checkpoint'], map_location='cpu', weights_only=True)
            for name in a['model']:
                torch.testing.assert_close(a['model'][name], packed['model'][name], rtol=0, atol=0)
            with self.assertRaises(learning.LearningError):
                learning.train(config, root, device='cpu', allow_cpu=True)
            changed = copy.deepcopy(config)
            changed['training']['learning_rate'] = 0.002
            with self.assertRaises(learning.LearningError):
                learning.train(changed, root, device='cpu', allow_cpu=True, resume='latest')

    def test_resume_interrupted_first_checkpoint_without_any_committed_receipt(self):
        config = configuration()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = learning.train(config, root, device='cpu', allow_cpu=True, stop_after_steps=1)
            Path(first['checkpoint']).with_suffix('.json').unlink()
            resumed = learning.train(config, root, device='cpu', allow_cpu=True, resume='latest')
            self.assertEqual(resumed['status'], 'complete')
            self.assertEqual(len(resumed['checkpoint_recoveries']), 1)
            self.assertTrue(resumed['budget_eligibility']['eligible'])


if __name__ == '__main__':
    unittest.main()
