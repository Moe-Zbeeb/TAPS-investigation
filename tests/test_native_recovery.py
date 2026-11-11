import ast
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from taps_rebuild import native_recovery as recovery


SOURCE = '''checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)
for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch+1)
    for batch_idx, data in enumerate(tqdm(train_loader)):
        model_engine.step()
'''


class RecoveryTests(unittest.TestCase):
    def test_pinned_transform_preserves_original_and_requires_every_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'main.py'
            source.write_text(SOURCE)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            transformed = recovery.transform(source, digest)
            text = ast.unparse(transformed)
            self.assertEqual(source.read_text(), SOURCE)
            self.assertIn('_taps_recovery.epochs', text)
            self.assertIn('_taps_recovery.batches', text)
            self.assertIn('_taps_recovery.commit_batch()', text)
            self.assertIsInstance(transformed.body[0].value, ast.Tuple)
            self.assertEqual([value.value for value in transformed.body[0].value.elts], [None, 0])
            source.write_text(SOURCE.replace('tqdm(train_loader)', 'train_loader'))
            with self.assertRaisesRegex(ValueError, 'pinned upstream bytes'):
                recovery.transform(source, digest)
            with self.assertRaisesRegex(ValueError, 'every pinned loop'):
                recovery.transform(source, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_resume_skips_committed_batches_then_restores_rng_and_epoch_metrics(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'TAPS_NATIVE_ATTEMPT_ID': '2'}):
            path = Path(directory)
            (path / 'attempts/attempt-2').mkdir(parents=True)
            manager = recovery.Recovery.__new__(recovery.Recovery)
            manager.plan = {'run_dir': directory}
            manager.cursor = {'epoch': 3, 'next_batch': 2}
            manager.rank, manager.observation = 0, {}
            sampler = {'epoch': 4, 'seed': 0, 'rank': 0, 'num_replicas': 8, 'shuffle': True, 'drop_last': False, 'num_samples': 4, 'total_size': 32}
            manager.namespace = {'train_sampler': SimpleNamespace(**sampler), 'epoch_acces': [], 'epoch_plosses': []}
            manager.torch = None
            manager.helpers = SimpleNamespace(canonical=lambda value: json.dumps(value).encode(), tensor_state_identity=lambda value, torch: str(value))
            random.seed(77)
            saved_rng = random.getstate()
            expected_draw = random.random()
            manager.restored = {'sampler': sampler, 'rng': saved_rng, 'epoch_metrics': {'epoch_acces': [[0.5]], 'epoch_plosses': [[1.0]]}}
            manager.restore_rng = random.setstate
            class Loader:
                def __len__(self):
                    return 4
                def __iter__(self):
                    random.random()
                    return iter([{'input_ids': index, 'attention_mask': 1} for index in range(4)])
            batches = list(manager.batches(Loader(), 3))
            self.assertEqual([index for index, _ in batches], [2, 3])
            self.assertEqual(manager.cursor, {'epoch': 3, 'next_batch': 4})
            self.assertEqual(manager.namespace['epoch_plosses'], [[1.0]])
            self.assertEqual(random.random(), expected_draw)
            self.assertTrue(manager.observation['restored_rng_and_cursor'])
            rows = [json.loads(line) for line in (path / 'attempts/attempt-2/batches-rank-0.jsonl').read_text().splitlines()]
            self.assertEqual([row['batch_index'] for row in rows], [2, 3])

    def test_recovery_budget_policy_rejects_uncommittable_stops(self):
        self.assertEqual(recovery.validate_policy({'checkpoint_interval': 100, 'stop_after_optimizer_step': None})['checkpoint_interval'], 100)
        for value in [{}, {'checkpoint_interval': True, 'stop_after_optimizer_step': None}, {'checkpoint_interval': 100, 'stop_after_optimizer_step': 101}, {'checkpoint_interval': 100, 'stop_after_optimizer_step': 0}]:
            with self.assertRaises(ValueError):
                recovery.validate_policy(value)
