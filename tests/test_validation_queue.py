import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


PATH = Path(__file__).resolve().parents[1] / 'experiments/validation-queue-20261110/driver.py'
SPEC = importlib.util.spec_from_file_location('validation_queue', PATH)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class ValidationQueueTests(unittest.TestCase):
    def test_sampling_streams_have_no_seed_collisions(self):
        seeds = [queue.sampling_seed(method, mode, prompt, repeat) for method in queue.RUNS for mode in ('target_only', 'speculative') for prompt in range(10) for repeat in range(1024)]
        self.assertEqual(len(seeds), 61440)
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertLess(max(seeds), 2 ** 32)
        with self.assertRaises(RuntimeError):
            queue.sampling_seed('eagle3', 'target_only', 0, 1024)

    def test_terminal_events_preserve_terminating_position(self):
        values = queue.continuation_events([11, 99], {99})
        self.assertEqual(values[:2], [queue.hash_partition(99, salt) for salt in (0, 1)])
        self.assertEqual(values[2:4], [queue.hash_partition('END', salt) for salt in (0, 1)])
        self.assertEqual(values[8:11], [1, 1, 1])
        self.assertEqual(values[11], 0)

    def test_repeated_four_grams_are_detected(self):
        self.assertEqual(queue.continuation_events([1, 2, 3, 4, 1, 2, 3, 4], set())[-1], 1)
        self.assertEqual(queue.continuation_events([1, 2, 3, 4], set())[-1], 0)

    def test_registered_bounds_have_declared_joint_power(self):
        self.assertLess(2 * queue.conditional_bound(65536), 0.02)
        self.assertLess(2 * queue.sampling_bound(10240), 0.06)
        self.assertGreater(queue.sampling_bound(20), 0.06)

    def test_missing_coverage_never_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            report = queue.summarize(path, 'sampling', 8, 1024, 65536)
            self.assertFalse(report['execution_complete'])
            self.assertFalse(report['passed_scoped_checks'])
            self.assertFalse(report['training_admission'])

    def test_changed_raw_file_invalidates_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'sampling-eagle3-0.jsonl'
            path.write_text('')
            queue.atomic(path.with_suffix('.receipt.json'), {'status': 'complete', 'raw_sha256': queue.digest(path), 'failed_checks': 0})
            path.write_text(json.dumps({'method': 'eagle3'}) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                queue.summarize(directory, 'sampling', 8, 1024, 65536)

    def test_duplicate_generation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'sampling-eagle3-0.jsonl'
            row = {'method': 'eagle3', 'prompt_index': 0, 'repeat': 0, 'mode': 'target_only', 'events': [0] * 12}
            path.write_bytes((queue.canonical(row) + b'\n') * 2)
            queue.atomic(path.with_suffix('.receipt.json'), {'status': 'complete', 'raw_sha256': queue.digest(path), 'failed_checks': 0})
            with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
                queue.summarize(directory, 'sampling', 8, 1024, 65536)

    def test_independent_reference_resets_inference_cache_inside_inference_mode(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch required for inference-tensor regression')
        with torch.inference_mode():
            lengths = torch.tensor([17])
        seen = []

        def forward(prefix, past_key_values, use_cache):
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertEqual(int(lengths[0]), 0)
            self.assertTrue(use_cache)
            seen.append(prefix.tolist())
            return SimpleNamespace(logits=torch.tensor([[[0.0, 1.0, 2.0]]]))

        model = SimpleNamespace(current_length_data=lengths, past_key_values=['fresh-reset-cache'], base_model=forward)
        backend = SimpleNamespace(model=model, module=SimpleNamespace(reset_tree_mode=lambda _: None))
        weights, q = queue.ar_reference(backend, torch.tensor([[11, 22]]))
        self.assertEqual(seen, [[[11, 22]]])
        self.assertEqual(int(q.argmax()), 2)
        self.assertAlmostEqual(float(q.sum()), 1.0)
        self.assertEqual(weights.shape, (3,))


if __name__ == '__main__':
    unittest.main()
