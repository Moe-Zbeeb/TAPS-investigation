import unittest
import json
from pathlib import Path
import tempfile

from taps_rebuild import pool
from taps_rebuild.pilot_cache import quantile_positions, rank_for, safe_positions, verified_corpus


class PilotMaskTests(unittest.TestCase):
    def test_seven_heads_cannot_select_prefix_or_future_padding(self):
        self.assertEqual(safe_positions([1] * 20, 10), [6, 7, 8])
        self.assertEqual(safe_positions([0] * 6 + [1, 0, 1, 1, 1], 10), [6, 8])

    def test_sampling_preserves_unique_order_for_sparse_and_minimum_pools(self):
        for size in (32, 33, 64, 65, 1000):
            eligible = list(range(7, size * 3 + 7, 3))
            for quota in (1, 32):
                for occurrence in (0, 1, 127, 128, 1535):
                    selected = quantile_positions(eligible, quota, 2, 'paired-id', occurrence)
                    self.assertEqual(len(set(selected)), quota)
                    self.assertEqual(selected, sorted(selected))
                    self.assertTrue(set(selected) <= set(eligible))

    def test_paired_quantiles_share_variates_on_different_lengths(self):
        small = quantile_positions(list(range(64)), 32, 1, 'pair', 4)
        large = quantile_positions(list(range(128)), 32, 1, 'pair', 4)
        self.assertEqual([value // 2 for value in small], [value // 4 for value in large])
        self.assertNotEqual(small, quantile_positions(list(range(64)), 32, 1, 'pair', 5))

    def test_quota_shortfall_is_rejected(self):
        for quota in (0, 33):
            with self.assertRaises(ValueError):
                quantile_positions(list(range(32)), quota, 0, 'pair', 0)

    def test_rank_assignment_has_no_policy_or_seed_input(self):
        ranks = [rank_for(str(index)) for index in range(1024)]
        self.assertEqual(set(ranks), set(range(8)))
        self.assertEqual(ranks, [rank_for(str(index)) for index in range(1024)])


class PilotProvenanceTests(unittest.TestCase):
    def fixture(self, root):
        for name in ('original', 'generated'):
            path = root / name
            path.mkdir()
            for split in ('train', 'dev', 'test'):
                (path / (split + '.jsonl')).write_text(json.dumps({'split': split}) + '\n')
        source = {'status': 'complete', 'scope': 'real_training_corpus_preparation', 'config_sha256': 'config', 'files': {split + '.jsonl': {'sha256': pool.digest(root / 'original' / (split + '.jsonl'))} for split in ('train', 'dev', 'test')}}
        pool.atomic_json(root / 'original/completion.json', source)
        (root / 'inputs').mkdir()
        (root / 'inputs/teacher.py').write_text('pass\n')
        (root / 'inputs/shard.jsonl').write_text('{}\n')
        plan = {'scope': 'full_training_data_preparation', 'corpus': str(root / 'original'), 'config_sha256': 'config', 'corpus_completion_sha256': pool.digest(root / 'original/completion.json'), 'target': {'model_identity_sha256': 'target'}, 'scripts': {'teacher.py': pool.digest(root / 'inputs/teacher.py')}, 'shards': [{'path': str(root / 'inputs/shard.jsonl'), 'sha256': pool.digest(root / 'inputs/shard.jsonl')}]}
        plan['identity_sha256'] = pool.identity(plan)
        pool.atomic_json(root / 'plan.json', plan)
        target = {'status': 'complete', 'scope': 'full_training_data_preparation', 'plan_path': str(root / 'plan.json'), 'plan_identity_sha256': plan['identity_sha256'], 'target_identity_sha256': 'target', 'response_regime': 'target_generated', 'files': {split + '.jsonl': {'sha256': pool.digest(root / 'generated' / (split + '.jsonl'))} for split in ('train', 'dev', 'test')}}
        pool.atomic_json(root / 'generated/completion.json', target)
        return {'original': 'original', 'generated': 'generated', 'original_completion_sha256': pool.digest(root / 'original/completion.json'), 'generated_completion_sha256': pool.digest(root / 'generated/completion.json'), 'teacher_plan_identity_sha256': plan['identity_sha256']}

    def test_actual_plan_mutation_is_rejected_even_when_receipt_identity_stays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            descriptor = self.fixture(root)
            self.assertTrue(verified_corpus(root, descriptor, 'target')[2])
            plan = json.loads((root / 'plan.json').read_text())
            plan['scope'] = 'changed'
            pool.atomic_json(root / 'plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'Actual teacher plan'):
                verified_corpus(root, descriptor, 'target')

    def test_pinned_output_and_frozen_input_bytes_are_both_checked(self):
        for artifact in ('generated/completion.json', 'inputs/shard.jsonl', 'inputs/teacher.py', 'generated/train.jsonl'):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                descriptor = self.fixture(root)
                path = root / artifact
                path.write_bytes(path.read_bytes() + b'\n')
                with self.assertRaises(ValueError):
                    verified_corpus(root, descriptor, 'target')
