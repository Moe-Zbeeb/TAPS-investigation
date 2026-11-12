import unittest

from taps_rebuild.pilot_cache import quantile_positions, rank_for, safe_positions


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
