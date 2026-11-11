import unittest

from taps_rebuild.validation_budget import comparison_tokens, output_budget


class ValidationBudgetTests(unittest.TestCase):
    def test_budget_is_fixed_from_prompt_and_context_before_decoding(self):
        self.assertEqual(output_budget(1542, 512, 2048, 59), 437)
        self.assertEqual(output_budget(1557, 512, 2048, 59), 422)
        self.assertEqual(output_budget(100, 512, 2048, 59), 512)
        with self.assertRaisesRegex(ValueError, 'No generation budget'):
            output_budget(1979, 512, 2048, 59)

    def test_unequal_overshoot_is_retained_in_raw_but_has_one_fixed_projection(self):
        ar, tree = [1] * 438, [1] * 440
        self.assertEqual(comparison_tokens(ar, 437, {2}), comparison_tokens(tree, 437, {2}))
        self.assertEqual((len(ar), len(tree)), (438, 440))
        with self.assertRaisesRegex(ValueError, 'before the predetermined budget'):
            comparison_tokens(ar, 439, {2})

    def test_first_terminal_is_included_and_later_tokens_are_ignored(self):
        self.assertEqual(comparison_tokens([1, 2, 9, 9], 3, {2}), [1, 2])
        self.assertEqual(comparison_tokens([1, 1, 1, 2], 3, {2}), [1, 1, 1])
        self.assertEqual(comparison_tokens([2], 3, {2}), [2])

    def test_projection_never_hides_a_token_disagreement_inside_the_fixed_budget(self):
        self.assertNotEqual(comparison_tokens([1, 3, 9], 2, {9}), comparison_tokens([1, 4, 9], 2, {9}))

    def test_invalid_budgets_and_unfinished_generations_are_rejected(self):
        for values in [(True, 1, 200, 59), (1, 0, 200, 59), (1, 1.0, 200, 59)]:
            with self.assertRaises(ValueError):
                output_budget(*values)
        for raw, budget in [([], 1), ([1], 2), ([True], 1), ([-1], 1), ([1], True)]:
            with self.assertRaises(ValueError):
                comparison_tokens(raw, budget, {2})
