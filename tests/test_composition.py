import copy
from fractions import Fraction
import itertools
import math
from pathlib import Path
import random
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taps_rebuild.composition import Calibrator, calibration_diagnostics, exact_sequential_distribution, fit_calibrator, evaluate_calibrator, merge_weights, pooled_budget_tree, sample_target_children, validate_tree, verify_tree


def draft(expert, probabilities, root=9):
    return {"expert_id": expert, "root_token": root, "root_source": "target_greedy",
            "nodes": [{"path": list(path), "log_probability": math.log(probability) if probability else -math.inf} for path, probability in probabilities.items()]}


def examples(expert="a"):
    return [{"example_id": expert + str(index), "log_probability": math.log(probability),
             "accepted": label, "split": "dev", "label_source": "target_verification"}
            for index, (probability, label) in enumerate([(0.1, 0), (0.2, 1), (0.3, 0), (0.8, 1), (0.9, 1)])]


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.first = {(1,): 0.6, (1, 2): 0.3, (3,): 0.4}
        self.second = {(1,): 0.6, (1, 4): 0.2, (3,): 0.4, (3, 5): 0.1}
        self.drafts = [draft("a", self.first), draft("b", self.second)]

    def build(self, budget=4, **kwargs):
        return pooled_budget_tree(self.drafts, root_token=9, root_source="target_greedy", budget=budget, score_mode="oracle", **kwargs)

    def test_exact_prefix_deduplication_retains_both_experts(self):
        tree = self.build()
        self.assertEqual(tree["metadata"]["unique_pool_nodes"], 6)
        self.assertEqual(tree["paths_after_root"], [[], [1], [3], [1, 2]])
        self.assertEqual(tree["parent_indices"], [-1, 0, 0, 1])
        self.assertEqual(tree["depth"], [0, 1, 1, 2])
        self.assertEqual(tree["retrieve_paths"], [[0, 2, -1], [0, 1, 3]])
        self.assertEqual({row["expert_id"] for row in tree["provenance"][1]}, {"a", "b"})
        self.assertTrue(validate_tree(tree))
        self.assertTrue(tree["metadata"]["oracle_not_available_at_inference"])
        self.assertFalse(tree["metadata"]["scientific_result"])

    def test_oracle_selection_is_optimal_over_every_prefix_valid_subset(self):
        probabilities = {**self.first, **self.second}
        paths = list(probabilities)
        for budget in range(1, len(paths) + 2):
            objective = self.build(budget)["metadata"]["selected_score_sum_excluding_root"]
            possible = []
            for subset in itertools.combinations(paths, budget - 1):
                chosen = set(subset)
                if all(not path[:-1] or path[:-1] in chosen for path in chosen):
                    possible.append(sum(probabilities[path] for path in chosen))
            self.assertAlmostEqual(objective, max(possible))

    def test_ties_keep_shallower_prefixes_and_are_deterministic(self):
        source = [draft("a", {(1,): 0.5, (1, 2): 0.5, (3,): 0.5})]
        first = pooled_budget_tree(source, root_token=9, root_source="target_greedy", budget=3, score_mode="oracle")
        second = pooled_budget_tree(source, root_token=9, root_source="target_greedy", budget=3, score_mode="oracle")
        self.assertEqual(first, second)
        self.assertEqual(first["paths_after_root"], [[], [1], [3]])

    def test_invalid_prefixes_duplicate_expert_nodes_and_root_assumptions_rejected(self):
        cases = [draft("a", {(1, 2): 0.2}), draft("a", {(1,): 0.2}, root=8)]
        repeated = draft("a", {(1,): 0.2})
        repeated["nodes"] *= 2
        cases.append(repeated)
        for item in cases:
            with self.subTest(item=item), self.assertRaises(ValueError):
                pooled_budget_tree([item], root_token=9, root_source="target_greedy", budget=2, score_mode="oracle")
        with self.assertRaisesRegex(ValueError, "fill the fixed"):
            self.build(7)
        broken = self.build()
        broken["parent_indices"][3] = 0
        with self.assertRaisesRegex(ValueError, "prefix hole"):
            validate_tree(broken)

    def test_oracle_disagreement_cannot_be_hidden_by_projection(self):
        with self.assertRaisesRegex(ValueError, "monotone"):
            pooled_budget_tree([draft("a", {(1,): 0.1, (1, 2): 0.8})], root_token=9, root_source="target_greedy", budget=2, score_mode="oracle")
        with self.assertRaisesRegex(ValueError, "disagree"):
            pooled_budget_tree([draft("a", {(1,): 0.1}), draft("b", {(1,): 0.2})], root_token=9, root_source="target_greedy", budget=2, score_mode="oracle")

    def test_calibration_is_dev_only_and_tracks_fit_identity(self):
        for method in ["temperature", "isotonic"]:
            calibrator = fit_calibrator("a", examples(), method=method, dataset_identity="a" * 64)
            metadata = calibrator.metadata()
            self.assertEqual(metadata["fitted_split"], "dev")
            self.assertEqual(len(metadata["calibrator_identity"]), 64)
            self.assertEqual(Calibrator.from_metadata(metadata), calibrator)
            self.assertTrue(all(0 <= calibrator.predict(math.log(p)) <= 1 for p in (0.01, 0.2, 0.99)))
            with self.assertRaisesRegex(ValueError, "development"):
                fit_calibrator("a", examples(), method=method, split="test", dataset_identity="a" * 64)
            with self.assertRaisesRegex(ValueError, "forbidden"):
                fit_calibrator("a", examples(), method=method, dataset_identity="a" * 64, forbidden_ids=["a0"])
            test = [{**row, "split": "test"} for row in examples()]
            with self.assertRaisesRegex(ValueError, "overlap"):
                evaluate_calibrator(calibrator, test, split="test", dataset_identity="b" * 64)

    def test_estimated_projection_has_no_prefix_holes(self):
        calibrator = fit_calibrator("a", examples(), dataset_identity="a" * 64)
        tree = pooled_budget_tree([draft("a", {(1,): 0.1, (1, 2): 0.8, (3,): 0.4})], root_token=9, root_source="target_greedy", budget=3, calibrators={"a": calibrator})
        self.assertGreater(tree["metadata"]["monotonic_projections"], 0)
        self.assertFalse(tree["metadata"]["estimated_optimality_claimed"])
        self.assertTrue(validate_tree(tree))
        diagnostics = calibration_diagnostics([0.0, 1.0], [0, 1])
        self.assertEqual(diagnostics["ece"], 0)
        self.assertEqual(diagnostics["brier_score"], 0)

    def test_exact_finite_distribution_with_all_child_orders_and_duplicates(self):
        target = [Fraction(1, 10), Fraction(2, 10), Fraction(3, 10), Fraction(4, 10)]
        for length in range(5):
            for children in itertools.permutations(range(4), length):
                remaining = target[:]
                reach = Fraction(1)
                exact = [Fraction(0) for _ in target]
                for token in children:
                    acceptance = remaining[token] / sum(remaining)
                    exact[token] += reach * acceptance
                    reach *= 1 - acceptance
                    remaining[token] = 0
                if sum(remaining):
                    for token, probability in enumerate(remaining):
                        exact[token] += reach * probability / sum(remaining)
                self.assertEqual(exact, target)
                repeated = children + children[:1]
                actual = exact_sequential_distribution(target, repeated)
                for expected, probability in zip(target, actual):
                    self.assertAlmostEqual(float(expected), probability)

    def test_actual_python_sampler_matches_target_in_small_monte_carlo(self):
        target = [0.1, 0.2, 0.3, 0.4]
        for children in [[], [0], [2, 2, 0], [3, 2, 1, 0]]:
            generator = random.Random(1234)
            counts = [0] * 4
            for _ in range(8000):
                result = sample_target_children(target, children, rng=generator)
                counts[result["token"]] += 1
                self.assertFalse(result["uses_draft_probabilities"])
            for probability, count in zip(target, counts):
                self.assertLess(abs(probability - count / 8000), 0.025)

    def test_full_tree_greedy_reference_preserves_uncommitted_bonus(self):
        tree = pooled_budget_tree([draft("a", {(1,): 0.6, (2,): 0.3, (1, 3): 0.48})], root_token=9, root_source="target_greedy", budget=4, score_mode="oracle")
        target = [[0.1, 0.6, 0.3, 0], [0, 0.1, 0.1, 0.8], [1, 0, 0, 0], [1, 0, 0, 0]]
        result = verify_tree(tree, target, greedy=True)
        self.assertEqual(result["emitted_tokens"], [9, 1, 3])
        self.assertEqual(result["accepted_draft_tokens"], 2)
        self.assertEqual(result["next_root_token"], 0)
        self.assertTrue(result["next_root_uncommitted"])

    def test_weight_merges_preserve_shapes_buffers_and_shared_ancestor_guards(self):
        ancestor = {"w": [1.0, 1.0], "step": 2}
        experts = {"a": {"w": [3.0, 1.0], "step": 2}, "b": {"w": [1.0, 5.0], "step": 2}}
        kwargs = {"ancestor_identity": "a" * 64, "expert_ancestors": {"a": "a" * 64, "b": "a" * 64}}
        original = copy.deepcopy(experts)
        self.assertEqual(merge_weights(ancestor, experts, **kwargs)["state_dict"]["w"], [2.0, 3.0])
        for method in ["task_arithmetic", "ties", "dare"]:
            result = merge_weights(ancestor, experts, method=method, density=1.0, **kwargs)
            self.assertEqual(result["state_dict"]["w"], [3.0, 5.0])
            self.assertEqual(result["state_dict"]["step"], 2)
            self.assertFalse(result["metadata"]["scientific_result"])
        self.assertEqual(experts, original)
        kwargs["expert_ancestors"]["b"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "Independent"):
            merge_weights(ancestor, experts, **kwargs)
        diagnostic = merge_weights(ancestor, experts, diagnostic_independent=True, **kwargs)
        self.assertTrue(diagnostic["metadata"]["diagnostic_independent_initializations"])

    def test_weight_schema_mismatches_and_dare_rng_are_explicit(self):
        ancestor = {"w": [0.0, 0.0, 0.0, 0.0]}
        experts = {"a": {"w": [1.0, 2.0, 3.0, 4.0]}}
        kwargs = {"ancestor_identity": "a" * 64, "expert_ancestors": {"a": "a" * 64}}
        first = merge_weights(ancestor, experts, method="dare", density=0.5, seed=7, **kwargs)
        second = merge_weights(ancestor, experts, method="dare", density=0.5, seed=7, **kwargs)
        self.assertEqual(first, second)
        for wrong in [{"a": {"w": [1.0]}}, {"a": {"other": [1.0] * 4}}, {"a": {"w": [1, 2, 3, 4]}}]:
            with self.assertRaises(ValueError):
                merge_weights(ancestor, wrong, **kwargs)
        with self.assertRaisesRegex(ValueError, "weights"):
            merge_weights(ancestor, experts, weights={}, **kwargs)


if __name__ == "__main__":
    unittest.main()
