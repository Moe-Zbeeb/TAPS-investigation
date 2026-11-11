import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / 'experiments/upstream-cli-20261111/driver.py'
SPEC = importlib.util.spec_from_file_location('native_cli', PATH)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class UpstreamCliTests(unittest.TestCase):
    def test_native_zero_based_iterations_are_converted_to_true_cycles_across_turns(self):
        questions = [{'question_id': 1, 'turns': ['a', 'b']}, {'question_id': 2, 'turns': ['c']}]
        rows = [{'question_id': 1, 'choices': [{'index': 0, 'turns': ['x', 'y'], 'idxs': [1, 3], 'new_tokens': [8, 16], 'wall_time': [1, 2]}]}, {'question_id': 2, 'choices': [{'index': 0, 'turns': ['z'], 'idxs': [1], 'new_tokens': [4], 'wall_time': [1]}]}]
        report = cli.summarize(rows, questions)
        self.assertEqual(report['official_counter_question_mean_tau'], 3)
        self.assertEqual(report['zero_based_index_ratio_diagnostic'], 5)
        self.assertFalse(report['raw_token_counter_equivalence_established'])

    def test_duplicate_or_incomplete_native_records_are_rejected(self):
        questions = [{'question_id': 1, 'turns': ['a', 'b']}, {'question_id': 2, 'turns': ['c']}]
        row = {'question_id': 1, 'choices': [{'index': 0, 'turns': ['x'], 'idxs': [1], 'new_tokens': [8], 'wall_time': [1]}]}
        with self.assertRaisesRegex(ValueError, 'coverage'):
            cli.summarize([row, row], questions)
        with self.assertRaisesRegex(ValueError, 'turn counters'):
            cli.summarize([row], questions[:1])
