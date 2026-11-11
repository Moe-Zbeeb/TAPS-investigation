import copy
import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / 'experiments/seed-policy-20261111/driver.py'
SPEC = importlib.util.spec_from_file_location('seed_policy', PATH)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def fixture():
    cases = [{'benchmark': benchmark, 'question_id': index, 'turn_id': 0, 'input_token_ids': [index + 100], 'input_sha256': str(index)} for benchmark in study.REFERENCES for index in range(80)]
    rows = [{'case': case, 'arm': arm, 'repeat': repeat, 'actual_seed': study.actual_seed(arm, case, repeat), 'seed': study.actual_seed(arm, case, repeat), 'temperature': 1, 'output_token_ids': [1, 2, 3], 'cycles': [{'emitted_tokens': 3}]} for case in cases for arm in study.ARMS for repeat in range(5)]
    return rows, cases


class SeedPolicyTests(unittest.TestCase):
    def test_hash_arm_matches_registered_runner(self):
        from taps_rebuild.runner import seed_for
        case = {'benchmark': 'gsm8k', 'question_id': 21}
        for repeat in range(5):
            self.assertEqual(study.actual_seed('upstream_seed_zero', case, repeat), 0)
            self.assertEqual(study.actual_seed('registered_hash', case, repeat), seed_for(20261110, 'gsm8k', 21, 0, repeat))

    def test_repeats_remain_within_question_clusters_and_never_admit_training(self):
        rows, cases = fixture()
        report = study.summarize_rows(rows, cases)
        self.assertEqual(report['measured_generations'], 1600)
        self.assertFalse(report['training_admission'])
        for cell in report['cells']:
            self.assertEqual(cell['questions'], 80)
            if 'arm' in cell:
                self.assertEqual(cell['question_mean_tokens_per_cycle'], 3)
                self.assertEqual(cell['questions_with_identical_replay_tokens'], 80)
            else:
                self.assertEqual(cell['bootstrap_ci95'], [0, 0])

    def test_missing_duplicate_and_wrong_seed_records_are_rejected(self):
        rows, cases = fixture()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete seed comparison'):
            study.summarize_rows(rows[:-1], cases)
        with self.assertRaisesRegex(RuntimeError, 'Duplicate measurement'):
            study.summarize_rows(rows + [rows[0]], cases)
        rows = copy.deepcopy(rows)
        rows[0]['actual_seed'] = 99
        with self.assertRaisesRegex(RuntimeError, 'Actual RNG policy'):
            study.summarize_rows(rows, cases)

    def test_input_or_cycle_mismatch_is_rejected(self):
        rows, cases = fixture()
        rows = copy.deepcopy(rows)
        rows[0]['case']['input_token_ids'] = [999]
        with self.assertRaisesRegex(RuntimeError, 'Input differs'):
            study.summarize_rows(rows, cases)
        rows, cases = fixture()
        rows[0]['cycles'][0]['emitted_tokens'] = 2
        with self.assertRaisesRegex(RuntimeError, 'Cycle coverage'):
            study.summarize_rows(rows, cases)


if __name__ == '__main__':
    unittest.main()
