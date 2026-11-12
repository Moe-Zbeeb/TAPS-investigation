from collections import Counter
from pathlib import Path
import tempfile
import unittest

from taps_rebuild import pilot_study as study, pilot_train as train


def pools(size):
    return {domain: [{'example_id': domain + str(index), 'index': index, 'eligible_positions': [list(range(6, 100)), list(range(8, 200))]} for index in range(size)] for domain in study.DOMAINS}


class PilotScheduleTests(unittest.TestCase):
    def test_full_union_budget_and_rank_offsets(self):
        for rank in range(8):
            rows = list(study.schedule(pools(137), 'generalist', 0, rank))
            self.assertEqual(len(rows), 1536)
            self.assertEqual(Counter(row['domain'] for row in rows), {'chat': 512, 'math': 512, 'code': 512})
            self.assertEqual([row['step'] for row in rows], list(range(1536)))

    def test_repeated_occurrences_are_contiguous_across_cycle_boundaries(self):
        rows = list(study.schedule(pools(7), 'math', 2, 0))
        seen = Counter()
        for row in rows:
            self.assertEqual(row['occurrence'], seen[row['example_id']])
            seen[row['example_id']] += 1
        self.assertEqual(set(seen.values()), {219, 220})
        self.assertEqual(Counter(row['domain'] for row in rows), {'math': 1536})

    def test_policy_pairing_and_specialist_union_domain_prefix(self):
        source = pools(11)
        specialist = list(study.schedule(source, 'code', 1, 0))
        union = [row for row in study.schedule(source, 'generalist', 1, 0) if row['domain'] == 'code']
        self.assertEqual([(row['example_id'], row['occurrence']) for row in specialist[:512]], [(row['example_id'], row['occurrence']) for row in union])
        self.assertEqual(specialist, list(study.schedule(source, 'code', 1, 0)))
        self.assertNotEqual(specialist, list(study.schedule(source, 'code', 2, 0)))

    def test_missing_admission_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                train.admission({'study_directory': directory})

    def test_checkpoint_and_qualification_weights_stay_in_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / 'outputs/qualification/original/uninterrupted'
            plan = {'project_root': str(root), 'trained_models': str(root / 'models/trained/pilot-24/one')}
            self.assertEqual(train.model_directory(plan, output, False), root / 'models/trained/pilot-24/one')
            self.assertTrue(train.model_directory(plan, output, True).is_relative_to(root / 'models/trained/pilot-qualifications'))
