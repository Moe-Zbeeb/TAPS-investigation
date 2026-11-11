import json
from pathlib import Path
import tempfile
import unittest

from taps_rebuild import admission, training_handoff


class HandoffTests(unittest.TestCase):
    def test_a_claimed_pass_without_complete_rank_coverage_cannot_admit_full_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'outputs').mkdir()
            config = {'restart_qualification': 'outputs/restart.json', 'export_qualification': 'outputs/export.json'}
            (root / 'outputs/restart.json').write_text(json.dumps({'status': 'passed', 'training_admission': False, 'ranks': [{'rank': 0, 'passed': True}] * 8}))
            with self.assertRaisesRegex(admission.AdmissionError, 'eight passing ranks'):
                training_handoff.qualifications(root, config)

    def test_another_training_implementation_cannot_supply_restart_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'outputs').mkdir()
            config = {'restart_qualification': 'outputs/restart.json', 'export_qualification': 'outputs/export.json'}
            (root / 'outputs/restart.json').write_text(json.dumps({'status': 'passed', 'training_admission': False, 'ranks': [{'rank': rank, 'passed': True} for rank in range(8)], 'native_training_sha256': 'other', 'native_recovery_sha256': 'other'}))
            with self.assertRaisesRegex(admission.AdmissionError, 'another training implementation'):
                training_handoff.qualifications(root, config)
