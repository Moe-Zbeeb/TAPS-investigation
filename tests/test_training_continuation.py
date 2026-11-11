import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / 'experiments/native-restart-20261111/continue_training.py'
SPEC = importlib.util.spec_from_file_location('native_training_continuation', PATH)
continuation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuation)


class ContinuationTests(unittest.TestCase):
    def test_only_the_exact_owned_terminal_full_training_job_is_eligible(self):
        self.assertEqual(continuation.completed_job('123|TIMEOUT|0:15|mohamadzbib|taps-full-train\n', '123'), 'TIMEOUT')
        for row in ['123|RUNNING|0:0|mohamadzbib|taps-full-train', '123|TIMEOUT|0:15|other|taps-full-train', '123|TIMEOUT|0:15|mohamadzbib|deep10-hold', '123.batch|TIMEOUT|0:15|mohamadzbib|taps-full-train']:
            with self.assertRaises(ValueError):
                continuation.completed_job(row, '123')

    def test_duplicate_accounting_rows_do_not_trigger_duplicate_submission(self):
        row = '123|TIMEOUT|0:15|mohamadzbib|taps-full-resume\n'
        with self.assertRaises(ValueError):
            continuation.completed_job(row * 2, '123')
