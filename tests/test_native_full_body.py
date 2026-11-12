import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from taps_rebuild.native_full_body import AuditRouter
from taps_rebuild.native_training import NativeTrainingError
from taps_rebuild.native_full_body_queue import select_maximum_shapes


def fake_preprocessor(batch):
    selected = int(batch['id'][0]) % 5 == 2
    return {'input_ids': [SimpleNamespace(shape=(1, 2048))] if selected else []}


class AuditRouterTests(unittest.TestCase):
    def test_routes_only_matching_frozen_audit(self):
        delegate = Mock()
        command = ['/env/bin/python', '-B', '/run/inputs/native_training.py', 'audit', '--plan', '/run/plan.json']
        AuditRouter(delegate).run(command, check=True)
        delegate.run.assert_called_once_with(['/env/bin/python', '-B', '/run/inputs/native_full_body.py', 'audit', '--plan', '/run/plan.json'], check=True)
        self.assertEqual(command[2], '/run/inputs/native_training.py')

    def test_rejects_other_legacy_operations_and_relocated_audits(self):
        for command in ([ '/python', '-B', '/run/inputs/native_training.py', 'worker', '--plan', '/run/plan.json'], ['/python', '-B', '/other/inputs/native_training.py', 'audit', '--plan', '/run/plan.json']):
            delegate = Mock()
            with self.assertRaises(NativeTrainingError):
                AuditRouter(delegate).run(command)
            delegate.run.assert_not_called()

    def test_preserves_unrelated_subprocesses(self):
        delegate = Mock()
        command = ['git', '-C', '/source', 'status', '--porcelain']
        AuditRouter(delegate).run(command, capture_output=True)
        delegate.run.assert_called_once_with(command, capture_output=True)
        self.assertIs(AuditRouter(delegate).Popen, delegate.Popen)


class MaximumShapeSelectionTests(unittest.TestCase):
    def select(self, count):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False), set_num_threads=lambda count: None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.jsonl'
            path.write_text(''.join(json.dumps({'id': str(index)}) + '\n' for index in range(count)))
            return select_maximum_shapes(path, fake_preprocessor, torch)

    def test_parallel_selection_preserves_first64_and_exact_prefix(self):
        rows, inspected = self.select(400)
        self.assertEqual([row['id'] for row in rows], [str(index) for index in range(2, 318, 5)])
        self.assertEqual(inspected, 318)

    def test_partial_final_chunk_counts_every_inspected_row(self):
        rows, inspected = self.select(50)
        self.assertEqual([row['id'] for row in rows], [str(index) for index in range(2, 50, 5)])
        self.assertEqual(inspected, 50)


if __name__ == '__main__':
    unittest.main()
