from pathlib import Path
import unittest
from unittest.mock import Mock

from taps_rebuild.native_full_body import AuditRouter
from taps_rebuild.native_training import NativeTrainingError


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


if __name__ == '__main__':
    unittest.main()
