import unittest

from taps_rebuild import corpus_continue


class ContinuationTests(unittest.TestCase):
    def test_allocation_timeout_and_scheduler_preemption_can_resume(self):
        for holder, coordinator, restarts in [('TIMEOUT', 'FAILED', 0), ('RUNNING', 'PREEMPTED', 0), ('RUNNING', 'FAILED', 1)]:
            self.assertTrue(corpus_continue.allow_resume(holder, coordinator, restarts))

    def test_computation_failure_and_explicit_cancel_stay_blocked(self):
        for holder, coordinator in [('RUNNING', 'FAILED'), ('CANCELLED', 'FAILED'), ('RUNNING', 'CANCELLED'), ('TIMEOUT', 'CANCELLED')]:
            self.assertFalse(corpus_continue.allow_resume(holder, coordinator, 0))
        self.assertFalse(corpus_continue.allow_resume('CANCELLED', 'FAILED', 2))


if __name__ == '__main__':
    unittest.main()
