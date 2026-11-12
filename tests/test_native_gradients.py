import unittest

from taps_rebuild.native_gradients import checkpoint_adapter


class CheckpointPolicyTests(unittest.TestCase):
    def test_nested_outputs_use_explicit_nonreentrant_policy(self):
        calls = []
        def checkpoint(function, *args, **options):
            calls.append(options)
            return function(*args)
        result = checkpoint_adapter(checkpoint, 'non_reentrant')(lambda x: ((x, x + 1), [[x]]), 3)
        self.assertEqual(result, ((3, 4), [[3]]))
        self.assertEqual(calls, [{'use_reentrant': False}])

    def test_reference_executes_the_function_without_checkpointing(self):
        def checkpoint(*args, **options):
            self.fail('Direct native reference entered checkpoint machinery')
        self.assertEqual(checkpoint_adapter(checkpoint, 'off_reference')(lambda x: ((x,), [[x]]), 2), ((2,), [[2]]))

    def test_reentrant_or_unknown_policy_cannot_be_selected(self):
        with self.assertRaises(ValueError):
            checkpoint_adapter(lambda: None, 'legacy_default')
        with self.assertRaises(ValueError):
            checkpoint_adapter(lambda: None, 'non_reentrant')(lambda: None, use_reentrant=True)
