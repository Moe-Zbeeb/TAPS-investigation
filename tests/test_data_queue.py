import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from taps_rebuild import data_queue


class DataQueueTests(unittest.TestCase):
    def test_attached_preparation_hides_cuda_then_restores_all_devices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            def prepare(root, config, corpus, run_id, limit=None):
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
                return {'directory': str(root / run_id)}
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}), patch.object(data_queue.pool, 'allocation'), patch.object(data_queue.corpus_generate, 'prepare', side_effect=prepare), patch.object(data_queue, 'gpu', return_value={'status': 'complete'}) as gpu:
                self.assertEqual(data_queue.attached(root, 'config', 'corpus', 'run'), {'status': 'complete'})
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
                gpu.assert_called_once_with(root, root / 'run-smoke/plan.json', root / 'run/plan.json')

    def test_preparation_failure_restores_cuda_and_never_starts_generation(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}), patch.object(data_queue.pool, 'allocation'), patch.object(data_queue.corpus_generate, 'prepare', side_effect=ValueError('changed corpus')), patch.object(data_queue, 'gpu') as gpu:
            with self.assertRaisesRegex(ValueError, 'changed corpus'):
                data_queue.attached('.', 'config', 'corpus', 'run')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
            gpu.assert_not_called()


if __name__ == '__main__':
    unittest.main()
