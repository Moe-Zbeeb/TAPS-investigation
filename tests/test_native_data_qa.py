import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from taps_rebuild import native_data_qa as qa


class NativeDataTests(unittest.TestCase):
    def test_vocabulary_builder_context_has_only_native_cpu_preparation_attributes(self):
        context = qa.VocabularyContext(128256, 32000)
        context.register_buffer('d2t', 'fixture')
        self.assertEqual(context.train_config.max_len, 2048)
        self.assertEqual(context.draft_vocab_size, 32000)
        self.assertEqual(context.d2t, 'fixture')
        self.assertFalse(hasattr(context, 'target_model'))

    def test_vocabulary_cache_cannot_be_added_to_an_already_launched_plan(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'SLURM_JOB_ID': '1', 'CUDA_VISIBLE_DEVICES': ''}):
            root = Path(directory)
            (root / 'work').mkdir()
            (root / 'launch.json').write_text('{}')
            (root / 'report.json').write_text('{}')
            plan = {'workdir': str(root / 'work'), 'frozen_files': {}}
            with patch.object(qa.native, 'load_plan', return_value=plan), self.assertRaisesRegex(ValueError, 'fresh unlaunched'):
                qa.bind_vocabulary(root / 'plan.json', root / 'report.json')

    def test_engineering_vocabulary_cannot_be_substituted_for_real_training_data(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'SLURM_JOB_ID': '1', 'CUDA_VISIBLE_DEVICES': ''}):
            root = Path(directory)
            (root / 'work').mkdir()
            (root / 'report.json').write_text(json.dumps({'status': 'passed', 'scope': 'engineering_vocabulary_preparation'}))
            plan = {'workdir': str(root / 'work'), 'frozen_files': {}, 'target': {'model_identity_sha256': 'target'}, 'train': {'sha256': 'train'}}
            with patch.object(qa.native, 'load_plan', return_value=plan), self.assertRaisesRegex(ValueError, 'exact native training contract'):
                qa.bind_vocabulary(root / 'plan.json', root / 'report.json')
