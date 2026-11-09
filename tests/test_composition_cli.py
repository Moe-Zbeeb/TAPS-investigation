import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from taps_rebuild import composition_cli as cli


def rows(split='dev', prefix='dev'):
    return [{'example_id': prefix + str(index), 'log_probability': math.log(probability), 'accepted': accepted, 'split': split, 'label_source': 'target_verification'} for index, (probability, accepted) in enumerate(((0.1, 0), (0.3, 0), (0.6, 1), (0.9, 1)))]


def fit_config(expert='a'):
    return {'scope': 'engineering', 'operation': 'fit-calibrator', 'expert_id': expert, 'method': 'isotonic', 'examples': {'inline': rows()}}


def merge_config(method='linear'):
    ancestor = {'w': [0.0, 0.0, 0.0, 0.0], 'step': 0}
    checksum = hashlib.sha256(cli.canonical(ancestor)).hexdigest()
    return {'scope': 'engineering', 'operation': 'merge-weights', 'method': method, 'ancestor': {'inline': ancestor}, 'experts': {'a': {'inline': {'w': [1.0, 2.0, 3.0, 4.0], 'step': 0}}, 'b': {'inline': {'w': [2.0, 4.0, 6.0, 8.0], 'step': 0}}}, 'expert_ancestors': {'a': checksum, 'b': checksum}, 'density': 1.0, 'seed': 3}


class CompositionCLITests(unittest.TestCase):
    def test_fit_saved_calibrator_pool_and_heldout_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved = cli.execute('fit-calibrator', fit_config(), root)
            self.assertFalse(saved['scientific_result'])
            output = root / 'outputs/composition/calibrator-a.json'
            cli.write_result(root, output, saved)
            calibrated = {'path': str(output), 'sha256': cli.digest(output)}
            pool = {'scope': 'engineering', 'operation': 'pool-tree', 'root_token': 9, 'root_source': 'target_greedy', 'budget': 3, 'score_mode': 'estimated', 'calibrators': {'a': calibrated}, 'drafts': {'inline': [{'expert_id': 'a', 'root_token': 9, 'root_source': 'target_greedy', 'nodes': [{'path': [1], 'log_probability': math.log(.6)}, {'path': [1, 2], 'log_probability': math.log(.3)}]}]}}
            result = cli.execute('pool-tree', pool, root)
            self.assertEqual(result['tree']['parent_indices'], [-1, 0, 1])
            self.assertFalse(result['tree']['metadata']['estimated_optimality_claimed'])
            self.assertIn('calibrator:a', result['provenance']['inputs'])
            evaluation = {'scope': 'engineering', 'calibrator': calibrated, 'examples': {'inline': rows('test', 'test')}, 'split': 'test'}
            diagnostics = cli.execute('evaluate-calibrator', evaluation, root)['diagnostics']
            self.assertEqual(diagnostics['count'], 4)
            self.assertFalse(diagnostics['overlaps_fitting_examples'])
            evaluation['examples'] = {'inline': rows('test', 'dev')}
            with self.assertRaisesRegex(ValueError, 'overlap'):
                cli.execute('evaluate-calibrator', evaluation, root)

    def test_all_four_merge_methods_keep_buffers_and_ancestor_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            for method in ('linear', 'task_arithmetic', 'ties', 'dare'):
                with self.subTest(method=method):
                    config = merge_config(method)
                    result = cli.execute('merge-weights', config, temporary)
                    self.assertEqual(result['state_dict']['step'], 0)
                    self.assertTrue(result['merge_metadata']['shared_ancestor'])
                    if method == 'linear':
                        self.assertEqual(result['state_dict']['w'], [1.5, 3.0, 4.5, 6.0])
                    else:
                        expected = [1.5, 3.0, 4.5, 6.0] if method == 'ties' else [3.0, 6.0, 9.0, 12.0]
                        self.assertEqual(result['state_dict']['w'], expected)
                    path = Path(temporary) / ('models/composed/' + method + '.json')
                    cli.write_result(temporary, path, result)
                    self.assertEqual(json.loads(path.read_text())['merge_metadata']['method'], method)

    def test_explicit_ancestor_and_checkpoint_metadata_cannot_disagree(self):
        config = merge_config()
        config['ancestor_identity'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Ancestor identity'):
            cli.execute('merge-weights', config, '.')
        config = merge_config()
        state = config['experts']['a']['inline']
        config['experts']['a']['inline'] = {'state_dict': state, 'metadata': {'shared_ancestor_identity_sha256': '0' * 64}}
        with self.assertRaisesRegex(ValueError, 'ancestor metadata'):
            cli.execute('merge-weights', config, '.')

    def test_file_sha_and_dataset_identity_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'verification.jsonl'
            path.write_bytes(b''.join(cli.canonical(row) + b'\n' for row in rows()))
            config = fit_config()
            config['examples'] = {'path': str(path), 'sha256': cli.digest(path)}
            result = cli.execute('fit-calibrator', config, root)
            self.assertEqual(result['calibrator']['dataset_identity'], cli.digest(path))
            path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                cli.execute('fit-calibrator', config, root)
            config = fit_config()
            config['dataset_identity'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'dataset_identity'):
                cli.execute('fit-calibrator', config, root)

    def test_output_is_scoped_atomic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = cli.execute('fit-calibrator', fit_config(), root)
            output = root / 'outputs/composition/fit.json'
            receipt = cli.write_result(root, output, result)
            original = output.read_bytes()
            self.assertEqual(receipt['sha256'], cli.digest(output))
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                cli.write_result(root, output, result)
            self.assertEqual(output.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'remain under'):
                cli.write_result(root, root / 'models/input.json', result)
            outside = root / 'outside'
            outside.mkdir()
            (root / 'outputs/composition/alias').symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'remain under'):
                cli.write_result(root, root / 'outputs/composition/alias/result.json', result)

    def test_tensor_file_loading_is_slurm_guarded_before_import(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            root = Path(temporary)
            path = root / 'weights.pt'
            path.write_bytes(b'not loaded outside Slurm')
            with self.assertRaisesRegex(ValueError, 'requires Slurm'):
                cli.read_input(root, {'path': str(path)}, {}, 'weights', 'engineering')

    def test_research_is_ineligible_without_all_official_reproduction(self):
        config = fit_config()
        config['scope'] = 'research'
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'Slurm'):
            cli.execute('fit-calibrator', config, '.')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '1'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'reproduction report'):
                cli.execute('fit-calibrator', config, '.')
            config['reproduction_report'] = {'inline': {'reproduction_gate': {'status': 'passed', 'passed': True}, 'cells': [{'method': 'eagle3', 'reference': {}, 'reproduction_gate': {'passed': True}}]}}
            with self.assertRaisesRegex(ValueError, 'All three'):
                cli.execute('fit-calibrator', config, '.')

    def test_cli_json_receipt_and_failure_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'config.json'
            config.write_text(json.dumps(fit_config()))
            argv = ['fit-calibrator', '--root', str(root), '--config', str(config), '--output', 'outputs/composition/calibrator.json']
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(cli.main(argv), 0)
            self.assertEqual(json.loads(output.getvalue())['status'], 'complete')
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(argv), 1)
            wrong = copy.deepcopy(fit_config())
            wrong['operation'] = 'pool-tree'
            with self.assertRaisesRegex(ValueError, 'operation differs'):
                cli.execute('fit-calibrator', wrong, root)

    def test_engineering_example_configs_run_in_documented_order(self):
        project = Path(__file__).resolve().parents[1]
        operations = [('fit-calibrator', 'calibrator-engineering-a.json'), ('evaluate-calibrator', 'calibration-engineering-a.json'), ('pool-tree', 'tree-engineering.json'), ('merge-weights', 'merge-engineering.json')]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            storage = root / 'shared-models'
            storage.mkdir()
            (root / 'models').symlink_to(storage)
            for operation, name in operations:
                config = json.loads((project / 'configs/composition' / (operation + '-engineering.json')).read_text())
                result = cli.execute(operation, config, root)
                relative = ('models/composed/' if operation == 'merge-weights' else 'outputs/composition/') + name
                receipt = cli.write_result(root, relative, result)
                self.assertEqual(receipt['status'], 'complete')
                self.assertFalse(result['scientific_result'])


if __name__ == '__main__':
    unittest.main()
