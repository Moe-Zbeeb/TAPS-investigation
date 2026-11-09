import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from taps_rebuild.assets import PreparationError, prepare_assets, validate_registry, verify_file


def expected_file(name, content, lfs=False):
    result = {'path': name, 'size_bytes': len(content), 'git_blob_id': hashlib.sha1(('blob ' + str(len(content)) + '\0').encode() + content).hexdigest()}
    if lfs:
        result['lfs_sha256'] = hashlib.sha256(content).hexdigest()
    return result


class AssetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / 'project'
        self.registry_path = self.base / 'registry.json'
        self.payloads = {'config.json': b'{"model_type":"llama"}\n', 'pytorch_model.bin': b'small-test-weights'}
        self.asset = {'id': 'test_model', 'model_path': 'models/test-model', 'source': {'repo_id': 'public/test-model', 'revision': 'a' * 40}, 'files': [expected_file(name, content, name.endswith('.bin')) for name, content in self.payloads.items()]}
        self.registry = {'schema_version': 1, 'huggingface_hub_version': '0.33.4', 'assets': [self.asset]}
        self.environment = patch.dict(os.environ, {'SLURM_JOB_ID': 'test-allocation', 'CUDA_VISIBLE_DEVICES': ''})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def write_registry(self):
        self.registry_path.write_text(json.dumps(self.registry))

    def download(self, **kwargs):
        directory = Path(kwargs['local_dir'])
        for name in kwargs['allow_patterns']:
            (directory / name).write_bytes(self.payloads[name])
        return str(directory)

    def run_preparation(self, downloader=None):
        self.write_registry()
        return prepare_assets(self.root, self.registry_path, downloader=downloader or self.download, hub_version='0.33.4')

    def legacy_reuse(self):
        directory = self.base / 'legacy-model'
        directory.mkdir()
        for name, content in self.payloads.items():
            (directory / name).write_bytes(content)
        for item in self.asset['files']:
            item['sha256'] = hashlib.sha256(self.payloads[item['path']]).hexdigest()
        old = {'status': 'complete', 'models': {'old_model': {'repo_id': self.asset['source']['repo_id'], 'revision': self.asset['source']['revision'], 'path': str(directory), 'files': {item['path']: {'sha256': item['sha256'], 'size_bytes': item['size_bytes']} for item in self.asset['files']}}}}
        path = self.base / 'original-receipt.json'
        path.write_text(json.dumps(old))
        self.asset['reuse'] = {'path': str(directory), 'receipt_path': str(path), 'receipt_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'receipt_model_key': 'old_model'}
        return directory, path

    def test_hashes_git_blobs_and_lfs_payloads_and_records_sha256(self):
        result = self.run_preparation()
        model = result['models']['test_model']
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['gpu_execution_performed'])
        self.assertFalse(result['scientific_results'])
        for name, payload in self.payloads.items():
            self.assertEqual(model['files'][name]['sha256'], hashlib.sha256(payload).hexdigest())
        self.assertEqual(model['files']['config.json']['verification_method'], 'git_blob_id')
        self.assertEqual(model['files']['pytorch_model.bin']['verification_method'], 'lfs_sha256')
        receipt = self.root / 'outputs/preparation/assets.json'
        self.assertEqual(json.loads(receipt.read_text()), result)
        self.assertEqual(list(receipt.parent.glob('assets.json.*.tmp')), [])

    def test_downloads_only_missing_files_and_resumes_without_overwriting(self):
        directory = self.root / self.asset['model_path']
        directory.mkdir(parents=True)
        config = directory / 'config.json'
        config.write_bytes(self.payloads['config.json'])
        stamp = config.stat().st_mtime_ns
        downloader = Mock(side_effect=self.download)
        first = self.run_preparation(downloader)
        options = downloader.call_args.kwargs
        self.assertEqual(options['revision'], 'a' * 40)
        self.assertIs(options['token'], False)
        self.assertEqual(options['allow_patterns'], ['pytorch_model.bin'])
        self.assertEqual(config.stat().st_mtime_ns, stamp)
        downloader.reset_mock()
        second = self.run_preparation(downloader)
        downloader.assert_not_called()
        self.assertEqual(first['models']['test_model']['model_identity_sha256'], second['models']['test_model']['model_identity_sha256'])

    def test_corrupt_existing_payload_is_preserved_and_download_not_called(self):
        directory = self.root / self.asset['model_path']
        directory.mkdir(parents=True)
        path = directory / 'pytorch_model.bin'
        wrong = b'x' * len(self.payloads[path.name])
        path.write_bytes(wrong)
        downloader = Mock()
        with self.assertRaisesRegex(PreparationError, 'content identity differs'):
            self.run_preparation(downloader)
        downloader.assert_not_called()
        self.assertEqual(path.read_bytes(), wrong)
        receipt = json.loads((self.root / 'outputs/preparation/assets.json').read_text())
        self.assertEqual(receipt['status'], 'failed')

    def test_reuses_verified_original_files_without_download_or_modification(self):
        directory, receipt = self.legacy_reuse()
        before = {name: (directory / name).stat().st_mtime_ns for name in self.payloads}
        downloader = Mock()
        result = self.run_preparation(downloader)
        downloader.assert_not_called()
        alias = self.root / self.asset['model_path']
        self.assertTrue(alias.is_symlink())
        self.assertEqual(alias.resolve(), directory.resolve())
        self.assertTrue(result['models']['test_model']['reused'])
        self.assertEqual(before, {name: (directory / name).stat().st_mtime_ns for name in self.payloads})
        (directory / 'pytorch_model.bin').write_bytes(b'x' * len(self.payloads['pytorch_model.bin']))
        with self.assertRaisesRegex(PreparationError, 'content identity differs'):
            self.run_preparation(downloader)
        self.assertTrue(alias.is_symlink())
        self.assertTrue(receipt.is_file())

    def test_changed_original_receipt_rejected_before_creating_alias(self):
        directory, receipt = self.legacy_reuse()
        receipt.write_text(receipt.read_text() + '\n')
        with self.assertRaisesRegex(PreparationError, 'receipt content differs'):
            self.run_preparation(Mock())
        self.assertFalse((self.root / self.asset['model_path']).exists())
        self.assertEqual((directory / 'pytorch_model.bin').read_bytes(), self.payloads['pytorch_model.bin'])

    def test_allocation_and_empty_cuda_guards_run_before_io(self):
        for environment in [{}, {'SLURM_JOB_ID': 'job'}, {'SLURM_JOB_ID': 'job', 'CUDA_VISIBLE_DEVICES': '0'}]:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(PreparationError):
                    prepare_assets(self.root, self.registry_path, downloader=Mock(), hub_version='0.33.4')
        self.assertFalse(self.root.exists())

    def test_rejects_wrong_hub_version(self):
        self.write_registry()
        with self.assertRaisesRegex(PreparationError, 'huggingface-hub==0.33.4'):
            prepare_assets(self.root, self.registry_path, downloader=Mock(), hub_version='1.0.0')
        self.assertFalse(self.root.exists())

    def test_registry_rejects_unpinned_or_escaping_paths(self):
        mutations = [lambda a: a.update(model_path='../escape'), lambda a: a.update(model_path='/absolute/model'), lambda a: a['source'].update(revision='main'), lambda a: a['files'][0].update(path='../config.json'), lambda a: a['files'][0].update(path='*.json')]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                registry = copy.deepcopy(self.registry)
                mutation(registry['assets'][0])
                with self.assertRaises(PreparationError):
                    validate_registry(registry)
        registry = copy.deepcopy(self.registry)
        second = copy.deepcopy(registry['assets'][0])
        second.update(id='second_model', model_path='models/test-model/inside')
        registry['assets'].append(second)
        with self.assertRaisesRegex(PreparationError, 'must not overlap'):
            validate_registry(registry)

    def test_rejects_unregistered_contents_and_weight_symlinks(self):
        directory = self.root / self.asset['model_path']
        directory.mkdir(parents=True)
        extra = directory / 'unregistered.bin'
        extra.write_bytes(b'keep')
        with self.assertRaisesRegex(PreparationError, 'Unexpected files'):
            self.run_preparation(Mock())
        self.assertEqual(extra.read_bytes(), b'keep')
        outside = self.base / 'payload'
        outside.write_bytes(self.payloads['pytorch_model.bin'])
        link = self.base / 'link'
        link.symlink_to(outside)
        with self.assertRaisesRegex(PreparationError, 'file type or size'):
            verify_file(link, self.asset['files'][1])

    def test_rejects_wrong_downloader_destination(self):
        with self.assertRaisesRegex(PreparationError, 'Download destination differs'):
            self.run_preparation(Mock(return_value=str(self.base / 'elsewhere')))

    def test_rechecks_existing_files_after_snapshot_finishes(self):
        directory = self.root / self.asset['model_path']
        directory.mkdir(parents=True)
        config = directory / 'config.json'
        config.write_bytes(self.payloads['config.json'])

        def changed_download(**kwargs):
            result = self.download(**kwargs)
            config.write_bytes(b'x' * len(self.payloads['config.json']))
            return result

        with self.assertRaisesRegex(PreparationError, 'content identity differs'):
            self.run_preparation(changed_download)

    def test_real_registry_pins_five_assets_and_metadata_evidence(self):
        registry = json.loads((PROJECT / 'configs/assets.json').read_text())
        assets = validate_registry(registry)
        self.assertEqual({asset['id'] for asset in assets}, {'llama31_instruct_8b', 'eagle3_llama31_8b', 'llama3_instruct_8b', 'eagle2_llama3_8b', 'hass_llama3_8b'})
        self.assertEqual(sum('reuse' in asset for asset in assets), 2)
        self.assertFalse(registry['collection_destination']['publish_automatically'])
        provider = next(asset for asset in assets if asset['id'] == 'llama31_instruct_8b')
        self.assertEqual(provider['source']['revision'], 'e63e9c4ae3b9074207cd49331b36a7d998eba86a')
        self.assertEqual(len(provider['files']), 12)
        self.assertEqual(sum(item['size_bytes'] for item in provider['files']), 16069734987)
        proof = json.loads((PROJECT / 'references/metadata/llama31-provider-verification.json').read_text())
        expected = {item['official']['rfilename']: item['official'] for item in proof['comparisons']}
        for item in provider['files']:
            original = expected[item['path']]
            self.assertEqual(item['size_bytes'], original['size'])
            self.assertEqual(item['git_blob_id'], original['blobId'])
            self.assertEqual(item.get('lfs_sha256'), original.get('lfs', {}).get('sha256'))
        for name, expected in registry['metadata_evidence'].items():
            self.assertEqual(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
