import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from taps_rebuild import native_full_body_bridge as bridge


class NativeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name).resolve()
        self.root = base / 'project'
        self.root.mkdir()
        self.python = base / 'runtime/python3'
        self.python.parent.mkdir()
        self.python.write_bytes(b'observed interpreter bytes')
        alias = self.root / 'envs/native-eagle3/bin/python'
        alias.parent.mkdir(parents=True)
        alias.symlink_to(self.python)
        self.artifact = self.root / 'artifact'
        self.artifact.write_bytes(b'unchanged qualified artifact')

    def descriptor(self, path):
        return {'path': str(path), 'sha256': bridge.native.digest(path)}

    def bundle(self, interpreter=None, extra=None):
        interpreter = interpreter or {'path': str(self.python), 'sha256': bridge.native.digest(self.python), 'size_bytes': self.python.stat().st_size}
        adapter = {'sha256': bridge.native.digest(bridge.__file__), 'runtime_interpreter': interpreter}
        files = {str(self.artifact): {'sha256': bridge.native.digest(self.artifact), 'size_bytes': self.artifact.stat().st_size}, interpreter['path']: {key: interpreter[key] for key in ('sha256', 'size_bytes')}}
        if extra:
            files.update(extra)
        reports = {}
        for name in ('qualification', 'profile'):
            path = self.root / (name + '.json')
            bridge.native.atomic_json(path, {'admission_bridge': adapter, 'evidence_files': files})
            reports[name] = self.descriptor(path)
        review = self.root / 'review.json'
        bridge.native.atomic_json(review, {'scientific_training_admitted': True, 'admission_bridge': adapter})
        approval = self.root / 'approval.json'
        bridge.native.atomic_json(approval, {**reports, 'admission_bridge': adapter, 'independent_review': self.descriptor(review)})
        config = {'pipeline_stage': bridge.STAGE, 'runtime_python': 'envs/native-eagle3/bin/python', 'full_body_v3': {'admission_bridge': adapter, 'scientific_admission': self.descriptor(approval)}}
        return config, reports

    def test_original_inventory_retained_and_exact_file_validated_separately(self):
        config, reports = self.bundle()
        original = (self.root / 'qualification.json').read_bytes()
        def delegate(root, temporary_config):
            self.assertEqual(temporary_config['pipeline_stage'], 'corrected-full-v3')
            document = bridge.document(root, reports['qualification'], 'corrected qualification')
            self.assertEqual(set(document['evidence_files']), {str(self.artifact)})
            return {'scientific_training_authorized': True}
        with patch.object(bridge, 'BASE_GATE', side_effect=delegate):
            self.assertTrue(bridge.gate(self.root, config)['scientific_training_authorized'])
        self.assertEqual((self.root / 'qualification.json').read_bytes(), original)
        self.assertIsNone(bridge.ACTIVE.get())
        self.assertIn(str(self.python), bridge.document(self.root, reports['qualification'], 'corrected qualification')['evidence_files'])

    def test_rejects_v3_and_direct_core_rejects_v4(self):
        config, _ = self.bundle()
        with self.assertRaises(ValueError):
            bridge.gate(self.root, {**config, 'pipeline_stage': 'corrected-full-v3'})
        with self.assertRaises(ValueError):
            bridge.BASE_GATE(self.root, config)

    def test_changed_interpreter_bytes_rejected(self):
        config, _ = self.bundle()
        self.python.write_bytes(b'x' * self.python.stat().st_size)
        with self.assertRaises(ValueError):
            bridge.gate(self.root, config)

    def test_original_report_sha_is_checked_before_temporary_view(self):
        config, _ = self.bundle()
        path = self.root / 'qualification.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            bridge.gate(self.root, config)

    def test_sibling_path_and_additional_external_artifact_rejected(self):
        sibling = self.python.with_name('sibling')
        sibling.write_bytes(self.python.read_bytes())
        value = {'path': str(sibling), 'sha256': bridge.native.digest(sibling), 'size_bytes': sibling.stat().st_size}
        config, _ = self.bundle(value)
        with self.assertRaises(ValueError):
            bridge.gate(self.root, config)
        config, _ = self.bundle(extra={str(sibling): {key: value[key] for key in ('sha256', 'size_bytes')}})
        with self.assertRaises(ValueError):
            bridge.gate(self.root, config)

    def test_missing_or_mismatched_bridge_binding_rejected(self):
        config, _ = self.bundle()
        missing = copy.deepcopy(config)
        del missing['full_body_v3']['admission_bridge']
        bad = copy.deepcopy(config)
        bad['full_body_v3']['admission_bridge']['sha256'] = '0' * 64
        for value in (missing, bad):
            with self.assertRaises(ValueError):
                bridge.gate(self.root, value)

    def test_context_reset_after_delegate_failure(self):
        config, _ = self.bundle()
        with patch.object(bridge, 'BASE_GATE', side_effect=RuntimeError('delegate failure')):
            with self.assertRaises(RuntimeError):
                bridge.gate(self.root, config)
        self.assertIsNone(bridge.ACTIVE.get())

    def test_audit_subprocess_routes_to_frozen_bridge(self):
        delegate = Mock()
        command = ['/python', '-B', '/run/inputs/native_training.py', 'audit', '--plan', '/run/plan.json']
        bridge.AuditRouter(delegate).run(command, check=True)
        delegate.run.assert_called_once_with(['/python', '-B', '/run/inputs/native_full_body_bridge.py', 'audit', '--plan', '/run/plan.json'], check=True)
        self.assertEqual(command[2], '/run/inputs/native_training.py')


if __name__ == '__main__':
    unittest.main()
