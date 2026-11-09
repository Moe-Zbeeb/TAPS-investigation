import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from taps_rebuild import verification


class FiniteVocabularyVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / 'project'
        environment = patch.dict(os.environ, {'SLURM_JOB_ID': 'test-allocation', 'CUDA_VISIBLE_DEVICES': ''})
        environment.start()
        self.addCleanup(environment.stop)
        self.rule_contents = b'def evaluate_posterior(logits, candidates, logits_processor):\n    return logits, candidates, logits_processor\n'

    def evidence(self):
        directory = self.root / 'repos/EAGLE'
        rule_file = directory / 'eagle/model/utils.py'
        rule_file.parent.mkdir(parents=True, exist_ok=True)
        rule_file.write_bytes(self.rule_contents)
        return {'name': 'EAGLE', 'revision': verification.METHODS['eagle3'][2], 'directory': directory, 'module_name': 'eagle.model.ea_model', 'module_file': rule_file.with_name('ea_model.py'), 'rule_file': rule_file, 'rule': verification.extract_rule_source(self.rule_contents), 'source_identity_sha256': 'a' * 64, 'sources_receipt_sha256': 'b' * 64, 'source_registry_sha256': 'c' * 64}

    def source_fixture(self):
        evidence = self.evidence()
        evidence['module_file'].write_bytes(b'from .utils import evaluate_posterior\n')
        registry_contents = (PROJECT / 'configs/sources.json').read_bytes()
        registry_path = self.root / 'configs/sources.json'
        registry_path.parent.mkdir(parents=True)
        registry_path.write_bytes(registry_contents)
        builder = self.root / 'src/taps_rebuild/sources.py'
        builder.parent.mkdir(parents=True)
        builder.write_bytes(b'fixture-builder')
        tracked = {}
        for path in (evidence['module_file'], evidence['rule_file']):
            tracked[str(path.relative_to(evidence['directory']))] = {'kind': 'regular_file', 'size_bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        identity = hashlib.sha256(json.dumps(tracked, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        receipt = {'status': 'complete', 'root': str(self.root), 'registry_sha256': hashlib.sha256(registry_contents).hexdigest(), 'source_sha256': hashlib.sha256(builder.read_bytes()).hexdigest(), 'sources': {'EAGLE': {'path': str(evidence['directory']), 'revision': verification.METHODS['eagle3'][2], 'tracked_files': tracked, 'source_identity_sha256': identity}}}
        path = self.root / 'outputs/preparation/sources.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(receipt))
        return evidence, tracked, identity

    def test_bound_uses_actual_counts_and_simultaneous_family(self):
        result = verification.summarize_counts([5000, 15000, 30000, 50000])
        expected = 2 * math.sqrt(math.log(32 / 0.05) / 200000)
        self.assertEqual(result['actual_trials'], 100000)
        self.assertEqual(result['empirical_tv'], 0)
        self.assertAlmostEqual(result['simultaneous_tv_upper_bound'], expected)
        self.assertTrue(result['equivalence_criterion_passed'])
        smaller_family = verification.summarize_counts([5000, 15000, 30000, 50000], case_count=1)
        self.assertGreater(result['simultaneous_tv_upper_bound'], smaller_family['simultaneous_tv_upper_bound'])

    def test_small_or_biased_samples_do_not_certify_margin(self):
        small = verification.summarize_counts([5, 15, 30, 50])
        self.assertEqual(small['empirical_tv'], 0)
        self.assertFalse(small['equivalence_criterion_passed'])
        biased = verification.summarize_counts([10000, 10000, 30000, 50000])
        self.assertAlmostEqual(biased['empirical_tv'], 0.05)
        self.assertFalse(biased['equivalence_criterion_passed'])
        for counts in ([0, 0, 0, 0], [-1, 1, 1, 1], [1, 2], [True, 1, 1, 1]):
            with self.subTest(counts=counts), self.assertRaises(verification.VerificationError):
                verification.summarize_counts(counts)

    def test_duplicate_paths_reach_dut_and_accepted_child_uses_selected_index(self):
        paths = verification.candidate_paths((0, 0, 1))
        self.assertEqual(paths, [[0, 0], [0, 0], [0, 1]])
        fallback = Mock()
        self.assertEqual(verification.interpret_outcome((2, 1, object()), paths, fallback), (1, True))
        self.assertEqual(verification.interpret_outcome((1, 1, object()), paths, fallback), (0, True))
        fallback.assert_not_called()

    def test_fallback_receives_exact_official_remaining_distribution(self):
        remaining = object()
        fallback = Mock(return_value=3)
        result = verification.interpret_outcome((0, 0, remaining), [[0, 0]], fallback)
        self.assertEqual(result, (3, False))
        fallback.assert_called_once_with(remaining)
        with self.assertRaises(verification.VerificationError):
            verification.interpret_outcome((0, 2, remaining), [[0, 0]], fallback)
        with self.assertRaises(verification.VerificationError):
            verification.interpret_outcome((0, 0, remaining), [[0, 0]], Mock(return_value=4))

    def test_non_none_processor_preserves_exact_logits(self):
        logits = object()
        self.assertIs(verification.identity_logits_processor(None, logits), logits)

    def test_rule_source_is_extracted_without_execution_or_replacement(self):
        contents = b'raise RuntimeError("do not execute")\n' + self.rule_contents
        rule = verification.extract_rule_source(contents)
        self.assertEqual(rule['parameters'], ['logits', 'candidates', 'logits_processor'])
        self.assertEqual(rule['start_line'], 2)
        self.assertEqual(rule['sha256'], hashlib.sha256(rule['text'].encode()).hexdigest())
        for source in (b'def other(): pass\n', self.rule_contents * 2, b'def evaluate_posterior(logits): pass\n'):
            with self.subTest(source=source), self.assertRaises(verification.VerificationError):
                verification.extract_rule_source(source)

    def test_receipt_commit_builder_and_actual_source_hashes_are_validated(self):
        evidence, tracked, identity = self.source_fixture()
        with patch.object(verification.sources, 'inspect_checkout') as checkout, patch.object(verification.sources, 'tracked_manifest', return_value=(tracked, identity)):
            result = verification.validate_source(self.root, 'eagle3')
            self.assertEqual(result['rule']['sha256'], evidence['rule']['sha256'])
            checkout.assert_called_once()
            evidence['rule_file'].write_bytes(self.rule_contents + b'\n')
            with self.assertRaisesRegex(verification.VerificationError, 'source identity differs'):
                verification.validate_source(self.root, 'eagle3')

    def test_runtime_profile_is_checked_before_loading_official_module(self):
        evidence = self.evidence()
        with patch.object(verification.importlib.metadata, 'version', return_value='wrong'), patch.object(verification.importlib, 'import_module') as import_module:
            with self.assertRaisesRegex(verification.VerificationError, 'appropriate to the method'):
                verification.load_actual_posterior(evidence, 'hass')
            import_module.assert_not_called()

    def test_audit_wires_four_cases_without_loading_weights_and_never_passes_pipeline(self):
        evidence = self.evidence()
        torch = object()
        posterior = Mock()

        def case(torch_module, function, order, trials, seed):
            self.assertIs(torch_module, torch)
            self.assertIs(function, posterior)
            return dict(verification.summarize_counts([5000, 15000, 30000, 50000]), child_order=list(order), candidate_paths=verification.candidate_paths(order), seed=seed)

        with patch.object(verification, 'validate_source', return_value=evidence) as validated, patch.object(verification, 'load_actual_posterior', return_value=(torch, posterior, {'torch': 'mock', 'transformers': 'mock'})), patch.object(verification, 'run_case', side_effect=case) as run_case:
            receipt = verification.run_audit(self.root, 'eagle3')
        self.assertEqual(validated.call_count, 2)
        self.assertEqual(run_case.call_count, 4)
        posterior.assert_not_called()
        self.assertEqual(receipt['actual_total_trials'], 400000)
        self.assertTrue(receipt['equivalence_criterion_passed'])
        self.assertFalse(receipt['automatic_pipeline_gate_pass'])
        self.assertFalse(receipt['full_model_output_distribution_proof'])
        self.assertTrue(receipt['full_model_temperature1_validation_still_required'])
        self.assertFalse(receipt['models_loaded'])
        self.assertEqual(receipt['cases'][2]['child_order'], [0, 0, 1])
        self.assertEqual([case['seed'] for case in receipt['cases']], list(range(20261109, 20261113)))
        path = self.root / 'outputs/verification/eagle3-finite-vocab.json'
        self.assertEqual(json.loads(path.read_text()), receipt)
        self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_runtime_failure_is_recorded_without_external_exception_details(self):
        evidence = self.evidence()
        with patch.object(verification, 'validate_source', return_value=evidence), patch.object(verification, 'load_actual_posterior', side_effect=RuntimeError('unreported external detail')):
            with self.assertRaises(RuntimeError):
                verification.run_audit(self.root, 'eagle3', trials=10)
        path = self.root / 'outputs/verification/eagle3-finite-vocab.json'
        receipt = json.loads(path.read_text())
        self.assertEqual(receipt['status'], 'failed')
        self.assertNotIn('unreported external detail', path.read_text())

    def test_cpu_guard_and_bounded_trial_validation_precede_io(self):
        for environment in ({}, {'SLURM_JOB_ID': 'job'}, {'SLURM_JOB_ID': 'job', 'CUDA_VISIBLE_DEVICES': '0'}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(verification.VerificationError):
                    verification.run_audit(self.root, 'eagle3')
        for trials in (0, verification.MAX_TRIALS + 1):
            with self.assertRaises(verification.VerificationError):
                verification.run_audit(self.root, 'eagle3', trials=trials)
        self.assertFalse(self.root.exists())


if __name__ == '__main__':
    unittest.main()
