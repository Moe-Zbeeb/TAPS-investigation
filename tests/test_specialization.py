import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from taps_rebuild import specialization as rq1


def write(root, name, value, jsonl=False):
    path = root / name
    contents = b''.join(rq1.canonical(row) + b'\n' for row in value) if jsonl else rq1.canonical(value)
    path.write_bytes(contents)
    return {'path': str(path), 'sha256': rq1.digest(path)}


def fixture(root, repeats=1):
    target, domains = 'a' * 64, ['chat', 'math']
    budget = {'max_new_tokens': 10, 'verifier_nodes': 10, 'temperature': 0.0, 'max_length': 128, 'batch_size': 1}
    panel = [{'domain': domain, 'question_id': domain + '-q' + str(question), 'turn_id': turn, 'repeat': repeat, 'seed': 1000 + turn * 17 + repeat, 'prompt_id': domain + '-prompt-' + str(question), 'prompt_sha256': rq1.identity([domain, question, turn])} for domain in domains for question in range(2) for repeat in range(repeats) for turn in range(2)]
    manifest = {'schema_version': 1, 'scope': 'engineering', 'data_origin': 'synthetic_fixture', 'target_identity_sha256': target, 'domains': domains, 'evaluation_budget': budget, 'minimum_training_seeds': 3, 'relative_compute_tolerance': .01, 'traffic_weights': {'chat': .25, 'math': .75}, 'panel': write(root, 'panel.jsonl', panel, True), 'drafters': []}
    for seed in (17, 29, 43):
        for role, domain in [('generalist', None), ('specialist', 'chat'), ('specialist', 'math')]:
            identifier = (domain or 'generalist') + '-seed-' + str(seed)
            checkpoint = rq1.identity(identifier)
            evidence = {'status': 'complete', 'scope': 'research', 'checkpoint_sha256': checkpoint, 'target_identity_sha256': target, 'training_seed': seed, 'domains': domains if role == 'generalist' else [domain], 'response_regime': 'target_generated', 'architecture_identity_sha256': 'b' * 64, 'observed': {'loss_tokens': 1000, 'optimizer_steps': 10}, 'declared': {'loss_tokens': 1000, 'optimizer_steps': 10}, 'compute': {'kind': 'measured_flops', 'value': 1000000, 'measurement_protocol_sha256': 'c' * 64, 'measurement_scope': 'training_forward_backward_optimizer'}}
            records = []
            for case in panel:
                cycle_count = (1 if case['turn_id'] == 0 else 10) if role == 'generalist' else (5 if case['domain'] == domain else 1)
                emitted = 10 // cycle_count
                cycles = [{'accepted_draft_tokens': emitted - 1, 'emitted_tokens': emitted, 'drafting_ms': .1, 'verification_ms': .2, 'verifier_nodes': 10, 'proposed_nodes': 9} for _ in range(cycle_count)]
                records.append({**case, 'status': 'completed', 'warmup': False, 'scope': 'reference', 'generation_mode': 'speculative', 'target_identity_sha256': target, 'drafter_checkpoint_sha256': checkpoint, 'evaluation_budget': budget, 'cycle_count': cycle_count, 'cycles': cycles, 'new_token_count': 10, 'output_token_ids': [1, 2] + [0] * 8, 'comparison_token_ids': [1, 2], 'wall_ms': 10 + cycle_count, 'unused_final_drafting_ms': .3, 'raw_reported_idx': 999})
            manifest['drafters'].append({'id': identifier, 'role': role, 'domain': domain, 'training_seed': seed, 'checkpoint_sha256': checkpoint, 'target_identity_sha256': target, 'training_evidence': write(root, identifier + '-training.json', evidence), 'logs': [write(root, identifier + '.jsonl', records, True)]})
    return manifest


def replace_log(root, drafter, mutate):
    path = Path(drafter['logs'][0]['path'])
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    mutate(rows)
    drafter['logs'][0] = write(root, path.name, rows, True)


def replace_evidence(root, drafter, mutate):
    path = Path(drafter['training_evidence']['path'])
    value = json.loads(path.read_bytes())
    mutate(value)
    drafter['training_evidence'] = write(root, path.name, value)


class SpecializationTests(unittest.TestCase):
    def test_canonical_turn_pooling_cross_domain_matrix_and_fixed_domain_oracle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = rq1.analyze(fixture(root), root, bootstrap_samples=30, seed=9)
            self.assertEqual(len(report['cross_domain_matrix']), 18)
            baseline = [row for row in report['cross_domain_matrix'] if row['role'] == 'generalist']
            for row in baseline:
                self.assertAlmostEqual(row['question_mean_tokens_per_cycle'], 20 / 11)
                self.assertAlmostEqual(row['equal_turn_mean_tokens_per_cycle'], 5.5)
                self.assertEqual(row['totals']['comparison_tokens'], 8)
                self.assertEqual(row['totals']['tokens'], 40)
            self.assertAlmostEqual(report['specialization_gap']['generalist_tau'], 20 / 11)
            self.assertAlmostEqual(report['specialization_gap']['domain_oracle_tau'], 2)
            self.assertAlmostEqual(report['specialization_gap']['estimate'], 2 - 20 / 11)
            self.assertTrue(all(abs(value - (2 - 20 / 11)) < 1e-12 for value in report['specialization_gap']['ci95']))
            off_domain = [row['question_mean_tokens_per_cycle'] for row in report['cross_domain_matrix'] if row['role'] == 'specialist' and row['training_domain'] != row['evaluation_domain']]
            self.assertTrue(all(value == 10 for value in off_domain))
            self.assertFalse(report['scientific_comparison_eligible'])
            self.assertIn('engineering_or_synthetic_inputs_not_scientific_results', report['ineligibility_reasons'])
            self.assertTrue(all(row['eligible'] for row in report['quality_checks']['training_budget_pairs']))

    def test_repeats_are_averaged_within_question_and_per_turn_seeds_ignored(self):
        rows = [{'prompt_id': 'one', 'repeat': 0, 'seed': 1, 'new_token_count': 10, 'cycle_count': 1}, {'prompt_id': 'one', 'repeat': 0, 'seed': 2, 'new_token_count': 10, 'cycle_count': 10}, {'prompt_id': 'one', 'repeat': 1, 'seed': 3, 'new_token_count': 10, 'cycle_count': 2}, {'prompt_id': 'one', 'repeat': 1, 'seed': 4, 'new_token_count': 10, 'cycle_count': 10}]
        self.assertAlmostEqual(rq1._question_values(rows)['one'], (20 / 11 + 20 / 12) / 2)

    def test_missing_case_never_uses_complete_case_intersection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            replace_log(root, manifest['drafters'][1], lambda rows: rows.pop())
            report = rq1.analyze(manifest, root, bootstrap_samples=10)
            self.assertFalse(report['quality_checks']['all_evaluations_exactly_paired'])
            self.assertIsNone(report['specialization_gap'])
            self.assertEqual(report['per_domain_gaps'], [])
            self.assertTrue(any(not row['complete'] for row in report['cross_domain_matrix']))

    def test_exact_prompt_seed_target_and_evaluation_budget_checks(self):
        for field, value in [('prompt_sha256', 'd' * 64), ('seed', 9999), ('target_identity_sha256', 'd' * 64), ('evaluation_budget', {'max_new_tokens': 9})]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = fixture(root)
                replace_log(root, manifest['drafters'][1], lambda rows: rows[0].__setitem__(field, value))
                report = rq1.analyze(manifest, root, bootstrap_samples=5)
                self.assertIsNone(report['specialization_gap'])
                self.assertFalse(report['scientific_comparison_eligible'])

    def test_token_step_matching_without_measured_compute_is_ineligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            replace_evidence(root, manifest['drafters'][1], lambda value: value.pop('compute'))
            report = rq1.analyze(manifest, root, bootstrap_samples=10)
            self.assertTrue(report['quality_checks']['all_evaluations_exactly_paired'])
            self.assertIsNotNone(report['specialization_gap'])
            self.assertTrue(report['specialization_gap']['exploratory_only'])
            failed = [row for row in report['quality_checks']['training_budget_pairs'] if not row['eligible']]
            self.assertIn('measured_training_compute_missing', failed[0]['reasons'])

    def test_measured_gpu_time_requires_same_hardware_and_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            first = json.loads(Path(manifest['drafters'][0]['training_evidence']['path']).read_text())
            second = json.loads(Path(manifest['drafters'][1]['training_evidence']['path']).read_text())
            first['compute'] = {'kind': 'measured_gpu_seconds', 'value': 100, 'measurement_protocol_sha256': 'c' * 64, 'hardware_identity_sha256': 'd' * 64, 'measurement_scope': 'training_end_to_end'}
            second['compute'] = {**first['compute'], 'value': 101}
            arguments = {'target': 'a' * 64, 'seed': 17, 'domains': ['chat', 'math'], 'domain': 'chat', 'tolerance': .01}
            self.assertTrue(rq1._training_pair(first, second, **arguments)['eligible'])
            second['compute']['hardware_identity_sha256'] = 'e' * 64
            self.assertFalse(rq1._training_pair(first, second, **arguments)['eligible'])
            second['compute'] = {**first['compute'], 'value': 102}
            self.assertIn('measured_training_compute_outside_tolerance', rq1._training_pair(first, second, **arguments)['reasons'])

    def test_t0_mismatch_and_training_checkpoint_identity_are_ineligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            def mismatch(rows):
                rows[0]['output_token_ids'][0] = 7
                rows[0]['comparison_token_ids'][0] = 7
            replace_log(root, manifest['drafters'][1], mismatch)
            replace_evidence(root, manifest['drafters'][1], lambda value: value.__setitem__('checkpoint_sha256', 'f' * 64))
            report = rq1.analyze(manifest, root, bootstrap_samples=10)
            self.assertEqual(report['quality_checks']['t0_pairwise_sequence_mismatch_count'], 1)
            self.assertIn('t0_comparison_outputs_mismatch', report['ineligibility_reasons'])
            self.assertIn('training_evidence_checkpoint_identity_mismatch', report['ineligibility_reasons'])
            self.assertFalse(report['quality_checks']['target_only_correctness_established_by_this_report'])

    def test_malformed_off_by_one_duplicate_and_file_sha_fail(self):
        for failure in ('cycle_count', 'new_token_count', 'duplicate', 'sha'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = fixture(root)
                if failure == 'sha':
                    Path(manifest['drafters'][0]['logs'][0]['path']).write_bytes(b'changed')
                elif failure == 'duplicate':
                    replace_log(root, manifest['drafters'][0], lambda rows: rows.append(copy.deepcopy(rows[0])))
                else:
                    replace_log(root, manifest['drafters'][0], lambda rows: rows[0].__setitem__(failure, 999))
                with self.assertRaises(ValueError):
                    rq1.analyze(manifest, root, bootstrap_samples=5)

    def test_acceptance_and_proposals_cannot_exceed_actual_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            row = json.loads(Path(manifest['drafters'][0]['logs'][0]['path']).read_text().splitlines()[0])
            for mutation in ({'proposed_nodes': 8}, {'verifier_nodes': 9}, {'proposed_nodes': 10}):
                value = copy.deepcopy(row)
                value['cycles'][0].update(mutation)
                with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'verifier-node tree'):
                    rq1.validate_record(value)

    def test_missing_training_seed_role_and_warmup_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            def warmup(rows):
                value = copy.deepcopy(rows[0])
                value['warmup'] = True
                rows.append(value)
            replace_log(root, manifest['drafters'][0], warmup)
            report = rq1.analyze(manifest, root, bootstrap_samples=5)
            self.assertEqual(report['warmups_excluded'], 1)
            self.assertTrue(report['quality_checks']['all_evaluations_exactly_paired'])
            manifest['drafters'].pop()
            report = rq1.analyze(manifest, root, bootstrap_samples=5)
            self.assertIsNone(report['specialization_gap'])
            self.assertTrue(report['quality_checks']['missing_cohort_roles'])

    def test_exact_sign_flip_and_deterministic_prompt_bootstrap(self):
        effects, prompts, weights = {'one': 1.0, 'two': 1.0}, {'chat': ['one', 'two']}, {'chat': 1.0}
        first = rq1.paired_question_statistics(effects, prompts, weights, bootstrap_samples=50, seed=17)
        second = rq1.paired_question_statistics(effects, prompts, weights, bootstrap_samples=50, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first['estimate'], 1)
        self.assertEqual(first['ci95'], [1, 1])
        self.assertEqual(first['paired_sign_flip_p_two_sided'], .5)
        self.assertEqual(first['sign_flip_method'], 'exact_prompt_sign_flip')
        zero = rq1.paired_question_statistics({'one': 0.0, 'two': 0.0}, prompts, weights, bootstrap_samples=10, seed=1)
        self.assertEqual(zero['paired_sign_flip_p_two_sided'], 1)

    def test_cli_outputs_from_raw_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            path = root / 'manifest.json'
            path.write_bytes(rq1.canonical(manifest))
            output = root / 'report.json'
            markdown = root / 'report.md'
            argv = ['--root', str(root), '--manifest', str(path), '--output', str(output), '--markdown', str(markdown), '--bootstrap-samples', '10']
            with redirect_stdout(io.StringIO()):
                self.assertEqual(rq1.main(argv), 0)
            report = json.loads(output.read_bytes())
            self.assertFalse(report['scientific_comparison_eligible'])
            self.assertIn('Canonical raw tau', markdown.read_text())
            self.assertEqual(report['provenance']['manifest_file_sha256'], rq1.digest(path))
            original = output.read_bytes()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(rq1.main(argv), 1)
            self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
