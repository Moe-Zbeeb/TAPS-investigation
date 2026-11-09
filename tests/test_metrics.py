import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from taps_rebuild.metrics import main, summarize


def record(question=0, turn=0, repeat=0, mode='speculative', emitted=(3,), wall=10.0, temperature=0.0, benchmark='gsm8k', comparison=None, seed=None, scope='reference'):
    count = sum(emitted)
    tokens = list(range(1, count + 1))
    return {'method': 'eagle3', 'generation_mode': mode, 'benchmark': benchmark, 'temperature': temperature, 'question_id': question, 'turn_id': turn, 'repeat': repeat, 'seed': 20261109 + repeat if seed is None else seed, 'cycles': [{'accepted_draft_tokens': value - 1 if mode == 'speculative' else 0, 'emitted_tokens': value, 'drafting_ms': 0.25 if mode == 'speculative' else 0.0, 'verification_ms': 1.0, 'verifier_nodes': value, 'proposed_nodes': max(0, value - 1)} for value in emitted], 'new_token_count': count, 'cycle_count': len(emitted), 'wall_ms': wall, 'output_token_ids': tokens, 'comparison_token_ids': tokens if comparison is None else comparison, 'warmup': False, 'case_id': f'{mode}/{question}/{turn}/{repeat}/{seed}', 'status': 'completed', 'scope': scope}


def reference(questions=2, turns=1, tau=3.0, temperature=0.0, benchmark='gsm8k', repeats=5):
    return {'method': 'eagle3', 'benchmark': benchmark, 'temperature': temperature, 'published_tau': tau, 'source_url': 'https://example.test/synthetic-reference', 'expected_questions': questions, 'expected_turns': turns, 'timing_repeats': repeats}


def paired_records(questions=2, turns=1, repeats=5, temperature=0.0, benchmark='gsm8k'):
    rows = []
    for question in range(questions):
        for turn in range(turns):
            for repeat in range(repeats):
                rows.append(record(question, turn, repeat, temperature=temperature, benchmark=benchmark))
                rows.append(record(question, turn, repeat, mode='target_only', emitted=(1, 1, 1), wall=20, temperature=temperature, benchmark=benchmark))
    return rows


def report(rows, references=None, samples=64, seed=17):
    return summarize(rows, reference_cells=references, bootstrap_samples=samples, seed=seed)


class MetricTests(unittest.TestCase):
    def test_question_mean_differs_from_token_weighted_pool(self):
        result = report([record(0, emitted=(1,) * 100), record(1, emitted=(10,))])
        cell = result['cells'][0]['speculative']
        self.assertEqual(cell['question_mean_tokens_per_cycle'], 5.5)
        self.assertAlmostEqual(cell['pooled_tokens_per_cycle'], 110 / 101)
        self.assertEqual(cell['question_mean_tokens_per_cycle_ci95'], [1.0, 10.0])
        self.assertEqual(result['reproduction_gate']['status'], 'not_configured')

    def test_equal_turn_statistic_is_explicitly_diagnostic(self):
        rows = [record(0, 0, 0, emitted=(2,)), record(0, 0, 1, emitted=(2,)), record(0, 1, 0, emitted=(8,)), record(1, emitted=(3,))]
        cell = report(rows)['cells'][0]['speculative']
        self.assertEqual(cell['question_mean_tokens_per_cycle'], 3.25)
        self.assertEqual(cell['equal_turn_mean_tokens_per_cycle'], 4.0)
        self.assertEqual(cell['pooled_tokens_per_cycle'], 3.75)

    def test_question_repeat_pools_turn_cycles_and_ignores_per_turn_seed(self):
        rows = [record(0, 0, emitted=(10,), comparison=[1, 2], seed=11), record(0, 1, emitted=(1,) * 10, seed=99)]
        cell = report(rows)['cells'][0]['speculative']
        self.assertAlmostEqual(cell['question_mean_tokens_per_cycle'], 20 / 11)
        self.assertAlmostEqual(cell['question_mean_comparison_tokens_per_cycle'], 12 / 11)
        self.assertEqual(cell['equal_turn_mean_tokens_per_cycle'], 5.5)
        self.assertEqual(cell['equal_turn_mean_comparison_tokens_per_cycle'], 1.5)

    def test_repeat_ratios_are_equally_weighted_within_each_question(self):
        rows = [record(0, 0, 0, emitted=(10,), seed=11), record(0, 1, 0, emitted=(1,) * 10, seed=99), record(0, 0, 1, emitted=(2,), seed=12), record(0, 1, 1, emitted=(2,), seed=100)]
        cell = report(rows)['cells'][0]['speculative']
        self.assertAlmostEqual(cell['question_mean_tokens_per_cycle'], (20 / 11 + 2) / 2)
        self.assertAlmostEqual(cell['pooled_tokens_per_cycle'], 24 / 13)
        self.assertNotAlmostEqual(cell['question_mean_tokens_per_cycle'], cell['pooled_tokens_per_cycle'])

    def test_reference_gate_uses_turn_concatenation_not_equal_turn_mean(self):
        rows = []
        for repeat in range(5):
            rows.extend([record(0, 0, repeat, emitted=(10,), seed=100 + repeat, benchmark='mt_bench'), record(0, 1, repeat, emitted=(1,) * 10, seed=200 + repeat, benchmark='mt_bench'), record(0, 0, repeat, mode='target_only', emitted=(1,) * 10, seed=100 + repeat, benchmark='mt_bench'), record(0, 1, repeat, mode='target_only', emitted=(1,) * 10, seed=200 + repeat, benchmark='mt_bench')])
        result = report(rows, [reference(questions=1, turns=2, tau=20 / 11, benchmark='mt_bench')])
        self.assertTrue(result['reproduction_gate']['passed'])
        cell = result['cells'][0]['speculative']
        self.assertAlmostEqual(cell['question_mean_tokens_per_cycle'], 20 / 11)
        self.assertEqual(cell['equal_turn_mean_tokens_per_cycle'], 5.5)

    def test_unused_final_drafting_cost_is_separate_and_included(self):
        a = record(0, emitted=(3, 3))
        b = record(1, emitted=(3,))
        a['unused_final_drafting_ms'] = 2.0
        b['unused_final_drafting_ms'] = 1.5
        warmup = {'warmup': True, 'unused_final_drafting_ms': 9999}
        cell = report([a, b, warmup])['cells'][0]['speculative']
        self.assertEqual(cell['totals']['cycle_drafting_ms'], 0.75)
        self.assertEqual(cell['totals']['drafting_ms'], 0.75)
        self.assertEqual(cell['totals']['unused_final_drafting_ms'], 3.5)
        self.assertEqual(cell['totals']['drafting_ms_including_unused_final'], 4.25)
        self.assertEqual(cell['totals']['wall_ms'], 20)
        self.assertEqual(cell['timing_coverage']['unused_final_drafting_ms_recorded_cases'], 2)
        self.assertEqual(cell['timing_coverage']['unused_final_drafting_ms_missing_cases'], 0)
        for value in (-1, math.inf):
            bad = copy.deepcopy(a)
            bad['unused_final_drafting_ms'] = value
            with self.assertRaises(ValueError):
                report([bad])

    def test_raw_legacy_index_is_never_cycle_count(self):
        row = record(emitted=(3, 3))
        row['raw_reported_idx'] = 1
        result = report([row])
        self.assertEqual(result['cells'][0]['speculative']['question_mean_tokens_per_cycle'], 3)
        self.assertEqual(result['legacy_raw_reported_idx_records_ignored'], 1)
        row['cycle_count'] = 1
        with self.assertRaisesRegex(ValueError, 'explicit cycles length'):
            report([row])

    def test_malformed_records_fail_instead_of_becoming_measurements(self):
        mutations = [lambda row: row.pop('cycle_count'), lambda row: row.update(new_token_count=99), lambda row: row['cycles'][0].update(accepted_draft_tokens=0), lambda row: row['cycles'][0].update(emitted_tokens=0), lambda row: row['cycles'][0].update(verifier_nodes=0), lambda row: row['cycles'][0].update(proposed_nodes=-1), lambda row: row.update(wall_ms=0), lambda row: row.update(wall_ms=math.nan), lambda row: row.update(warmup='false'), lambda row: row.update(output_token_ids=[True, 2, 3]), lambda row: row.update(comparison_token_ids=[9]), lambda row: row.update(method=[]), lambda row: row.update(status='partial')]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                row = record()
                mutation(row)
                with self.assertRaises(ValueError):
                    report([row])

    def test_duplicate_samples_cannot_inflate_repeats(self):
        row = record()
        duplicate = copy.deepcopy(row)
        duplicate['case_id'] = 'another-label'
        with self.assertRaisesRegex(ValueError, 'Duplicate completed sample'):
            report([row, duplicate])

    def test_complete_reference_passes_qualified_gate(self):
        rows = paired_records()
        before = copy.deepcopy(rows)
        result = report(rows, [reference()])
        self.assertEqual(rows, before)
        self.assertEqual(result['reproduction_gate']['status'], 'passed')
        self.assertFalse(result['reproduction_gate']['losslessness_established'])
        cell = result['cells'][0]
        self.assertTrue(cell['reproduction_gate']['eligible'])
        self.assertEqual(cell['pairing']['matched_cases'], 10)
        self.assertEqual(cell['pairing']['timing']['tokens_per_second_speedup'], 2)
        self.assertEqual(cell['pairing']['timing']['wall_time_speedup'], 2)
        self.assertFalse(cell['pairing']['timing']['memory_comparison_established'])

    def test_missing_pair_is_explicitly_ineligible(self):
        rows = paired_records()
        rows.pop()
        result = report(rows, [reference()])
        self.assertEqual(result['reproduction_gate']['status'], 'ineligible')
        cell = result['cells'][0]
        self.assertEqual(len(cell['pairing']['missing_target_only_pairs']), 1)
        self.assertIn('missing_pairs', cell['reproduction_gate']['reasons'])
        self.assertFalse(cell['coverage']['target_only']['complete'])

    def test_missing_reference_cell_is_not_vacuous_success(self):
        result = report([], [reference()])
        self.assertEqual(result['reproduction_gate']['status'], 'ineligible')
        self.assertIsNone(result['cells'][0]['speculative']['question_mean_tokens_per_cycle'])
        self.assertFalse(result['cells'][0]['coverage']['speculative']['question_count_matches'])

    def test_mt_bench_requires_both_turns_for_every_question(self):
        rows = paired_records(turns=2, benchmark='mt_bench')
        refs = [reference(turns=2, benchmark='mt_bench')]
        self.assertTrue(report(rows, refs)['reproduction_gate']['passed'])
        rows = [row for row in rows if not (row['question_id'] == 1 and row['turn_id'] == 1)]
        result = report(rows, refs)
        self.assertEqual(result['reproduction_gate']['status'], 'ineligible')
        self.assertFalse(result['cells'][0]['coverage']['speculative']['complete'])
        self.assertTrue(result['cells'][0]['pairing']['pairing_complete'])

    def test_mt_bench_requires_both_turns_within_each_repeat(self):
        rows = paired_records(questions=1, turns=2, benchmark='mt_bench')
        for row in rows:
            if row['turn_id'] == 1:
                row['repeat'] += 5
                row['seed'] += 5
        result = report(rows, [reference(questions=1, turns=2, benchmark='mt_bench')])
        self.assertEqual(result['reproduction_gate']['status'], 'ineligible')
        cell = result['cells'][0]
        self.assertTrue(cell['pairing']['pairing_complete'])
        self.assertIn('repeat_turn_count', [item['reason'] for item in cell['coverage']['speculative']['incomplete_question_turns']])

    def test_at_least_five_repeats_required_even_if_reference_says_fewer(self):
        rows = paired_records(repeats=4)
        result = report(rows, [reference(repeats=3)])
        self.assertEqual(result['reproduction_gate']['status'], 'ineligible')
        self.assertEqual(result['cells'][0]['coverage']['speculative']['required_timing_repeats'], 5)

    def test_extra_questions_do_not_substitute_for_expected_panel(self):
        result = report(paired_records(questions=3), [reference(questions=2)])
        self.assertEqual(result['reproduction_gate']['status'], 'ineligible')
        self.assertFalse(result['cells'][0]['coverage']['speculative']['question_count_matches'])

    def test_seed_is_part_of_pair_identity(self):
        rows = paired_records()
        rows[1]['seed'] += 1000
        cell = report(rows, [reference()])['cells'][0]
        self.assertEqual(len(cell['pairing']['missing_target_only_pairs']), 1)
        self.assertEqual(len(cell['pairing']['missing_speculative_pairs']), 1)

    def test_t0_token_mismatch_fails_and_suppresses_wall_speedup(self):
        rows = paired_records()
        rows[1]['output_token_ids'][1] = 999
        rows[1]['comparison_token_ids'] = list(rows[1]['output_token_ids'])
        result = report(rows, [reference()])
        pairing = result['cells'][0]['pairing']
        self.assertEqual(result['reproduction_gate']['status'], 'failed')
        self.assertEqual(pairing['t0_token_prefix_mismatch_count'], 1)
        self.assertEqual(pairing['t0_mismatches'][0]['first_mismatch_position'], 1)
        self.assertIsNone(pairing['timing']['wall_time_speedup'])

    def test_t0_unequal_prefix_lengths_are_not_treated_as_correct(self):
        rows = paired_records()
        rows[1]['comparison_token_ids'] = rows[1]['comparison_token_ids'][:2]
        result = report(rows, [reference()])
        pairing = result['cells'][0]['pairing']
        self.assertEqual(pairing['t0_token_prefix_mismatch_count'], 0)
        self.assertEqual(pairing['t0_length_mismatch_count'], 1)
        self.assertFalse(result['reproduction_gate']['passed'])

    def test_distinct_terminal_tokens_are_not_discarded_from_t0_comparison(self):
        a = record(emitted=(2,))
        b = record(mode='target_only', emitted=(1, 1))
        a.update(output_token_ids=[41, 128001], comparison_token_ids=[41, 128001])
        b.update(output_token_ids=[41, 128009], comparison_token_ids=[41, 128009])
        pairing = report([a, b])['cells'][0]['pairing']
        self.assertEqual(pairing['t0_token_prefix_mismatch_count'], 1)
        self.assertEqual(pairing['t0_mismatches'][0]['first_mismatch_position'], 1)

    def test_t1_numeric_pass_never_establishes_distributional_correctness(self):
        result = report(paired_records(temperature=1), [reference(temperature=1)])
        self.assertTrue(result['reproduction_gate']['passed'])
        self.assertFalse(result['reproduction_gate']['losslessness_established'])
        pairing = result['cells'][0]['pairing']
        self.assertEqual(pairing['t0_checked_pairs'], 0)
        self.assertEqual(pairing['temperature_one_distributional_correctness'], 'requires_separate_statistical_check')
        self.assertIsNone(pairing['timing']['wall_time_speedup'])

    def test_discarded_overshoot_does_not_inflate_primary_throughput(self):
        a = record(emitted=(3, 3), comparison=[1, 2, 3])
        b = record(mode='target_only', emitted=(1, 1, 1), wall=20)
        cell = report([a, b])['cells'][0]
        self.assertEqual(cell['speculative']['question_mean_tokens_per_cycle'], 3)
        self.assertEqual(cell['speculative']['question_mean_comparison_tokens_per_cycle'], 1.5)
        self.assertEqual(cell['speculative']['raw_tokens_per_second'], 600)
        self.assertEqual(cell['speculative']['comparison_tokens_per_second'], 300)
        self.assertEqual(cell['pairing']['timing']['tokens_per_second_speedup'], 2)
        self.assertEqual(cell['pairing']['timing']['raw_tokens_per_second_speedup'], 4)
        self.assertEqual(cell['speculative']['raw_minus_comparison_tokens']['total'], 3)

    def test_empty_comparison_has_zero_rate_and_no_divide_by_zero(self):
        a = record(comparison=[])
        b = record(mode='target_only', emitted=(1, 1, 1), comparison=[])
        cell = report([a, b])['cells'][0]
        self.assertEqual(cell['speculative']['comparison_tokens_per_second'], 0)
        self.assertIsNone(cell['pairing']['timing']['tokens_per_second_speedup'])
        self.assertIsNone(cell['pairing']['timing']['tokens_per_second_speedup_ci95'])
        self.assertIsNone(cell['pairing']['timing']['wall_time_speedup'])
        self.assertIn('empty_comparison_tokens', cell['reproduction_gate']['reasons'])

    def test_warmups_are_excluded_even_if_not_instrumented(self):
        rows = paired_records()
        rows.append({'warmup': True})
        result = report(rows, [reference()])
        self.assertTrue(result['reproduction_gate']['passed'])
        self.assertEqual(result['warmup_records_excluded'], 1)
        self.assertEqual(result['analyzed_records'], 20)

    def test_smoke_scope_cannot_pass_reference_gate(self):
        rows = paired_records()
        for row in rows:
            row['scope'] = 'smoke'
        result = report(rows, [reference()])
        self.assertEqual(result['reproduction_gate']['status'], 'ineligible')
        self.assertIn('smoke_scope_not_reference_eligible', result['cells'][0]['reproduction_gate']['reasons'])

    def test_reference_values_affect_gate_only_not_timing_measurements(self):
        rows = paired_records()
        a = report(rows, [reference(tau=3.1)])
        b = report(rows, [reference(tau=3.2)])
        self.assertTrue(a['reproduction_gate']['passed'])
        self.assertFalse(b['reproduction_gate']['passed'])
        self.assertEqual(a['cells'][0]['pairing']['timing'], b['cells'][0]['pairing']['timing'])

    def test_cluster_bootstrap_is_deterministic_and_input_order_independent(self):
        rows = [record(0, repeat=repeat, emitted=(1,)) for repeat in range(5)] + [record(1, repeat=repeat, emitted=(10,)) for repeat in range(5)]
        a = report(rows, samples=128)
        b = report(list(reversed(rows)), samples=128)
        self.assertEqual(a, b)
        self.assertEqual(a['cells'][0]['speculative']['question_mean_tokens_per_cycle_ci95'], [1.0, 10.0])
        self.assertTrue(a['bootstrap']['retains_all_turns_repeats_and_seeds'])

    def test_invalid_reference_and_bootstrap_configuration(self):
        ref = reference()
        ref.pop('source_url')
        with self.assertRaises(ValueError):
            report([], [ref])
        with self.assertRaises(ValueError):
            report([], [reference(), reference()])
        with self.assertRaises(ValueError):
            summarize([], bootstrap_samples=0)

    def test_cli_reads_globs_writes_json_markdown_and_input_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = paired_records()
            for question in (0, 1):
                path = root / f'part-{question}.jsonl'
                path.write_text('\n'.join(json.dumps(row) for row in rows if row['question_id'] == question) + '\n')
            refs = root / 'references.json'
            refs.write_text(json.dumps({'reference_cells': [reference()]}))
            output = root / 'report.json'
            markdown = root / 'report.md'
            main(['--input', str(root / 'part-*.jsonl'), str(root / 'part-0.jsonl'), '--references', str(refs), '--output', str(output), '--markdown', str(markdown), '--bootstrap-samples', '16'])
            value = json.loads(output.read_text())
            self.assertTrue(value['reproduction_gate']['passed'])
            self.assertEqual(value['analyzed_records'], 20)
            self.assertEqual(len(value['provenance']), 3)
            self.assertEqual(value['provenance'][0]['sha256'], hashlib.sha256((root / 'part-0.jsonl').read_bytes()).hexdigest())
            self.assertIn('Canonical raw tau', markdown.read_text())
            self.assertIn('does not prove losslessness', markdown.read_text())


if __name__ == '__main__':
    unittest.main()
