import argparse
import glob
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path


METHODS = {'eagle3', 'eagle2', 'hass'}
MODES = ('speculative', 'target_only')
BENCHMARKS = {'mt_bench', 'humaneval', 'gsm8k', 'alpaca', 'cnn_dm'}


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return value


def _number(value, name, minimum=0, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    if value < minimum or (positive and value <= minimum):
        raise ValueError(name + ' is outside its permitted range')
    return float(value)


def _identifier(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == '':
        raise ValueError(name + ' must be a nonempty string or integer')
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _ratio(numerator, denominator):
    if not denominator:
        return None
    result = numerator / denominator
    if not math.isfinite(result):
        raise ValueError('Derived metric is nonfinite')
    return result


def _cell_key(row):
    return row['method'], row['benchmark'], row['temperature']


def _sample_key(row):
    return row['_question'], row['_turn'], row['repeat'], row['seed']


def _validate_record(value):
    if not isinstance(value, dict):
        raise ValueError('Each record must be an object')
    required = ('method', 'generation_mode', 'benchmark', 'temperature', 'question_id', 'turn_id', 'repeat', 'seed', 'cycles', 'new_token_count', 'cycle_count', 'wall_ms', 'output_token_ids', 'comparison_token_ids', 'warmup', 'case_id', 'status')
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError('Record is missing fields: ' + ', '.join(missing))
    if value['status'] != 'completed':
        raise ValueError('Only completed records are accepted')
    if any(not isinstance(value[name], str) for name in ('method', 'generation_mode', 'benchmark')) or value['method'] not in METHODS or value['generation_mode'] not in MODES or value['benchmark'] not in BENCHMARKS:
        raise ValueError('Unknown method, generation_mode or benchmark')
    if type(value['warmup']) is not bool:
        raise ValueError('warmup must be a boolean')
    if not isinstance(value['case_id'], str) or not value['case_id']:
        raise ValueError('case_id must be a nonempty string')
    row = dict(value)
    if value.get('scope', 'reference') not in ('smoke', 'reference'):
        raise ValueError('scope must be smoke or reference when provided')
    row['_scope'] = value.get('scope', 'reference')
    row['temperature'] = _number(value['temperature'], 'temperature')
    if row['temperature'] == 0:
        row['temperature'] = 0.0
    row['_question'] = _identifier(value['question_id'], 'question_id')
    row['_turn'] = _identifier(value['turn_id'], 'turn_id')
    _integer(value['repeat'], 'repeat')
    _integer(value['seed'], 'seed')
    _integer(value['cycle_count'], 'cycle_count', 1)
    _integer(value['new_token_count'], 'new_token_count', 1)
    row['wall_ms'] = _number(value['wall_ms'], 'wall_ms', positive=True)
    row['unused_final_drafting_ms'] = _number(value.get('unused_final_drafting_ms', 0.0), 'unused_final_drafting_ms')
    row['_unused_final_drafting_recorded'] = 'unused_final_drafting_ms' in value
    if not isinstance(value['cycles'], list) or len(value['cycles']) != value['cycle_count']:
        raise ValueError('cycle_count must equal the explicit cycles length')
    emitted = 0
    for cycle in value['cycles']:
        if not isinstance(cycle, dict):
            raise ValueError('Each cycle must be an object')
        fields = ('accepted_draft_tokens', 'emitted_tokens', 'drafting_ms', 'verification_ms', 'verifier_nodes', 'proposed_nodes')
        if any(name not in cycle for name in fields):
            raise ValueError('Cycle is missing required fields')
        accepted = _integer(cycle['accepted_draft_tokens'], 'accepted_draft_tokens')
        count = _integer(cycle['emitted_tokens'], 'emitted_tokens', 1)
        _integer(cycle['verifier_nodes'], 'verifier_nodes', 1)
        _integer(cycle['proposed_nodes'], 'proposed_nodes')
        _number(cycle['drafting_ms'], 'drafting_ms')
        _number(cycle['verification_ms'], 'verification_ms')
        if value['generation_mode'] == 'speculative' and accepted + 1 != count:
            raise ValueError('Speculative emitted_tokens must equal accepted_draft_tokens + 1')
        emitted += count
    for name in ('output_token_ids', 'comparison_token_ids'):
        if not isinstance(value[name], list):
            raise ValueError(name + ' must be a list')
        for token in value[name]:
            _integer(token, name + ' token')
        row[name] = tuple(value[name])
    if emitted != value['new_token_count'] or emitted != len(value['output_token_ids']):
        raise ValueError('Emitted sum, new_token_count and output_token_ids length must agree')
    comparison = row['comparison_token_ids']
    if len(comparison) > emitted or comparison != row['output_token_ids'][:len(comparison)]:
        raise ValueError('comparison_token_ids must be a trimmed prefix of output_token_ids')
    return row


def _validate_references(reference_cells):
    if reference_cells is None:
        return {}
    if isinstance(reference_cells, dict):
        reference_cells = reference_cells.get('reference_cells')
    if not isinstance(reference_cells, (list, tuple)):
        raise ValueError('reference_cells must be a list')
    references = {}
    for value in reference_cells:
        if not isinstance(value, dict):
            raise ValueError('Each reference cell must be an object')
        required = ('method', 'benchmark', 'temperature', 'published_tau', 'source_url', 'expected_questions', 'expected_turns', 'timing_repeats')
        if any(name not in value for name in required):
            raise ValueError('Reference cell is missing required fields')
        if any(not isinstance(value[name], str) for name in ('method', 'benchmark')) or value['method'] not in METHODS or value['benchmark'] not in BENCHMARKS:
            raise ValueError('Unknown reference method or benchmark')
        cell = dict(value)
        cell['temperature'] = _number(value['temperature'], 'reference temperature')
        if cell['temperature'] == 0:
            cell['temperature'] = 0.0
        cell['published_tau'] = _number(value['published_tau'], 'published_tau', positive=True)
        for name in ('expected_questions', 'expected_turns', 'timing_repeats'):
            _integer(value[name], name, 1)
        if not isinstance(value['source_url'], str) or not value['source_url']:
            raise ValueError('Reference source_url must be a nonempty string')
        key = _cell_key(cell)
        if key in references:
            raise ValueError('Duplicate reference cell: ' + str(key))
        references[key] = cell
    return references


def _seed(seed, key):
    text = json.dumps([seed, key], ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:16], 'big')


def _quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap(clusters, reducer, samples, seed):
    if not clusters:
        return {}
    generator = random.Random(seed)
    draws = defaultdict(list)
    for _ in range(samples):
        selected = [clusters[generator.randrange(len(clusters))] for _ in clusters]
        for name, value in reducer(selected).items():
            draws[name].append(value)
    return {name: None if any(value is None for value in values) else [_quantile(values, 0.025), _quantile(values, 0.975)] for name, values in draws.items()}


def _mode_summary(records, samples, seed):
    records = sorted(records, key=_sample_key)
    questions = defaultdict(lambda: defaultdict(list))
    totals = {'tokens': 0, 'comparison_tokens': 0, 'cycles': 0, 'wall_ms': 0.0, 'accepted_draft_tokens': 0, 'drafting_ms': 0.0, 'unused_final_drafting_ms': 0.0, 'verification_ms': 0.0, 'verifier_nodes': 0, 'proposed_nodes': 0}
    for row in records:
        questions[row['_question']][row['_turn']].append(row)
        totals['tokens'] += row['new_token_count']
        totals['comparison_tokens'] += len(row['comparison_token_ids'])
        totals['cycles'] += row['cycle_count']
        totals['wall_ms'] += row['wall_ms']
        totals['unused_final_drafting_ms'] += row['unused_final_drafting_ms']
        for cycle in row['cycles']:
            for name in ('accepted_draft_tokens', 'drafting_ms', 'verification_ms', 'verifier_nodes', 'proposed_nodes'):
                totals[name] += cycle[name]
    clusters = []
    for question in sorted(questions):
        turns = questions[question]
        rows = [row for turn in sorted(turns) for row in turns[turn]]
        repeats = defaultdict(list)
        for row in rows:
            repeats[row['repeat']].append(row)
        repeat_taus = [sum(row['new_token_count'] for row in repeats[repeat]) / sum(row['cycle_count'] for row in repeats[repeat]) for repeat in sorted(repeats)]
        repeat_comparison_taus = [sum(len(row['comparison_token_ids']) for row in repeats[repeat]) / sum(row['cycle_count'] for row in repeats[repeat]) for repeat in sorted(repeats)]
        clusters.append({'tau': _mean(repeat_taus), 'comparison_tau': _mean(repeat_comparison_taus), 'equal_turn_tau': _mean([_mean([row['new_token_count'] / row['cycle_count'] for row in turns[turn]]) for turn in sorted(turns)]), 'equal_turn_comparison_tau': _mean([_mean([len(row['comparison_token_ids']) / row['cycle_count'] for row in turns[turn]]) for turn in sorted(turns)]), 'tokens': sum(row['new_token_count'] for row in rows), 'comparison_tokens': sum(len(row['comparison_token_ids']) for row in rows), 'cycles': sum(row['cycle_count'] for row in rows)})

    def reduce_tau(selected):
        return {'question_mean_tokens_per_cycle': _mean([x['tau'] for x in selected]), 'pooled_tokens_per_cycle': sum(x['tokens'] for x in selected) / sum(x['cycles'] for x in selected), 'question_mean_comparison_tokens_per_cycle': _mean([x['comparison_tau'] for x in selected]), 'pooled_comparison_tokens_per_cycle': sum(x['comparison_tokens'] for x in selected) / sum(x['cycles'] for x in selected), 'equal_turn_mean_tokens_per_cycle': _mean([x['equal_turn_tau'] for x in selected]), 'equal_turn_mean_comparison_tokens_per_cycle': _mean([x['equal_turn_comparison_tau'] for x in selected])}

    point = reduce_tau(clusters) if clusters else dict.fromkeys(('question_mean_tokens_per_cycle', 'pooled_tokens_per_cycle', 'question_mean_comparison_tokens_per_cycle', 'pooled_comparison_tokens_per_cycle', 'equal_turn_mean_tokens_per_cycle', 'equal_turn_mean_comparison_tokens_per_cycle'))
    intervals = _bootstrap(clusters, reduce_tau, samples, seed)
    discarded = [row['new_token_count'] - len(row['comparison_token_ids']) for row in records]
    comparison_tps = _ratio(totals['comparison_tokens'] * 1000, totals['wall_ms'])
    totals['cycle_drafting_ms'] = totals['drafting_ms']
    totals['drafting_ms_including_unused_final'] = totals['drafting_ms'] + totals['unused_final_drafting_ms']
    return {'cases': len(records), 'questions': len(questions), 'question_turns': sum(len(turns) for turns in questions.values()), **point, **{name + '_ci95': intervals.get(name) for name in point}, 'tokens_per_second': comparison_tps, 'comparison_tokens_per_second': comparison_tps, 'raw_tokens_per_second': _ratio(totals['tokens'] * 1000, totals['wall_ms']), 'raw_minus_comparison_tokens': {'total': sum(discarded), 'mean_per_completion': _mean(discarded), 'maximum_per_completion': max(discarded) if discarded else None, 'completion_fraction_with_discarded_tokens': _mean([int(value > 0) for value in discarded]), 'definition': 'Raw tokens discarded after the first inclusive terminator or requested cap; includes block overshoot'}, 'totals': totals, 'timing_coverage': {'unused_final_drafting_ms_recorded_cases': sum(row['_unused_final_drafting_recorded'] for row in records), 'unused_final_drafting_ms_missing_cases': sum(not row['_unused_final_drafting_recorded'] for row in records)}, 'cycle_means': {name: _ratio(totals[name], totals['cycles']) for name in ('accepted_draft_tokens', 'drafting_ms', 'verification_ms', 'verifier_nodes', 'proposed_nodes')}}


def _case(row):
    return {name: row[name] for name in ('case_id', 'question_id', 'turn_id', 'repeat', 'seed')}


def _pairing(speculative, target_only, temperature, samples, seed):
    spec = {_sample_key(row): row for row in speculative}
    target = {_sample_key(row): row for row in target_only}
    common = sorted(spec.keys() & target.keys())
    missing_target = [_case(spec[key]) for key in sorted(spec.keys() - target.keys())]
    missing_spec = [_case(target[key]) for key in sorted(target.keys() - spec.keys())]
    mismatches = []
    clusters = defaultdict(lambda: {'spec_tokens': 0, 'target_tokens': 0, 'spec_comparison_tokens': 0, 'target_comparison_tokens': 0, 'spec_wall_ms': 0.0, 'target_wall_ms': 0.0})
    for key in common:
        a, b = spec[key], target[key]
        cluster = clusters[key[0]]
        cluster['spec_tokens'] += a['new_token_count']
        cluster['target_tokens'] += b['new_token_count']
        cluster['spec_comparison_tokens'] += len(a['comparison_token_ids'])
        cluster['target_comparison_tokens'] += len(b['comparison_token_ids'])
        cluster['spec_wall_ms'] += a['wall_ms']
        cluster['target_wall_ms'] += b['wall_ms']
        left, right = a['comparison_token_ids'], b['comparison_token_ids']
        if temperature == 0 and left != right:
            first = next((index for index, (x, y) in enumerate(zip(left, right)) if x != y), min(len(left), len(right)))
            mismatches.append({**_case(a), 'kind': 'token' if first < min(len(left), len(right)) else 'length', 'first_mismatch_position': first, 'speculative_token': left[first] if first < len(left) else None, 'target_only_token': right[first] if first < len(right) else None, 'speculative_comparison_length': len(left), 'target_only_comparison_length': len(right)})
    values = [clusters[key] for key in sorted(clusters)]
    equal_t0_outputs = temperature == 0 and not mismatches and bool(common) and all(spec[key]['comparison_token_ids'] and target[key]['comparison_token_ids'] for key in common)

    def reduce_timing(selected):
        summed = {name: math.fsum(x[name] for x in selected) for name in ('spec_tokens', 'target_tokens', 'spec_comparison_tokens', 'target_comparison_tokens', 'spec_wall_ms', 'target_wall_ms')}
        speculative_tps = _ratio(summed['spec_comparison_tokens'] * 1000, summed['spec_wall_ms'])
        target_tps = _ratio(summed['target_comparison_tokens'] * 1000, summed['target_wall_ms'])
        raw_speculative_tps = _ratio(summed['spec_tokens'] * 1000, summed['spec_wall_ms'])
        raw_target_tps = _ratio(summed['target_tokens'] * 1000, summed['target_wall_ms'])
        return {'speculative_tokens_per_second': speculative_tps, 'target_only_tokens_per_second': target_tps, 'speculative_comparison_tokens_per_second': speculative_tps, 'target_only_comparison_tokens_per_second': target_tps, 'speculative_raw_tokens_per_second': raw_speculative_tps, 'target_only_raw_tokens_per_second': raw_target_tps, 'tokens_per_second_speedup': _ratio(speculative_tps, target_tps), 'raw_tokens_per_second_speedup': _ratio(raw_speculative_tps, raw_target_tps), 'wall_time_speedup': _ratio(summed['target_wall_ms'], summed['spec_wall_ms']) if equal_t0_outputs else None}

    names = ('speculative_tokens_per_second', 'target_only_tokens_per_second', 'speculative_comparison_tokens_per_second', 'target_only_comparison_tokens_per_second', 'speculative_raw_tokens_per_second', 'target_only_raw_tokens_per_second', 'tokens_per_second_speedup', 'raw_tokens_per_second_speedup', 'wall_time_speedup')
    timing = reduce_timing(values) if values else dict.fromkeys(names)
    intervals = _bootstrap(values, reduce_timing, samples, seed)
    return {'matched_cases': len(common), 'matched_questions': len(clusters), 'missing_target_only_pairs': missing_target, 'missing_speculative_pairs': missing_spec, 'pairing_complete': not missing_target and not missing_spec and bool(common), 't0_checked_pairs': len(common) if temperature == 0 else 0, 't0_token_prefix_mismatch_count': sum(x['kind'] == 'token' for x in mismatches), 't0_length_mismatch_count': sum(x['kind'] == 'length' for x in mismatches), 't0_sequence_mismatch_count': len(mismatches), 't0_mismatches': mismatches, 't0_check_status': 'checked' if temperature == 0 and common else 'not_applicable' if temperature != 0 else 'no_pairs', 'timing': {**timing, **{name + '_ci95': intervals.get(name) for name in names}, 'basis': 'Matched completed cases; primary comparison tokens including terminal tokens and measured wall times; warm-up excluded', 'wall_time_speedup_definition': 'sum(target_only wall_ms) / sum(speculative wall_ms), only for T=0 identical comparison outputs', 'sampled_length_qualification': 'Nonzero-temperature rates are normalized comparison-token throughput; sampled lengths may differ and no equal-work wall-time speedup is asserted.', 'memory_comparison_established': False}, 'temperature_one_distributional_correctness': 'requires_separate_statistical_check' if temperature == 1 else 'not_established_by_this_report'}


def _coverage(records, reference):
    questions = defaultdict(lambda: defaultdict(set))
    repeat_turns = defaultdict(lambda: defaultdict(set))
    identifiers = {}
    for row in records:
        questions[row['_question']][row['_turn']].add(row['repeat'])
        repeat_turns[row['_question']][row['repeat']].add(row['_turn'])
        identifiers[row['_question']] = row['question_id']
    repeats = max(5, reference['timing_repeats'])
    incomplete = []
    for question, turns in sorted(questions.items()):
        if len(turns) != reference['expected_turns']:
            incomplete.append({'question_id': identifiers[question], 'reason': 'turn_count', 'observed': len(turns), 'required': reference['expected_turns']})
        for turn, values in sorted(turns.items()):
            if len(values) < repeats:
                incomplete.append({'question_id': identifiers[question], 'turn_id': json.loads(turn), 'reason': 'timing_repeats', 'observed': len(values), 'required': repeats})
        for repeat, values in sorted(repeat_turns[question].items()):
            if len(values) != reference['expected_turns']:
                incomplete.append({'question_id': identifiers[question], 'repeat': repeat, 'reason': 'repeat_turn_count', 'observed': len(values), 'required': reference['expected_turns']})
    complete = len(questions) == reference['expected_questions'] and not incomplete
    return {'complete': complete, 'observed_questions': len(questions), 'expected_questions': reference['expected_questions'], 'expected_turns_per_question': reference['expected_turns'], 'required_timing_repeats': repeats, 'minimum_expected_cases': reference['expected_questions'] * reference['expected_turns'] * repeats, 'observed_cases': len(records), 'incomplete_question_turns': incomplete, 'question_count_matches': len(questions) == reference['expected_questions']}


def summarize(records, *, reference_cells=None, relative_tolerance=0.05, bootstrap_samples=2000, seed=20261109):
    tolerance = _number(relative_tolerance, 'relative_tolerance')
    _integer(bootstrap_samples, 'bootstrap_samples', 1)
    _integer(seed, 'seed')
    references = _validate_references(reference_cells)
    groups = defaultdict(lambda: {mode: [] for mode in MODES})
    seen = set()
    warmups = 0
    legacy_ignored = 0
    input_count = 0
    for value in records:
        input_count += 1
        if isinstance(value, dict) and value.get('warmup') is True:
            warmups += 1
            continue
        row = _validate_record(value)
        key = (_cell_key(row), row['generation_mode'], _sample_key(row))
        if key in seen:
            raise ValueError('Duplicate completed sample key: ' + str(key))
        seen.add(key)
        legacy_ignored += 'raw_reported_idx' in row
        groups[_cell_key(row)][row['generation_mode']].append(row)
    cells = []
    for key in sorted(groups.keys() | references.keys()):
        modes = groups[key]
        reference = references.get(key)
        cell = {'method': key[0], 'benchmark': key[1], 'temperature': key[2]}
        for mode in MODES:
            cell[mode] = _mode_summary(modes[mode], bootstrap_samples, _seed(seed, [*key, mode, 'tau']))
        pairing = _pairing(modes['speculative'], modes['target_only'], key[2], bootstrap_samples, _seed(seed, [*key, 'paired_timing']))
        cell['pairing'] = pairing
        cell['reference'] = reference
        reasons = []
        if pairing['missing_target_only_pairs'] or pairing['missing_speculative_pairs']:
            reasons.append('missing_pairs')
        if pairing['t0_sequence_mismatch_count']:
            reasons.append('t0_comparison_mismatch')
        if any(row['_scope'] == 'smoke' for mode in MODES for row in modes[mode]):
            reasons.append('smoke_scope_not_reference_eligible')
        if any(not row['comparison_token_ids'] for mode in MODES for row in modes[mode]):
            reasons.append('empty_comparison_tokens')
        gate = {'status': 'not_referenced', 'passed': False, 'eligible': False, 'relative_tolerance': tolerance, 'tau_relative_error': None, 'reasons': reasons}
        if reference:
            coverage = {mode: _coverage(modes[mode], reference) for mode in MODES}
            cell['coverage'] = coverage
            if not all(value['complete'] for value in coverage.values()):
                reasons.append('incomplete_reference_coverage')
            tau = cell['speculative']['question_mean_tokens_per_cycle']
            error = abs(tau - reference['published_tau']) / reference['published_tau'] if tau is not None else None
            eligible = not reasons and pairing['pairing_complete']
            if not pairing['pairing_complete'] and 'missing_pairs' not in reasons:
                reasons.append('missing_pairs')
            if eligible and error > tolerance + 1e-12:
                reasons.append('tau_outside_relative_tolerance')
            status = 'passed' if eligible and not reasons else 'failed' if 't0_comparison_mismatch' in reasons or 'tau_outside_relative_tolerance' in reasons else 'ineligible'
            gate.update(status=status, passed=status == 'passed', eligible=eligible, tau_relative_error=error)
        cell['reproduction_gate'] = gate
        cells.append(cell)
    referenced = [cell for cell in cells if cell['reference'] is not None]
    missing_pairs = any(cell['pairing']['missing_target_only_pairs'] or cell['pairing']['missing_speculative_pairs'] for cell in cells)
    t0_mismatch = any(cell['pairing']['t0_sequence_mismatch_count'] for cell in cells)
    if not referenced:
        status = 'not_configured'
    elif t0_mismatch or any(cell['reproduction_gate']['status'] == 'failed' for cell in referenced):
        status = 'failed'
    elif missing_pairs or any(cell['reproduction_gate']['status'] != 'passed' for cell in referenced):
        status = 'ineligible'
    else:
        status = 'passed'
    return {'schema_version': 1, 'input_records': input_count, 'analyzed_records': input_count - warmups, 'warmup_records_excluded': warmups, 'legacy_raw_reported_idx_records_ignored': legacy_ignored, 'primary_tau': 'question_mean_tokens_per_cycle', 'primary_tau_definition': 'Within each question and repeat, sum raw emitted tokens across all turns and divide by the sum of explicit verification cycles; average repeat ratios within each question, then average questions. Per-turn seed does not split a question-repeat.', 'equal_turn_tau_definition': 'Diagnostic mean across questions of the equal-turn mean of per-completion tokens/cycle, averaged across repeats and seeds within each turn; not the canonical reproduction statistic.', 'pooled_tau_definition': 'Sum of raw generated tokens / sum of explicit verification cycles', 'drafting_cost_definition': 'drafting_ms and cycle_drafting_ms sum cycle drafting only; drafting_ms_including_unused_final adds separately recorded unused_final_drafting_ms. Measured wall time is not adjusted or reconstructed from components.', 'primary_throughput': 'comparison_tokens_per_second', 'comparison_token_definition': 'Raw output prefix through the first EOS/eot inclusive, then capped at max_new_tokens; the logger supplies this prefix and metrics validates prefix consistency.', 'bootstrap': {'unit': 'question', 'retains_all_turns_repeats_and_seeds': True, 'samples': bootstrap_samples, 'seed': seed, 'confidence': 0.95, 'interval': 'percentile'}, 'cells': cells, 'reproduction_gate': {'status': status, 'passed': status == 'passed', 'reference_cells': len(referenced), 'passed_reference_cells': sum(cell['reproduction_gate']['passed'] for cell in referenced), 'scope': 'Published tau reproduction, complete paired coverage, at least five timing repeats, and observed T=0 token comparisons', 'losslessness_established': False, 'temperature_one_distributional_correctness': 'requires_separate_statistical_check', 'qualification': 'Passing this gate reproduces configured numerical reference cells and finite T=0 checks; it does not prove losslessness or establish T=1 distributional correctness.'}, 'qualifications': ['Primary throughput uses comparison tokens including the terminating token and excludes discarded block overshoot. Raw throughput is diagnostic; no wall-clock metric is derived from tau.', 'Target-only timing may retain an unused drafter; no autoregressive memory advantage is established.', 'Matching same-seed outputs or acceptance lengths at T=1 does not establish distributional correctness.']}


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _markdown(report):
    lines = ['# Reproduction report', '', 'Gate: **' + report['reproduction_gate']['status'] + '**', '', report['reproduction_gate']['qualification'], '', 'Canonical raw tau is the question mean. Intervals resample whole questions and retain all turns and repeats. Primary throughput counts comparison tokens including the first EOS/eot, capped at the requested budget.', '', '| Method | Benchmark | T | Questions | Canonical raw tau [95% CI] | Pooled raw tau | Comparison-token throughput speedup | Gate |', '|---|---|---:|---:|---:|---:|---:|---|']
    for cell in report['cells']:
        spec = cell['speculative']
        values = [spec['question_mean_tokens_per_cycle'], spec['pooled_tokens_per_cycle'], cell['pairing']['timing']['tokens_per_second_speedup']]
        display = ['—' if value is None else format(value, '.4f') for value in values]
        interval = spec['question_mean_tokens_per_cycle_ci95']
        if interval:
            display[0] += ' [' + ', '.join(format(value, '.4f') for value in interval) + ']'
        lines.append('| ' + ' | '.join([cell['method'], cell['benchmark'], str(cell['temperature']), str(spec['questions']), *display, cell['reproduction_gate']['status']]) + ' |')
    lines.extend(['', 'Warm-up records excluded: ' + str(report['warmup_records_excluded']) + '.', ''])
    for cell in report['cells']:
        reference = cell['reference']
        if reference:
            label = cell['method'] + '/' + cell['benchmark'] + '/T=' + str(cell['temperature'])
            lines.append('- ' + label + ': [published reference](' + reference['source_url'] + '); tau=' + str(reference['published_tau']) + '; reasons=' + ', '.join(cell['reproduction_gate']['reasons']) + '.')
    lines.extend(['', *report['qualifications'], ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--references')
    parser.add_argument('--output', required=True)
    parser.add_argument('--markdown')
    parser.add_argument('--relative-tolerance', type=float, default=0.05)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261109)
    args = parser.parse_args(argv)
    try:
        paths = set()
        for pattern in args.input:
            matches = glob.glob(pattern, recursive=True)
            if not matches:
                raise ValueError('Input pattern has no matches: ' + pattern)
            for match in matches:
                path = Path(match).resolve()
                if not path.is_file():
                    raise ValueError('Input is not a file: ' + str(path))
                paths.add(path)
        records = []
        provenance = []
        for path in sorted(paths):
            content = path.read_bytes()
            provenance.append({'path': str(path), 'sha256': hashlib.sha256(content).hexdigest()})
            for number, line in enumerate(content.decode('utf-8').splitlines(), 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(str(path) + ':' + str(number) + ': invalid JSON') from error
        references = None
        if args.references:
            path = Path(args.references).resolve()
            content = path.read_bytes()
            references = json.loads(content)
            provenance.append({'reference_path': str(path), 'sha256': hashlib.sha256(content).hexdigest()})
        report = summarize(records, reference_cells=references, relative_tolerance=args.relative_tolerance, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        report['provenance'] = provenance
        _write(args.output, json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
        if args.markdown:
            _write(args.markdown, _markdown(report))
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
