import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile

from . import metrics


class SpecializationError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
        raise SpecializationError(name + ' must be a lowercase SHA256')
    return value


def _name(value, name):
    if not isinstance(value, str) or not value:
        raise SpecializationError(name + ' must be a nonempty string')
    return value


def _read(root, descriptor, inventory, name):
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get('path'), str):
        raise SpecializationError(name + ' requires a file path and SHA256')
    expected = _sha(descriptor.get('sha256'), name + ' SHA256')
    path = Path(descriptor['path'])
    path = (path if path.is_absolute() else root / path).resolve(strict=True)
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise SpecializationError('Input SHA256 differs: ' + name)
    inventory[name] = {'path': str(path), 'sha256': expected, 'bytes': len(contents)}
    try:
        return [json.loads(line) for line in contents.splitlines() if line.strip()] if path.suffix == '.jsonl' else json.loads(contents)
    except (ValueError, UnicodeError) as error:
        raise SpecializationError('Malformed JSON input: ' + name) from error


def _case(row):
    return (_name(row['domain'], 'domain'), metrics._identifier(row['question_id'], 'question_id'), metrics._identifier(row['turn_id'], 'turn_id'), metrics._integer(row['repeat'], 'repeat'), metrics._integer(row['seed'], 'seed'))


def _label(key):
    return {'domain': key[0], 'question_id': json.loads(key[1]), 'turn_id': json.loads(key[2]), 'repeat': key[3], 'seed': key[4]}


def _panel(rows, domains):
    if not isinstance(rows, list) or not rows:
        raise SpecializationError('Expected evaluation panel must be a nonempty JSONL file')
    panel, prompts, questions, turn_repeats = {}, {}, {}, set()
    for row in rows:
        if not isinstance(row, dict):
            raise SpecializationError('Expected panel records must be objects')
        key = _case(row)
        if key[0] not in domains or key in panel or key[:4] in turn_repeats:
            raise SpecializationError('Panel contains an unknown domain or duplicate case')
        turn_repeats.add(key[:4])
        prompt = _name(row.get('prompt_id'), 'prompt_id')
        _sha(row.get('prompt_sha256'), 'prompt_sha256')
        question = key[:2]
        if prompt in prompts and prompts[prompt] != question:
            raise SpecializationError('One prompt_id cannot identify different questions or domains')
        if question in questions and questions[question] != prompt:
            raise SpecializationError('All turns and repeats of one question must share prompt_id')
        prompts[prompt], questions[question] = question, prompt
        panel[key] = dict(row)
    if {key[0] for key in panel} != set(domains):
        raise SpecializationError('Every declared domain requires planned evaluation cases')
    turn_sets = defaultdict(lambda: defaultdict(set))
    for key in panel:
        turn_sets[key[:2]][key[3]].add(key[2])
    if any(len({tuple(sorted(turns)) for turns in repeats.values()}) != 1 for repeats in turn_sets.values()):
        raise SpecializationError('Every planned repeat of a question must contain the same complete turns')
    return panel


def validate_record(value):
    if not isinstance(value, dict):
        raise SpecializationError('Raw evaluation rows must be objects')
    required = ('status', 'warmup', 'generation_mode', 'question_id', 'turn_id', 'repeat', 'seed', 'domain', 'prompt_id', 'prompt_sha256', 'target_identity_sha256', 'drafter_checkpoint_sha256', 'evaluation_budget', 'cycle_count', 'cycles', 'new_token_count', 'output_token_ids', 'comparison_token_ids', 'wall_ms')
    missing = [name for name in required if name not in value]
    if missing:
        raise SpecializationError('Evaluation row is missing: ' + ', '.join(missing))
    if value['status'] != 'completed' or value['generation_mode'] != 'speculative' or type(value['warmup']) is not bool:
        raise SpecializationError('Only completed speculative rows with an explicit warmup boolean are accepted')
    row = dict(value)
    key = _case(row)
    row['_question'], row['_turn'] = key[1:3]
    row['_unused_final_drafting_recorded'] = 'unused_final_drafting_ms' in row
    row['unused_final_drafting_ms'] = metrics._number(row.get('unused_final_drafting_ms', 0), 'unused_final_drafting_ms')
    _name(row['prompt_id'], 'prompt_id')
    for name in ('prompt_sha256', 'target_identity_sha256', 'drafter_checkpoint_sha256'):
        _sha(row[name], name)
    metrics._integer(row['cycle_count'], 'cycle_count', 1)
    metrics._integer(row['new_token_count'], 'new_token_count', 1)
    row['wall_ms'] = metrics._number(row['wall_ms'], 'wall_ms', positive=True)
    if not isinstance(row['cycles'], list) or len(row['cycles']) != row['cycle_count']:
        raise SpecializationError('cycle_count must equal the explicit true cycle list length')
    emitted = 0
    for cycle in row['cycles']:
        if not isinstance(cycle, dict):
            raise SpecializationError('Each explicit verification cycle must be an object')
        accepted = metrics._integer(cycle.get('accepted_draft_tokens'), 'accepted_draft_tokens')
        count = metrics._integer(cycle.get('emitted_tokens'), 'emitted_tokens', 1)
        if accepted + 1 != count:
            raise SpecializationError('Speculative emitted tokens must equal accepted draft tokens plus one')
        for name, minimum in (('verifier_nodes', 1), ('proposed_nodes', 0)):
            metrics._integer(cycle.get(name), name, minimum)
        if accepted > cycle['proposed_nodes'] or count > cycle['verifier_nodes'] or cycle['proposed_nodes'] >= cycle['verifier_nodes']:
            raise SpecializationError('Cycle acceptance or proposal count exceeds the actual verifier-node tree')
        for name in ('drafting_ms', 'verification_ms'):
            metrics._number(cycle.get(name), name)
        emitted += count
    for name in ('output_token_ids', 'comparison_token_ids'):
        if not isinstance(row[name], list):
            raise SpecializationError(name + ' must be a token-ID list')
        for token in row[name]:
            metrics._integer(token, name + ' token')
        row[name] = tuple(row[name])
    if emitted != row['new_token_count'] or emitted != len(row['output_token_ids']):
        raise SpecializationError('Raw emitted sum, new_token_count and output token length must agree')
    if row['comparison_token_ids'] != row['output_token_ids'][:len(row['comparison_token_ids'])] or len(row['comparison_token_ids']) > emitted:
        raise SpecializationError('Comparison tokens must be a trimmed raw output prefix')
    if not isinstance(row['evaluation_budget'], dict):
        raise SpecializationError('Each row must record its actual evaluation_budget')
    return row


def _training_pair(generalist, specialist, *, target, seed, domains, domain, tolerance):
    reasons = []
    for label, receipt, expected_domains in (('generalist', generalist, set(domains)), ('specialist', specialist, {domain})):
        if not isinstance(receipt, dict):
            raise SpecializationError('Training evidence must be a JSON object')
        if receipt.get('status') != 'complete' or receipt.get('scope') != 'research':
            reasons.append(label + '_training_not_complete_research')
        if receipt.get('training_seed') != seed or receipt.get('target_identity_sha256') != target:
            reasons.append(label + '_training_seed_or_target_mismatch')
        if set(receipt.get('domains', [])) != expected_domains:
            reasons.append(label + '_training_domains_mismatch')
        observed, declared = receipt.get('observed', {}), receipt.get('declared', {})
        for name in ('loss_tokens', 'optimizer_steps'):
            value = observed.get(name)
            if type(value) is not int or value < 1 or declared.get(name) != value:
                reasons.append(label + '_unverified_observed_' + name)
    for name in ('loss_tokens', 'optimizer_steps'):
        if generalist.get('observed', {}).get(name) != specialist.get('observed', {}).get(name):
            reasons.append('unequal_observed_' + name)
    for name in ('architecture_identity_sha256', 'response_regime'):
        if not generalist.get(name) or generalist.get(name) != specialist.get(name):
            reasons.append('unmatched_' + name)
    for receipt in (generalist, specialist):
        try:
            _sha(receipt.get('architecture_identity_sha256'), 'architecture_identity_sha256')
        except ValueError:
            reasons.append('invalid_architecture_identity')
    if generalist.get('response_regime') not in ('target_generated', 'original'):
        reasons.append('unknown_response_regime')
    compute = [value.get('compute') for value in (generalist, specialist)]
    valid_compute = all(isinstance(value, dict) and value.get('kind') in ('measured_flops', 'measured_gpu_seconds') and isinstance(value.get('value'), (int, float)) and not isinstance(value.get('value'), bool) and math.isfinite(value['value']) and value['value'] > 0 for value in compute)
    if not valid_compute:
        reasons.append('measured_training_compute_missing')
        relative = None
    else:
        first, second = compute
        for name in ('kind', 'measurement_protocol_sha256', 'measurement_scope'):
            if not first.get(name) or first.get(name) != second.get(name):
                reasons.append('unmatched_compute_' + name)
        for value in compute:
            try:
                _sha(value.get('measurement_protocol_sha256'), 'measurement_protocol_sha256')
            except ValueError:
                reasons.append('invalid_compute_measurement_protocol_identity')
        if first['kind'] == 'measured_gpu_seconds':
            try:
                _sha(first.get('hardware_identity_sha256'), 'hardware_identity_sha256')
            except ValueError:
                reasons.append('gpu_time_hardware_identity_missing')
            if first.get('hardware_identity_sha256') != second.get('hardware_identity_sha256'):
                reasons.append('unmatched_gpu_time_hardware')
        relative = abs(first['value'] - second['value']) / first['value']
        if relative > tolerance + 1e-12:
            reasons.append('measured_training_compute_outside_tolerance')
    return {'eligible': not reasons, 'reasons': sorted(set(reasons)), 'observed_generalist': generalist.get('observed'), 'observed_specialist': specialist.get('observed'), 'generalist_compute': compute[0], 'specialist_compute': compute[1], 'relative_compute_difference': relative, 'relative_compute_tolerance': tolerance, 'definition': 'Observed loss tokens and optimizer steps must exactly equal their declared budgets and match across drafters. Measured FLOPs or measured GPU-seconds must match within the declared tolerance under identical measurement protocols; GPU-seconds additionally requires identical hardware. Token/step counts alone never establish matched compute.'}


def _question_values(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['prompt_id']][row['repeat']].append(row)
    return {prompt: statistics.fmean(sum(row['new_token_count'] for row in values) / sum(row['cycle_count'] for row in values) for values in repeats.values()) for prompt, repeats in groups.items()}


def _weighted(values, prompts, weights):
    return math.fsum(weights[domain] * statistics.fmean(values[prompt] for prompt in identifiers) for domain, identifiers in prompts.items())


def paired_question_statistics(values, prompts, weights, *, bootstrap_samples, seed, permutation_samples=10000):
    metrics._integer(bootstrap_samples, 'bootstrap_samples', 1)
    metrics._integer(permutation_samples, 'permutation_samples', 1)
    if not values or any(not identifiers for identifiers in prompts.values()):
        raise SpecializationError('Paired statistics require complete nonempty prompt clusters')
    point = _weighted(values, prompts, weights)
    generator = random.Random(seed)
    draws = []
    for _ in range(bootstrap_samples):
        resampled = {domain: [identifiers[generator.randrange(len(identifiers))] for _ in identifiers] for domain, identifiers in prompts.items()}
        draws.append(_weighted(values, resampled, weights))
    order = [prompt for domain in prompts for prompt in prompts[domain]]
    if len(order) <= 16:
        signed = [itertools.product((-1, 1), repeat=len(order)), 2 ** len(order), 'exact_prompt_sign_flip']
    else:
        signed = [([1 if generator.random() < .5 else -1 for _ in order] for _ in range(permutation_samples)), permutation_samples, 'monte_carlo_prompt_sign_flip_plus_one']
    extreme = 0
    for signs in signed[0]:
        simulated = _weighted({prompt: values[prompt] * sign for prompt, sign in zip(order, signs)}, prompts, weights)
        extreme += abs(simulated) >= abs(point) - 1e-12
    probability = extreme / signed[1] if signed[2].startswith('exact') else (extreme + 1) / (signed[1] + 1)
    return {'estimate': point, 'ci95': [metrics._quantile(draws, .025), metrics._quantile(draws, .975)], 'paired_sign_flip_p_two_sided': probability, 'sign_flip_method': signed[2], 'sign_flip_draws': signed[1], 'sign_flip_assumption': 'Paired prompt effects are exchangeable under sign reversal under the null; this is a prompt-level conditional test, not inference over unseen training seeds.', 'bootstrap_unit': 'prompt', 'bootstrap_scheme': 'Stratified by declared evaluation domain; resample whole prompts and retain every turn, evaluation repeat and observed training seed; traffic weights fixed.', 'bootstrap_samples': bootstrap_samples, 'training_seed_uncertainty_established': False}


def analyze(value, root, *, bootstrap_samples=2000, seed=20261109):
    metrics._integer(bootstrap_samples, 'bootstrap_samples', 1)
    metrics._integer(seed, 'seed')
    root, manifest = Path(root).resolve(), dict(value)
    if manifest.get('schema_version') != 1 or manifest.get('scope') not in ('engineering', 'research') or manifest.get('data_origin') not in ('observed_logs', 'synthetic_fixture'):
        raise SpecializationError('Manifest requires schema_version=1, explicit scope and data_origin')
    target = _sha(manifest.get('target_identity_sha256'), 'target_identity_sha256')
    domains = manifest.get('domains')
    if not isinstance(domains, list) or not domains or any(not isinstance(domain, str) or not domain or domain == 'generalist' for domain in domains) or len(set(domains)) != len(domains):
        raise SpecializationError('domains must contain unique explicit domain names')
    budget = manifest.get('evaluation_budget')
    if not isinstance(budget, dict):
        raise SpecializationError('An explicit common evaluation_budget is required')
    metrics._integer(budget.get('max_new_tokens'), 'max_new_tokens', 1)
    metrics._integer(budget.get('verifier_nodes'), 'verifier_nodes', 1)
    temperature = metrics._number(budget.get('temperature'), 'temperature')
    canonical(budget)
    tolerance = metrics._number(manifest.get('relative_compute_tolerance', .01), 'relative_compute_tolerance')
    if tolerance > .1:
        raise SpecializationError('Declare a compute tolerance no larger than 10 percent')
    minimum_seeds = metrics._integer(manifest.get('minimum_training_seeds', 3), 'minimum_training_seeds', 1)
    inventory = {}
    panel = _panel(_read(root, manifest['panel'], inventory, 'panel'), domains)
    prompts = {domain: sorted({row['prompt_id'] for key, row in panel.items() if key[0] == domain}) for domain in domains}
    weights = manifest.get('traffic_weights', {domain: len(prompts[domain]) / sum(map(len, prompts.values())) for domain in domains})
    if not isinstance(weights, dict) or set(weights) != set(domains) or any(isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0 for weight in weights.values()) or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-10):
        raise SpecializationError('traffic_weights must cover all domains, be positive and sum to one')
    drafters = manifest.get('drafters')
    if not isinstance(drafters, list) or not drafters:
        raise SpecializationError('Explicit matched drafter descriptors are required')
    cohorts, identifiers, matrix, quality, all_rows, evidence, training_identity_issues = defaultdict(dict), set(), [], [], {}, {}, []
    checkpoint_seeds = defaultdict(set)
    warmups = 0
    for drafter in drafters:
        identifier = _name(drafter.get('id'), 'drafter id')
        if identifier in identifiers:
            raise SpecializationError('Duplicate drafter descriptor ID')
        identifiers.add(identifier)
        training_seed = metrics._integer(drafter.get('training_seed'), 'training_seed')
        role = drafter.get('role')
        domain = drafter.get('domain') if role == 'specialist' else None
        if role not in ('generalist', 'specialist') or (role == 'specialist' and domain not in domains) or (role == 'generalist' and drafter.get('domain') is not None):
            raise SpecializationError('Drafter must be one generalist or one explicitly domain-labeled specialist')
        slot = 'generalist' if role == 'generalist' else domain
        if slot in cohorts[training_seed]:
            raise SpecializationError('Every training seed requires exactly one drafter for each role/domain')
        cohorts[training_seed][slot] = identifier
        checkpoint = _sha(drafter.get('checkpoint_sha256'), 'checkpoint_sha256')
        checkpoint_seeds[checkpoint].add(training_seed)
        if drafter.get('target_identity_sha256') != target:
            raise SpecializationError('Drafter descriptor target differs from the study target')
        evidence[identifier] = _read(root, drafter['training_evidence'], inventory, 'training:' + identifier)
        if not isinstance(evidence[identifier], dict) or evidence[identifier].get('checkpoint_sha256') != checkpoint:
            training_identity_issues.append({'drafter_id': identifier, 'reason': 'training_evidence_checkpoint_identity_mismatch'})
        descriptors = drafter.get('logs')
        if not isinstance(descriptors, list) or not descriptors:
            raise SpecializationError('Each drafter requires explicit raw log file descriptors')
        records, keys, mismatches = [], {}, []
        for index, descriptor in enumerate(descriptors):
            raw = _read(root, descriptor, inventory, 'log:' + identifier + ':' + str(index))
            if not isinstance(raw, list):
                raise SpecializationError('Raw evaluation files must contain JSONL rows or JSON row arrays')
            for item in raw:
                row = validate_record(item)
                if row['warmup']:
                    warmups += 1
                    continue
                key = _case(row)
                if key in keys:
                    raise SpecializationError('Duplicate measured raw case for drafter ' + identifier)
                keys[key] = row
                expected = panel.get(key)
                errors = []
                if expected is not None and any(row[name] != expected[name] for name in ('prompt_id', 'prompt_sha256')):
                    errors.append('prompt_identity_mismatch')
                if row['target_identity_sha256'] != target or row['drafter_checkpoint_sha256'] != checkpoint:
                    errors.append('target_or_checkpoint_identity_mismatch')
                if row['evaluation_budget'] != budget or any(cycle['verifier_nodes'] != budget['verifier_nodes'] for cycle in row['cycles']):
                    errors.append('evaluation_budget_mismatch')
                if not row['comparison_token_ids'] or len(row['comparison_token_ids']) > budget['max_new_tokens']:
                    errors.append('invalid_comparison_token_budget')
                if manifest['scope'] == 'research' and row.get('scope') not in ('research', 'reference'):
                    errors.append('raw_row_scope_not_research_or_reference')
                if errors:
                    mismatches.append({'case': _label(key), 'reasons': errors})
                records.append(row)
        missing, unexpected = sorted(set(panel) - set(keys)), sorted(set(keys) - set(panel))
        complete = not missing and not unexpected and not mismatches
        item_quality = {'drafter_id': identifier, 'complete': complete, 'expected_cases': len(panel), 'observed_cases': len(records), 'missing_cases': [_label(key) for key in missing], 'unexpected_cases': [_label(key) for key in unexpected], 'identity_or_budget_mismatches': mismatches}
        quality.append(item_quality)
        all_rows[identifier] = keys
        for evaluation_domain in domains:
            selected = [row for row in records if row['domain'] == evaluation_domain]
            expected_domain = {key for key in panel if key[0] == evaluation_domain}
            domain_complete = set(key for key in keys if key[0] == evaluation_domain) == expected_domain and not any(item['case']['domain'] == evaluation_domain for item in mismatches)
            summary = metrics._mode_summary(selected, bootstrap_samples if domain_complete else 0, metrics._seed(seed, [identifier, evaluation_domain]))
            matrix.append({'training_seed': training_seed, 'drafter_id': identifier, 'role': role, 'training_domain': domain, 'evaluation_domain': evaluation_domain, 'complete': domain_complete, 'descriptive_only_if_incomplete': not domain_complete, **summary})
    cohort_reasons = []
    for training_seed, members in sorted(cohorts.items()):
        absent = sorted({'generalist', *domains} - set(members))
        if absent:
            cohort_reasons.append({'training_seed': training_seed, 'missing_roles': absent})
    budget_checks = []
    for training_seed, members in sorted(cohorts.items()):
        if 'generalist' not in members:
            continue
        for domain in domains:
            if domain in members:
                check = _training_pair(evidence[members['generalist']], evidence[members[domain]], target=target, seed=training_seed, domains=domains, domain=domain, tolerance=tolerance)
                budget_checks.append({'training_seed': training_seed, 'domain': domain, 'generalist_id': members['generalist'], 'specialist_id': members[domain], **check})
    paired = not cohort_reasons and all(item['complete'] for item in quality)
    mismatches = []
    if paired and temperature == 0:
        for training_seed, members in sorted(cohorts.items()):
            generalist = all_rows[members['generalist']]
            for domain in domains:
                for key, row in all_rows[members[domain]].items():
                    if row['comparison_token_ids'] != generalist[key]['comparison_token_ids']:
                        mismatches.append({'training_seed': training_seed, 'specialist_id': members[domain], 'case': _label(key)})
    reasons = []
    if manifest['scope'] != 'research' or manifest['data_origin'] != 'observed_logs':
        reasons.append('engineering_or_synthetic_inputs_not_scientific_results')
    if len(cohorts) < minimum_seeds:
        reasons.append('insufficient_training_seed_cohorts')
    if len(cohorts) < 3:
        reasons.append('fewer_than_three_training_seed_cohorts')
    if any(len(values) < 2 for values in prompts.values()):
        reasons.append('insufficient_prompt_clusters_for_uncertainty')
    if training_identity_issues:
        reasons.append('training_evidence_checkpoint_identity_mismatch')
    if any(len(seeds) > 1 for seeds in checkpoint_seeds.values()):
        reasons.append('checkpoint_reused_across_training_seeds')
    if not paired:
        reasons.append('incomplete_or_unpaired_evaluation_panel')
    if not budget_checks or any(not check['eligible'] for check in budget_checks):
        reasons.append('training_tokens_steps_or_measured_compute_not_matched')
    if mismatches:
        reasons.append('t0_comparison_outputs_mismatch')
    for row in matrix:
        row['scientific_comparison_eligible'] = not reasons
    result = {'schema_version': 1, 'status': 'complete', 'scope': manifest['scope'], 'data_origin': manifest['data_origin'], 'scientific_comparison_eligible': not reasons, 'ineligibility_reasons': reasons, 'target_identity_sha256': target, 'evaluation_budget': budget, 'training_seed_count': len(cohorts), 'training_seeds': sorted(cohorts), 'minimum_training_seeds': minimum_seeds, 'traffic_weights': weights, 'expected_panel_cases': len(panel), 'expected_prompts': sum(map(len, prompts.values())), 'warmups_excluded': warmups, 'primary_tau': 'question_mean_tokens_per_cycle', 'primary_tau_definition': 'Pool raw emitted tokens and explicit verification cycles across turns within each question/repeat; average repeat ratios within question, then questions. Per-turn evaluation seed never splits one question-repeat.', 'primary_oracle_definition': 'Fixed domain oracle: route each heldout prompt to the specialist trained for its declared domain; never choose the best specialist from its measured heldout outcome.', 'quality_checks': {'all_evaluations_exactly_paired': paired, 'drafter_coverage': quality, 'missing_cohort_roles': cohort_reasons, 'training_evidence_identity_issues': training_identity_issues, 'training_budget_pairs': budget_checks, 't0_pairwise_sequence_mismatch_count': len(mismatches), 't0_pairwise_sequence_mismatches': mismatches, 'target_only_correctness_established_by_this_report': False}, 'cross_domain_matrix': matrix, 'specialization_gap': None, 'per_domain_gaps': [], 'per_training_seed_gaps': [], 'qualifications': ['No estimate uses an intersection of incomplete or unpaired panels.', 'Wall-clock throughput is computed from comparison-token counts and actual wall time; acceptance is never converted to speedup.', 'Compute eligibility relies on measured training evidence; observed token/step equality alone is insufficient.', 'Prompt bootstrap intervals condition on the observed training seeds and do not establish uncertainty across unseen training seeds.', 'T=1 acceptance or same-seed outputs do not establish distributional correctness.'], 'provenance': {'manifest_canonical_sha256': identity(manifest), 'inputs': inventory, 'specialization_source_sha256': digest(__file__), 'metrics_source_sha256': digest(metrics.__file__), 'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': seed}}
    if paired:
        effects = {}
        for training_seed, members in sorted(cohorts.items()):
            baseline = _question_values(all_rows[members['generalist']].values())
            routed = {}
            for domain in domains:
                specialist = _question_values([row for row in all_rows[members[domain]].values() if row['domain'] == domain])
                routed.update(specialist)
            delta = {prompt: routed[prompt] - baseline[prompt] for domain in domains for prompt in prompts[domain]}
            effects[training_seed] = delta
            result['per_training_seed_gaps'].append({'training_seed': training_seed, 'generalist_tau': _weighted(baseline, prompts, weights), 'domain_oracle_tau': _weighted(routed, prompts, weights), 'gap': _weighted(delta, prompts, weights)})
        mean_effect = {prompt: statistics.fmean(values[prompt] for values in effects.values()) for domain in domains for prompt in prompts[domain]}
        contrast = paired_question_statistics(mean_effect, prompts, weights, bootstrap_samples=bootstrap_samples, seed=metrics._seed(seed, 'primary_gap'))
        baseline = statistics.fmean(item['generalist_tau'] for item in result['per_training_seed_gaps'])
        contrast.update(generalist_tau=baseline, domain_oracle_tau=baseline + contrast['estimate'], relative_gap=contrast['estimate'] / baseline, eligible=not reasons, exploratory_only=bool(reasons), training_seed_gap_sample_sd=statistics.stdev(item['gap'] for item in result['per_training_seed_gaps']) if len(effects) > 1 else None)
        result['specialization_gap'] = contrast
        for domain in domains:
            statistics_result = paired_question_statistics(mean_effect, {domain: prompts[domain]}, {domain: 1.0}, bootstrap_samples=bootstrap_samples, seed=metrics._seed(seed, ['domain_gap', domain]))
            result['per_domain_gaps'].append({'domain': domain, 'eligible': not reasons, 'exploratory_only': bool(reasons), **statistics_result})
    result['qualifications'].extend(['Scientific comparison eligibility concerns the declared matched RQ1 panel; it is not a reproduction, distributional-correctness, training-authorization or publication gate.', 'The domain-oracle contrast is primary; cross-domain and per-domain intervals are unadjusted exploratory diagnostics.', 'Training-set disjointness is not inferred from prompt hashes; its separate data-preparation audit must be retained.'])
    return result


def markdown(report):
    lines = ['# RQ1 specialization analysis', '', 'Scientific comparison eligible: ' + str(report['scientific_comparison_eligible']).lower(), '', report['primary_oracle_definition'], '', '| Training seed | Drafter | Training domain | Evaluation domain | Canonical raw tau | Complete |', '|---|---|---|---|---:|---|']
    for row in report['cross_domain_matrix']:
        tau = row['question_mean_tokens_per_cycle']
        lines.append('| ' + ' | '.join([str(row['training_seed']), row['drafter_id'], row['training_domain'] or 'all domains', row['evaluation_domain'], 'unavailable' if tau is None else format(tau, '.6f'), str(row['complete']).lower()]) + ' |')
    gap = report['specialization_gap']
    if gap is None:
        lines.extend(['', 'Paired specialization gap withheld because the planned panel or training-seed cohort is incomplete or unpaired.'])
    else:
        lines.extend(['', 'Domain-oracle minus matched generalist raw tau: ' + format(gap['estimate'], '.6f') + '; prompt-cluster 95% interval [' + ', '.join(format(value, '.6f') for value in gap['ci95']) + '].', '', 'Exploratory only: ' + str(gap['exploratory_only']).lower() + '.'])
    if report['ineligibility_reasons']:
        lines.extend(['', 'Ineligibility reasons: ' + ', '.join(report['ineligibility_reasons']) + '.'])
    lines.extend(['', *report['qualifications'], ''])
    return '\n'.join(lines)


def _write(path, contents):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise SpecializationError('Refusing to overwrite an existing report')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.' + path.name + '-', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261109)
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        path = args.manifest if args.manifest.is_absolute() else root / args.manifest
        contents = path.read_bytes()
        manifest_sha256 = hashlib.sha256(contents).hexdigest()
        manifest = json.loads(contents)
        if manifest.get('scope') == 'research' and not os.environ.get('SLURM_JOB_ID'):
            raise SpecializationError('Research analysis requires Slurm; small engineering fixtures can run locally')
        report = analyze(manifest, root, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        if digest(path) != manifest_sha256:
            raise SpecializationError('Manifest changed while analysis was running')
        report['provenance']['manifest_file_sha256'] = manifest_sha256
        output = args.output if args.output.is_absolute() else root / args.output
        rendered = args.markdown if args.markdown is None or args.markdown.is_absolute() else root / args.markdown
        if output.exists() or output.is_symlink() or (rendered is not None and (rendered.exists() or rendered.is_symlink())):
            raise SpecializationError('Refusing to overwrite existing analysis artifacts')
        _write(output, json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n')
        if rendered is not None:
            _write(rendered, markdown(report).encode())
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}), flush=True)
        return 1
    print(json.dumps({'status': report['status'], 'scientific_comparison_eligible': report['scientific_comparison_eligible'], 'output': str(output)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
