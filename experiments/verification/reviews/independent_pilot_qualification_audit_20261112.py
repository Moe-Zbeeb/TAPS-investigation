import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import sys
import time


DOMAINS = ('chat', 'math', 'code')
STATE_KEYS = ('model', 'optimizer_scaler', 'scheduler', 'rng', 'global_steps', 'skipped_steps', 'micro_steps')


class AuditFailure(Exception):
    pass


def require(value, code):
    if not value:
        raise AuditFailure(code)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def identity(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def verified_identity(value):
    copied = dict(value)
    require(copied.pop('identity_sha256') == identity(copied), 'self_identity_mismatch')
    return value


def read_lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def positions(pool, quota, seed, example, occurrence):
    require(len(pool) >= quota and pool == sorted(set(pool)), 'invalid_context_pool')
    result = []
    for index in range(quota):
        random = int(identity(['pilot-position-v1', seed, example, occurrence, index])[:16], 16)
        start, end = index * len(pool) // quota, (index + 1) * len(pool) // quota
        result.append(pool[start + random * (end - start) // (2 ** 64)])
    require(len(set(result)) == quota, 'reconstructed_mask_not_distinct')
    return result


def tensor_identity(value, torch):
    result = hashlib.sha256()
    tensor = value.detach().cpu().contiguous()
    result.update(json.dumps({'dtype': str(tensor.dtype), 'shape': list(tensor.shape)}, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())
    result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


class Audit:
    def __init__(self, root):
        self.root = root
        self.inventory = {}
        self.phase = 'initialization'
        self.plans = {}
        self.gradient_contexts = {}

    def artifact(self, descriptor):
        if isinstance(descriptor, str):
            descriptor = {'path': descriptor}
        path = Path(descriptor['path'])
        path = (self.root / path if not path.is_absolute() else path).resolve()
        storage = (self.root / 'models').resolve()
        require(path.is_relative_to(self.root) or path.is_relative_to(storage), 'artifact_outside_project_or_model_storage')
        actual = {'sha256': digest(path), 'size_bytes': path.stat().st_size}
        require('sha256' not in descriptor or descriptor['sha256'] == actual['sha256'], 'artifact_digest_mismatch')
        require('size_bytes' not in descriptor or descriptor['size_bytes'] == actual['size_bytes'], 'artifact_size_mismatch')
        require(str(path) not in self.inventory or self.inventory[str(path)] == actual, 'artifact_changed_during_audit')
        self.inventory[str(path)] = actual
        return path

    def json(self, descriptor):
        return json.loads(self.artifact(descriptor).read_bytes())

    def automatic_request(self, study, qualification):
        study_path = study / 'manifest.json' if study.is_dir() else study
        manifest = self.json(str(study_path))
        common = manifest['common']
        for name, expected in common['source_sha256'].items():
            self.artifact({'path': str(Path(common['source_directory']) / name), 'sha256': expected})
        for path in qualification.rglob('*'):
            if path.is_file() and path.name not in ('controller.lock', 'run.lock'):
                self.artifact(str(path))
        launch_path = qualification / 'launch.json'
        if launch_path.exists():
            launch = self.json(str(launch_path))
            if 'bundle_directory' in launch:
                for name, expected in launch['bundle']['files'].items():
                    self.artifact({'path': str(Path(launch['bundle_directory']) / name), 'sha256': expected})
        execution_path = qualification / 'execution.json'
        if execution_path.exists():
            execution = self.json(str(execution_path))
            require(execution['study_manifest_sha256'] == digest(study_path) and execution['source_sha256'] == common['source_sha256'] and execution['scientific_training_admitted'] is False, 'qualification_execution_binding_mismatch')
            for phase in execution['phases']:
                require(phase['returncode'] == 0, 'qualification_phase_failed')
                self.artifact({'path': phase['log'], 'sha256': phase['log_sha256']})
        cases, comparisons = [], []
        for policy in ('original', 'target'):
            for variant, attempt in (('uninterrupted', 'full'), ('alternate-pad', 'alternate')):
                directory = qualification / policy / variant / 'attempts' / attempt
                cases.extend({'path': str(path)} for path in sorted(directory.glob('rank-*-step-*-gradients.pt')))
            for rank in range(8):
                def segment(variant, attempt):
                    directory = qualification / policy / variant / 'attempts' / attempt
                    return {key: {'path': str(directory / ('rank-' + str(rank) + suffix))} for key, suffix in (('initial', '.initial.json'), ('complete', '.complete.json'), ('journal', '.jsonl'))}
                control = segment('uninterrupted', 'full')
                reference = segment('native-reference', 'reference')
                prefix, restored = segment('split', 'prefix'), segment('split', 'restore')
                for kind, segments in (('native_step_parity', [reference]), ('restart_parity', [prefix, restored])):
                    if all(Path(item['path']).exists() for part in [control, *segments] for item in part.values()):
                        comparisons.append({'name': policy + '-' + kind + '-rank' + str(rank), 'kind': kind, 'group': policy, 'left': {'rank': rank, 'segments': [control]}, 'right': {'rank': rank, 'segments': segments}})
        return {'schema_version': 1, 'study_manifest': {'path': str(study_path)}, 'gradient_cases': cases, 'state_comparisons': comparisons}

    def schedules(self, descriptor):
        self.phase = 'independent_schedule_reconstruction'
        manifest_path = self.artifact(descriptor)
        study = verified_identity(json.loads(manifest_path.read_bytes()))
        require(study['status'] == 'prepared_24_waiting_qualification' and study['training_admitted'] is False, 'study_scope_mismatch')
        common = study['common']
        if 'source_directory' in common:
            for name, expected in common['source_sha256'].items():
                self.artifact({'path': str(Path(common['source_directory']) / name), 'sha256': expected})
        recipe = common['recipe']
        require(recipe['cohorts'] == [*DOMAINS, 'generalist'] and recipe['response_context_policies'] == ['original', 'target'] and recipe['seeds'] == [0, 1, 2], 'study_grid_mismatch')
        require(recipe['world_size'] == 8 and recipe['length'] == 2048 and recipe['examples_per_rank'] == 1536, 'study_budget_mismatch')
        quota = common['quota']
        require(quota in (32, 64), 'study_quota_mismatch')
        for name, item in study['files'].items():
            self.artifact({'path': str(manifest_path.parent / name), **item})
        cache = Path(common['cache'])
        gate_directory = Path(common['eligibility'])
        cache_receipt = verified_identity(self.json(str(cache / 'completion.json')))
        gate = verified_identity(self.json(str(gate_directory / 'completion.json')))
        require(cache_receipt['identity_sha256'] == common['cache_identity_sha256'] and gate['identity_sha256'] == common['eligibility_identity_sha256'] and gate['quota'] == quota, 'study_data_identity_mismatch')
        for directory, receipt in ((cache, cache_receipt), (gate_directory, gate)):
            for name, item in receipt['files'].items():
                self.artifact({'path': str(directory / name), **item})
        require(gate['status'] == 'eligible_for_schedule_preparation' and gate['provenance_audit']['status'] == 'passed', 'study_data_not_eligible')
        for item in gate['provenance_audit']['proofs']:
            self.artifact(item)
        self.artifact({'path': gate['provenance_audit']['config_path'], 'sha256': gate['provenance_audit']['config_sha256']})
        require(common['hardware'] == gate['shared_binding']['hardware'] and common['scan_runtime'] == gate['shared_binding']['runtime'], 'study_scan_runtime_hardware_mismatch')
        for role in ('target', 'ancestor'):
            require(common[role] == cache_receipt[role], 'study_model_inventory_mismatch')
            for name, item in common[role]['files'].items():
                self.artifact({'path': str(Path(common[role]['path']) / name), **item})
        for name, expected in gate['shared_binding']['upstream'].items():
            self.artifact({'path': str(self.root / 'repos/EAGLE/eagle/traineagle3' / name), 'sha256': expected})
        all_ids, families = set(), {}
        populations = {q: {domain: [0] * 8 for domain in DOMAINS} for q in (64, 32)}
        schedule_count = 0
        schedule_examples = 0
        for rank in range(8):
            count = cache_receipt['rank_examples'][rank]
            metadata_path = self.artifact(str(cache / ('rank-' + str(rank) + '.jsonl')))
            metadata = list(read_lines(metadata_path))
            require(len(metadata) == count, 'cache_metadata_coverage_mismatch')
            masks_path = self.artifact(str(cache / ('rank-' + str(rank) + '.masks')))
            gates_path = self.artifact(str(gate_directory / ('rank-' + str(rank) + '.gates')))
            ids_path = self.artifact(str(cache / ('rank-' + str(rank) + '.ids')))
            require(masks_path.stat().st_size == gates_path.stat().st_size == count * 2 * 2048 and ids_path.stat().st_size == count * 2 * 2048 * 4, 'cache_binary_geometry_mismatch')
            eligible_path = self.artifact(str(gate_directory / ('rank-' + str(rank) + '.eligible.jsonl')))
            eligible_rows = list(read_lines(eligible_path))
            require(len(eligible_rows) == count, 'eligibility_metadata_coverage_mismatch')
            pools = {domain: [] for domain in DOMAINS}
            by_index = {}
            with masks_path.open('rb') as mask_stream, gates_path.open('rb') as gate_stream:
                with mmap.mmap(mask_stream.fileno(), 0, access=mmap.ACCESS_READ) as masks, mmap.mmap(gate_stream.fileno(), 0, access=mmap.ACCESS_READ) as gates:
                    require(set(masks[:]) <= {0, 1} and set(gates[:]) <= {0, 1}, 'nonbinary_cache_mask_or_gate')
                    for index, (base, row) in enumerate(zip(metadata, eligible_rows)):
                        require(all(row[key] == value for key, value in base.items()), 'eligibility_changed_base_metadata')
                        require(row['index'] == index and row['rank'] == rank and int(identity(['pilot-rank-v1', row['example_id']])[:16], 16) % 8 == rank, 'cache_rank_index_mismatch')
                        require(row['domain'] in DOMAINS and row['split'] in ('train', 'dev', 'test') and len(row['lengths']) == 2, 'cache_domain_split_mismatch')
                        require(row['example_id'] not in all_ids, 'cache_example_duplicate')
                        all_ids.add(row['example_id'])
                        family = row['group_id']
                        require(family not in families or families[family] == (row['domain'], row['split']), 'conversation_family_crosses_split_or_domain')
                        families[family] = row['domain'], row['split']
                        expected = []
                        for policy, length in enumerate(row['lengths']):
                            require(type(length) is int and 0 < length <= 2048, 'invalid_real_length')
                            offset = (index * 2 + policy) * 2048
                            expected.append([position for position in range(6, length - 1) if masks[offset + position] == 1 and gates[offset + position] == 1])
                        require(row['eligible_positions'] == expected, 'eligibility_mask_reconstruction_mismatch')
                        for q in (64, 32):
                            if row['split'] == 'train' and min(map(len, expected)) >= q:
                                populations[q][row['domain']][rank] += 1
                        if row['split'] == 'train' and min(map(len, expected)) >= quota:
                            pools[row['domain']].append(row)
                            by_index[index] = row
            require(all(len(values) >= 128 for values in pools.values()), 'rank_domain_pool_shortfall')
            for seed in (0, 1, 2):
                for cohort in (*DOMAINS, 'generalist'):
                    path = manifest_path.parent / 'schedules' / (cohort + '-seed' + str(seed) + '-rank' + str(rank) + '.jsonl')
                    observed = list(read_lines(self.artifact(str(path))))
                    require(len(observed) == 1536, 'schedule_step_coverage_mismatch')
                    cursors, occurrence, domain_counts = Counter(), Counter(), Counter()
                    context_coverage = [set(), set()]
                    orders = {}
                    for step, actual in enumerate(observed):
                        domain = DOMAINS[(step + rank) % 3] if cohort == 'generalist' else cohort
                        cycle, offset = divmod(cursors[domain], len(pools[domain]))
                        key = domain, cycle
                        if key not in orders:
                            orders[key] = sorted(pools[domain], key=lambda row: (identity(['pilot-order-v1', seed, rank, domain, cycle, row['example_id']]), row['example_id']))
                        row = orders[key][offset]
                        sample = [positions(row['eligible_positions'][policy], quota, seed, row['example_id'], occurrence[row['example_id']]) for policy in (0, 1)]
                        expected = {'step': step, 'domain': domain, 'example_id': row['example_id'], 'index': row['index'], 'rank': rank, 'cycle': cycle, 'occurrence': occurrence[row['example_id']], 'mask_sha256': [identity(value) for value in sample], 'effective_contexts': quota, 'real_lengths': row['lengths']}
                        require(all(actual.get(key) == value for key, value in expected.items()), 'schedule_or_mask_reconstruction_mismatch')
                        self.gradient_contexts[(seed, cohort, rank, step)] = {'row': row, 'positions': sample}
                        for policy in (0, 1):
                            context_coverage[policy].update((row['example_id'], position) for position in sample[policy])
                        cursors[domain] += 1
                        occurrence[row['example_id']] += 1
                        domain_counts[domain] += 1
                    require(dict(domain_counts) == ({domain: 512 for domain in DOMAINS} if cohort == 'generalist' else {cohort: 1536}), 'schedule_domain_exposure_mismatch')
                    statistics = {'domain_examples': dict(domain_counts), 'pool_examples': {domain: len(rows) for domain, rows in pools.items()}, 'unique_examples': len(occurrence), 'repeated_exposures': 1536 - len(occurrence), 'repeat_multiplicity_histogram': {str(key): value for key, value in Counter(occurrence.values()).items()}, 'unique_base_contexts_original_target': list(map(len, context_coverage)), 'effective_contexts': 1536 * quota, 'per_head_effective_contexts': [1536 * quota] * 7, 'dense_slots': 1536 * 2048}
                    require(study['schedule_statistics'][path.name] == statistics, 'reported_schedule_statistics_mismatch')
                    schedule_count += 1
                    schedule_examples += len(observed)
        chosen = next((q for q in (64, 32) if all(value >= 128 for values in populations[q].values() for value in values)), None)
        require(chosen == quota, 'quota_priority_rule_mismatch')
        require({str(key): value for key, value in populations.items()} == gate['train_populations'], 'reported_gate_training_population_mismatch')
        condition_keys = set()
        order = []
        for seed in (0, 1, 2):
            conditions = [(cohort, policy) for cohort in (*DOMAINS, 'generalist') for policy in ('original', 'target')]
            order.extend((seed, *item) for item in sorted(conditions, key=lambda item: identity(['pilot-execution-order-v1', seed, *item])))
        output_paths, model_paths = set(), set()
        for ordinal, descriptor in enumerate(study['plans']):
            plan_path = self.artifact(descriptor)
            plan = verified_identity(json.loads(plan_path.read_bytes()))
            require(all(plan[key] == value for key, value in common.items()), 'plan_common_contract_mismatch')
            key = plan['seed'], plan['cohort'], plan['policy']
            require(key not in condition_keys and plan['policy_index'] == ['original', 'target'].index(plan['policy']) and plan['steps'] == 1536, 'plan_condition_duplicate_or_invalid')
            require(key == order[ordinal] and plan['ordinal'] == ordinal, 'randomized_seed_block_execution_order_mismatch')
            require(plan['output'] not in output_paths and plan['trained_models'] not in model_paths, 'plans_share_output_or_model_namespace')
            output_paths.add(plan['output'])
            model_paths.add(plan['trained_models'])
            condition_keys.add(key)
            for rank, item in enumerate(plan['schedules']):
                expected_path = manifest_path.parent / 'schedules' / (plan['cohort'] + '-seed' + str(plan['seed']) + '-rank' + str(rank) + '.jsonl')
                require(Path(item['path']) == expected_path and digest(expected_path) == item['sha256'], 'policy_schedule_binding_mismatch')
            self.plans[plan['identity_sha256']] = plan
        require(condition_keys == {(seed, cohort, policy) for seed in (0, 1, 2) for cohort in (*DOMAINS, 'generalist') for policy in ('original', 'target')}, 'plan_grid_coverage_mismatch')
        return {'status': 'passed', 'plan_count': 24, 'unique_rank_schedules': schedule_count, 'scheduled_rank_examples_single_policy': schedule_examples, 'train_populations': populations, 'quota': quota, 'raw_cached_examples': len(all_ids), 'no_project_helpers_imported': True, 'scope': 'independent masks, eligibility Boolean intersections, paired schedules and committed exposure only; not independent GPU argmax recomputation or tokenizer replay'}

    def gradients(self, descriptors, torch):
        self.phase = 'fp64_ce_and_gradient_reconstruction'
        cases, signal = [], set()
        outside_available = bool(descriptors)
        for descriptor in descriptors:
            path = self.artifact(descriptor)
            value = torch.load(path, map_location='cpu', weights_only=True)
            for key, expected in descriptor.get('expected', {}).items():
                require(value.get(key) == expected, 'gradient_case_metadata_mismatch')
            quota = value['quota']
            require(quota in (32, 64) and value['loss_scale'] == 128 and value['normalization'] == 2048 / quota, 'gradient_units_mismatch')
            require(value['head_weights'] == [0.8 ** head for head in range(7)], 'gradient_head_weights_mismatch')
            base = value['base_positions']
            require(len(base) == quota and base == sorted(set(base)) and min(base) >= 6 and max(base) < 2047, 'gradient_base_positions_invalid')
            context_bound = False
            if value.get('plan_identity_sha256') in self.plans:
                plan = self.plans[value['plan_identity_sha256']]
                context = self.gradient_contexts[(plan['seed'], plan['cohort'], value['rank'], value['step'])]
                require(base == context['positions'][plan['policy_index']] and value['example_id'] == context['row']['example_id'] and value['domain'] == context['row']['domain'] and value['policy'] == plan['policy'], 'gradient_case_schedule_binding_mismatch')
                require(max(base) + 1 < context['row']['lengths'][plan['policy_index']], 'gradient_case_reaches_suffix_padding')
                if 'input_ids' in value:
                    inputs = value['input_ids']
                    require(inputs.dtype == torch.long and inputs.shape == (1, 2048), 'captured_input_geometry_invalid')
                    require(tensor_identity(inputs, torch) == value['input_token_sha256'], 'captured_input_hash_mismatch')
                    cache_path = self.artifact(str(Path(plan['cache']) / ('rank-' + str(value['rank']) + '.ids')))
                    offset = (context['row']['index'] * 2 + plan['policy_index']) * 2048 * 4
                    with cache_path.open('rb') as stream:
                        stream.seek(offset)
                        original_ids = list(struct.unpack('<2048i', stream.read(2048 * 4)))
                    length = context['row']['lengths'][plan['policy_index']]
                    if value.get('alternate_pad'):
                        original_ids[length:] = [0] * (2048 - length)
                    require(inputs[0].tolist() == original_ids and value['real_length'] == length and value['cache_index'] == context['row']['index'], 'captured_inputs_do_not_match_frozen_cache_policy')
                context_bound = True
            mapping = value['vocabulary_t2d']
            teacher = value['teacher_full_vocab_logits_original_s_plus_one']
            require(mapping.dtype == torch.bool and mapping.shape == (128256,) and int(mapping.sum()) == 32000, 'gradient_vocabulary_mapping_invalid')
            require(teacher.dtype == torch.float32 and teacher.shape == (quota, 128256) and bool(torch.isfinite(teacher).all()), 'gradient_teacher_capture_invalid')
            require(bool(mapping[teacher.argmax(-1)].all()), 'gradient_full_vocab_gate_failed')
            target_p = torch.softmax(teacher[:, mapping].double(), dim=-1)
            logits = value['head_logits_fp16_before_float_cast']
            gradients = value['head_logit_gradients_fp16_scaled128_before_unscale_and_dp_reduce']
            losses = value['native_head_mean_losses']
            require(len(logits) == len(gradients) == len(losses) == 7, 'gradient_head_coverage_mismatch')
            sparse = value.get('full_row_sparse_head_logit_gradients_fp16', value.get('head_logit_gradients_full_sparse_rows_fp16_scaled128'))
            outside_available = outside_available and sparse is not None
            if sparse is not None:
                require(len(sparse) == 7, 'sparse_gradient_head_coverage_mismatch')
            heads = []
            reference_total = observed_total = 0.0
            for head in range(7):
                z, actual = logits[head], gradients[head]
                require(z.dtype == actual.dtype == torch.float16 and z.shape == actual.shape == (quota, 32000) and bool(torch.isfinite(z).all()) and bool(torch.isfinite(actual).all()), 'gradient_tensor_geometry_or_finiteness_failed')
                logq = torch.log_softmax(z.double(), dim=-1)
                q = logq.exp()
                expected_ce = float(-(target_p * logq).sum(-1).mean())
                observed_ce = float(losses[head]) * 2048 / quota
                ce_error = abs(expected_ce - observed_ce)
                require(math.isfinite(observed_ce) and ce_error <= 5e-5, 'fp64_head_ce_bound_failed')
                weight = 0.8 ** head
                reference_total += weight * expected_ce
                observed_total += weight * observed_ce
                expected = ((q - target_p) * (128 * weight / quota)).half()
                require(bool(torch.isfinite(expected).all()), 'fp64_reference_gradient_not_representable')
                expected64, actual64 = expected.double(), actual.double()
                up = torch.nextafter(expected, torch.full_like(expected, math.inf)).double()
                down = torch.nextafter(expected, torch.full_like(expected, -math.inf)).double()
                ulp = torch.maximum((up - expected64).abs(), (down - expected64).abs())
                error = (actual64 - expected64).abs()
                tolerance = torch.maximum(4 * ulp, torch.full_like(ulp, 2e-6))
                violations = int((error > tolerance).sum())
                require(violations == 0, 'fp64_logit_gradient_element_bound_failed')
                norm = float(torch.linalg.vector_norm(expected64))
                relative = float(torch.linalg.vector_norm(actual64 - expected64)) / norm if norm else None
                signal_bearing = norm > 100 * 2e-6 * math.sqrt(quota * 32000)
                if signal_bearing:
                    require(relative <= 0.005, 'fp64_logit_gradient_relative_norm_bound_failed')
                    signal.add((value['domain'], value['policy'], head))
                if sparse is not None:
                    full = sparse[head]
                    require(full.layout == torch.sparse_coo and full.dtype == torch.float16 and full.shape == (2048, 32000) and full.sparse_dim() == 1 and full.dense_dim() == 1, 'full_sparse_gradient_geometry_mismatch')
                    full = full.coalesce()
                    indices, entries = full.indices()[0].tolist(), full.values()
                    selected = [position - head for position in base]
                    require(set(indices) <= set(selected) and bool(torch.isfinite(entries).all()), 'outside_mask_gradient_nonzero')
                    reconstructed = torch.zeros_like(actual)
                    offsets = {position: index for index, position in enumerate(selected)}
                    for index, position in enumerate(indices):
                        reconstructed[offsets[position]] = entries[index]
                    require(torch.equal(reconstructed, actual), 'full_sparse_and_selected_gradient_mismatch')
                heads.append({'head': head, 'ce_absolute_error': ce_error, 'gradient_max_absolute_error': float(error.max()), 'element_bound_violations': violations, 'reference_gradient_l2': norm, 'relative_l2_error': relative, 'signal_bearing': signal_bearing})
            require(abs(reference_total - observed_total) <= 5e-5, 'fp64_total_ce_bound_failed')
            total = value.get('actual_normalized_total_loss', value.get('normalized_total_loss'))
            if total is not None:
                require(abs(reference_total - float(total)) <= 5e-5, 'captured_total_loss_bound_failed')
            if 'real_length' in value:
                require(max(base) + 1 < value['real_length'] <= 2048, 'captured_real_length_bound_failed')
            if 'head_grid_indices' in value:
                require(value['head_grid_indices'] == [[position - head for position in base] for head in range(7)], 'captured_head_grid_alignment_failed')
            cases.append({'path': str(path), 'rank': value.get('rank'), 'step': value.get('step'), 'domain': value.get('domain'), 'policy': value.get('policy'), 'schedule_and_real_length_bound': context_bound, 'heads': heads, 'weighted_total_ce_absolute_error': abs(reference_total - observed_total), 'outside_mask_gradient_independently_checked': sparse is not None})
            del value, target_p
        wanted = {(domain, policy, head) for domain in DOMAINS for policy in ('original', 'target') for head in range(7)}
        missing = sorted(wanted - signal)
        return {'status': 'passed' if cases and not missing else 'partial_evidence' if cases else 'unavailable', 'cases': cases, 'missing_signal_bearing_domain_policy_heads': missing, 'outside_mask_status': 'passed' if cases and outside_available else 'unavailable', 'all_cases_bound_to_independent_schedules': bool(cases) and all(item['schedule_and_real_length_bound'] for item in cases)}

    def stream(self, descriptor):
        rank = descriptor['rank']
        initial, updates, final = None, {}, None
        segment_ranges = []
        for segment in descriptor['segments']:
            first = self.json(segment['initial'])
            complete = self.json(segment['complete'])
            require(complete['rank'] == rank and complete['qualification'] is True and complete['stop_step'] <= 4, 'trace_not_bounded_qualification')
            require(first['plan_identity_sha256'] == complete['plan_identity_sha256'], 'trace_plan_identity_mismatch')
            if initial is None:
                initial = first['state']
            events = list(read_lines(self.artifact(segment['journal'])))
            begins = {}
            observed_steps = []
            for event in events:
                step = event['step']
                if event['event'] == 'begin':
                    require(step not in begins, 'duplicate_physical_begin_in_trace_segment')
                    begins[step] = event
                elif event['event'] in ('committed_update', 'successful_update'):
                    require(step in begins and step not in updates, 'trace_update_without_begin_or_duplicate')
                    require(event['global_step'] == step + 1 and event['skipped_steps'] == 0 and event['effective_head_contexts'] == [event['effective_base_contexts']] * 7 and event['dense_slots'] == 2048, 'trace_budget_or_skip_mismatch')
                    state = event['state_identity']
                    require(set(STATE_KEYS) <= set(state) and state['global_steps'] == step + 1 and state['skipped_steps'] == 0, 'trace_state_schema_mismatch')
                    updates[step] = event
                    observed_steps.append(step)
            expected = list(range(complete['start_step'], complete['stop_step']))
            require(observed_steps == expected and sorted(begins) == expected, 'trace_segment_step_coverage_mismatch')
            require(complete['successful_updates_this_attempt'] == len(expected), 'trace_completion_update_count_mismatch')
            if 'journal_sha256' in complete:
                require(digest(Path(self.artifact(segment['journal']))) == complete['journal_sha256'], 'trace_journal_completion_digest_mismatch')
            final = complete
            segment_ranges.append([complete['start_step'], complete['stop_step']])
        return {'rank': rank, 'initial': initial, 'updates': updates, 'final': final, 'segments': segment_ranges}

    def traces(self, comparisons):
        self.phase = 'native_and_recovery_trace_comparison'
        results, coverage, bindings = [], defaultdict(set), {}
        for item in comparisons:
            require(item['kind'] in ('native_step_parity', 'restart_parity'), 'unknown_trace_comparison_kind')
            left, right = self.stream(item['left']), self.stream(item['right'])
            require(left['rank'] == right['rank'] and set(left['updates']) == set(right['updates']) == set(range(4)), 'four_step_trace_coverage_mismatch')
            require(left['initial'] == right['initial'], 'qualification_initial_state_mismatch')
            require(left['final']['plan_identity_sha256'] == right['final']['plan_identity_sha256'] and left['final']['source_sha256'] == right['final']['source_sha256'], 'trace_binding_mismatch')
            group = item['kind'], item.get('group', item['kind'])
            binding = left['final']['plan_identity_sha256'], left['final']['source_sha256']
            require(group not in bindings or bindings[group] == binding, 'eight_rank_comparison_group_mixes_plans_or_sources')
            require(left['rank'] not in coverage[group], 'duplicate_rank_in_comparison_group')
            bindings[group] = binding
            if item['kind'] == 'native_step_parity':
                require(left['final']['reference'] is not right['final']['reference'], 'native_comparison_has_no_reference_path')
            else:
                require(left['segments'] == [[0, 4]] and right['segments'] == [[0, 2], [2, 4]], 'restart_comparison_is_not_four_vs_two_plus_two')
            for step in range(4):
                a, b = left['updates'][step], right['updates'][step]
                require(all(a['state_identity'][key] == b['state_identity'][key] for key in STATE_KEYS), 'qualification_update_state_mismatch')
                require(a['loss'] == b['loss'] and a['native_head_losses'] == b['native_head_losses'], 'qualification_update_loss_mismatch')
                require(all(a[key] == b[key] for key in ('effective_base_contexts', 'effective_head_contexts', 'dense_slots', 'real_tokens')), 'qualification_update_exposure_mismatch')
                for key in ('pre_step_scaled_gradient_partition_sha256', 'learning_rates_before_step', 'local_fp32_master_update_l2'):
                    require(key in a and key in b and a[key] == b[key], 'qualification_gradient_lr_or_update_norm_mismatch')
            require(any(any(rate > 0 for rate in value['learning_rates_before_step']) and value['local_fp32_master_update_l2'] > 0 for value in left['updates'].values()), 'qualification_has_no_observed_positive_lr_parameter_update')
            require(left['final']['state_identity'] == right['final']['state_identity'], 'qualification_final_state_mismatch')
            coverage[group].add(left['rank'])
            results.append({'name': item['name'], 'kind': item['kind'], 'rank': left['rank'], 'status': 'passed', 'compared_steps': [0, 1, 2, 3], 'state_fields': list(STATE_KEYS)})
        statuses = {}
        for kind in ('native_step_parity', 'restart_parity'):
            groups = [ranks for (group_kind, _), ranks in coverage.items() if group_kind == kind]
            statuses[kind] = 'passed' if any(ranks == set(range(8)) for ranks in groups) else 'partial_evidence' if groups else 'unavailable'
        return {'statuses': statuses, 'comparisons': results, 'limitation': 'Compares actual retained source-bound state identities and scalar traces. Does not independently unpickle/recompute full raw ZeRO model/Adam/scaler states; checkpoint inventory and loader assertions remain separate evidence.'}

    def finish(self):
        self.phase = 'final_consumed_artifact_rehash'
        for name, item in self.inventory.items():
            path = Path(name)
            require(path.stat().st_size == item['size_bytes'] and digest(path) == item['sha256'], 'consumed_artifact_changed_after_analysis')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'qualification_analysis_requires_cpu_slurm')
    require(bool(args.manifest) != bool(args.study) and (args.study is None or args.qualification is not None), 'choose_manifest_or_study_and_qualification')
    if args.root is None:
        require(args.study is not None, 'explicit_root_required_for_manifest_mode')
        study_path = args.study / 'manifest.json' if args.study.is_dir() else args.study
        args.root = Path(json.loads(study_path.read_bytes())['common']['project_root'])
    root = args.root.resolve()
    output = args.output.resolve() if args.output.suffix == '.json' else args.output.resolve() / 'report.json'
    require(output.is_relative_to(root / 'outputs') and not output.exists(), 'choose_fresh_report_under_project_outputs')
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    audit = Audit(root)
    try:
        request = audit.automatic_request(args.study.resolve(), args.qualification.resolve()) if args.study else audit.json(str(args.manifest))
        require(request['schema_version'] == 1, 'request_schema_mismatch')
        schedules = audit.schedules(request['study_manifest']) if request.get('study_manifest') else {'status': 'unavailable'}
        gradient_descriptors = request.get('gradient_cases', [])
        if gradient_descriptors:
            import torch
            torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
            gradients = audit.gradients(gradient_descriptors, torch)
        else:
            gradients = {'status': 'unavailable', 'outside_mask_status': 'unavailable'}
        traces = audit.traces(request.get('state_comparisons', []))
        checks = {'schedules': schedules['status'], 'ce_gradients': gradients['status'], 'outside_mask_gradients': gradients['outside_mask_status'], **traces['statuses'], 'suffix_probe': 'unavailable_in_this_analyzer_version', 'stock_export_compatibility': 'unavailable_in_this_analyzer_version'}
        required = request.get('required_checks', list(checks))
        require(all(name in checks for name in required), 'unknown_required_check')
        status = 'passed_requested_checks' if required and all(checks[name] == 'passed' for name in required) else 'partial_evidence'
        audit.finish()
        report = {'status': status, 'scope': 'independent_cpu_pilot_qualification_evidence_analysis_only', 'checks': checks, 'required_checks': required, 'schedules': schedules, 'gradients': gradients, 'state_comparisons': traces, 'scientific_training_authorized': False, 'project_helpers_imported': False}
    except Exception as error:
        report = {'status': 'failed', 'phase': audit.phase, 'error_code': str(error) if isinstance(error, AuditFailure) else type(error).__name__, 'scientific_training_authorized': False}
    report.update(audit_script_sha256=digest(Path(__file__)), slurm_job_id=os.environ['SLURM_JOB_ID'], cpu_only=True, wall_seconds=time.monotonic() - start, artifact_inventory=audit.inventory, artifact_inventory_sha256=identity(audit.inventory))
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': digest(output), 'report_path': str(output)}, sort_keys=True))
    return 0 if report['status'] in ('passed_requested_checks', 'partial_evidence') else 1


if __name__ == '__main__':
    sys.exit(main())
