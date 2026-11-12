import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time


V1_SHA256 = '32f142ee0142b54d6dd2cd4f0650656ff06d3fba609f55b40ab7ad3f87bce85b'
RUNTIME_PROBE_SHA256 = '46573f893aff54ab1606040facc098e4a98f961c460621a4348bd28cb065d431'
V1_PATH = Path(__file__).with_name('independent_pilot_qualification_audit_20261112.py')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


if digest(V1_PATH) != V1_SHA256:
    raise ValueError('frozen_v1_analyzer_changed')
spec = importlib.util.spec_from_file_location('frozen_independent_pilot_v1', V1_PATH)
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)
require = v1.require


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def semantic_identity(value, torch):
    result = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            result.update(canonical({'dtype': str(tensor.dtype), 'shape': list(tensor.shape)}))
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, torch.dtype):
            result.update(canonical(['torch.dtype', str(item)]))
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                result.update(canonical([type(key).__name__, key]))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(canonical([type(item).__name__, len(item)]))
            for child in item:
                visit(child)
        elif type(item).__module__.startswith('deepspeed.runtime.fp16.loss_scaler'):
            result.update(canonical([type(item).__module__, type(item).__qualname__]))
            visit(vars(item))
        elif (type(item).__module__, type(item).__qualname__) == ('deepspeed.utils.tensor_fragment', 'fragment_address'):
            require(set(vars(item)) == {'numel', 'start'}, 'unexpected_fragment_address_schema')
            result.update(canonical([type(item).__module__, type(item).__qualname__]))
            visit(vars(item))
        else:
            result.update(canonical([type(item).__name__, item]))
    visit(value)
    return result.hexdigest()


class Audit(v1.Audit):
    def suffix(self, qualification, torch):
        self.phase = 'independent_suffix_invariance'
        cases, coverage, trace_coverage = [], set(), set()
        for policy in ('original', 'target'):
            for rank in range(8):
                normal = qualification / policy / 'uninterrupted/attempts/full'
                alternate = qualification / policy / 'alternate-pad/attempts/alternate'
                for step in range(4):
                    name = 'rank-' + str(rank) + '-step-' + str(step) + '-gradients.pt'
                    if not (normal / name).exists() or not (alternate / name).exists():
                        continue
                    a = torch.load(self.artifact(str(normal / name)), map_location='cpu', weights_only=True)
                    b = torch.load(self.artifact(str(alternate / name)), map_location='cpu', weights_only=True)
                    for key in ('rank', 'step', 'domain', 'policy', 'example_id', 'cache_index', 'plan_identity_sha256', 'real_length', 'base_positions', 'head_grid_indices', 'quota', 'loss_scale', 'normalization', 'head_weights'):
                        require(a[key] == b[key], 'suffix_probe_binding_or_indices_mismatch')
                    require(a['alternate_pad'] is False and b['alternate_pad'] is True, 'suffix_probe_did_not_use_alternate_branch')
                    length = a['real_length']
                    require(torch.equal(a['input_ids'][:, :length], b['input_ids'][:, :length]), 'suffix_probe_changed_real_input_prefix')
                    changed = length < 2048 and not torch.equal(a['input_ids'][:, length:], b['input_ids'][:, length:])
                    require(torch.equal(a['teacher_full_vocab_logits_original_s_plus_one'], b['teacher_full_vocab_logits_original_s_plus_one']), 'suffix_probe_changed_guarded_teacher_logits')
                    for key in ('head_logits_fp16_before_float_cast', 'head_logit_gradients_fp16_scaled128_before_unscale_and_dp_reduce'):
                        require(len(a[key]) == len(b[key]) == 7 and all(torch.equal(x, y) for x, y in zip(a[key], b[key])), 'suffix_probe_changed_selected_head_logits_or_gradients')
                    errors = [abs(float(x) - float(y)) * a['normalization'] for x, y in zip(a['native_head_mean_losses'], b['native_head_mean_losses'])]
                    require(max(errors) <= 5e-5 and abs(a['actual_normalized_total_loss'] - b['actual_normalized_total_loss']) <= 5e-5, 'suffix_probe_ce_bound_failed')
                    if changed:
                        coverage.add((a['domain'], policy))
                    cases.append({'policy': policy, 'rank': rank, 'step': step, 'domain': a['domain'], 'actual_suffix_changed': changed, 'maximum_scaled_head_ce_difference': max(errors), 'guarded_logits_and_scaled_logit_gradients_tensor_equal': True})
                    del a, b
                descriptors = []
                for directory in (normal, alternate):
                    descriptors.append({'rank': rank, 'segments': [{key: {'path': str(directory / ('rank-' + str(rank) + suffix))} for key, suffix in (('initial', '.initial.json'), ('journal', '.jsonl'), ('complete', '.complete.json'))}]})
                if all(Path(item['path']).exists() for descriptor in descriptors for segment in descriptor['segments'] for item in segment.values()):
                    left, right = (self.stream(descriptor) for descriptor in descriptors)
                    require(left['initial'] == right['initial'] and set(left['updates']) == set(right['updates']) == set(range(4)), 'suffix_probe_initial_state_or_step_coverage_mismatch')
                    for step in range(4):
                        a, b = left['updates'][step], right['updates'][step]
                        for key in ('state_identity', 'pre_step_scaled_gradient_partition_sha256', 'learning_rates_before_step', 'local_fp32_master_update_l2'):
                            require(a[key] == b[key], 'suffix_probe_changed_full_partition_state_or_update_trace')
                    require(left['final']['state_identity'] == right['final']['state_identity'], 'suffix_probe_changed_final_state_trace')
                    trace_coverage.add((policy, rank))
        wanted = {(domain, policy) for domain in v1.DOMAINS for policy in ('original', 'target')}
        complete = len(cases) == 64 and coverage == wanted and len(trace_coverage) == 16
        return {'status': 'passed' if complete else 'partial_evidence' if cases else 'unavailable', 'cases': cases, 'trace_pairs_compared': len(trace_coverage), 'missing_actual_changed_suffix_domain_policies': sorted(wanted - coverage), 'bounds': 'At fixed production shape, retained real-prefix teacher/head logits and scaled logit gradients require tensor equality, treating signed zero as equal; normalized head and total CE absolute difference<=5e-5. Full owned pre-step partition/state/update traces require exact equality. A numerical mismatch is not waived by the former decoder policy.'}

    def raw_checkpoint(self, checkpoint, torch, plan=None, expected_manifest=None):
        self.phase = 'raw_checkpoint_reconstruction'
        manifest_descriptor = {'path': str(checkpoint / 'manifest.json')}
        if expected_manifest:
            manifest_descriptor['sha256'] = expected_manifest
        manifest = self.json(manifest_descriptor)
        if 'identity_sha256' in manifest:
            v1.verified_identity(manifest)
        require(manifest['world_size'] == 8, 'raw_checkpoint_world_size_mismatch')
        if plan:
            require(manifest['plan_identity_sha256'] == plan['identity_sha256'] and manifest['source_sha256'] == plan['source_sha256'] and manifest['hardware'] == plan['hardware'] and manifest['trajectory']['qualification'] is True, 'raw_checkpoint_study_trajectory_mismatch')
        for name, expected in manifest['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe_raw_checkpoint_inventory')
            self.artifact({'path': str(checkpoint / name), 'sha256': expected if isinstance(expected, str) else expected['sha256']})
        models = [name for name in manifest['files'] if name.endswith('model_states.pt')]
        require(len(models) == 1, 'raw_zero2_model_state_coverage_mismatch')
        saved = torch.load(checkpoint / models[0], map_location='cpu', weights_only=False)
        if plan:
            require(all(saved['ds_config'].get(key) == value for key, value in plan['recipe']['ds_config'].items()), 'raw_deepspeed_recipe_differs')
            require(saved['mp_world_size'] == 1 and saved['dp_world_size'] == 8 and saved['global_samples'] == saved['global_steps'] * 8, 'raw_engine_sample_or_parallel_geometry_mismatch')
            require(saved['lr_scheduler'] == {'last_batch_iteration': saved['global_steps'] - 1}, 'raw_warmup_decay_scheduler_cursor_mismatch')
        module = saved['module']
        shapes = saved['param_shapes']
        names = [name for group in shapes for name in group]
        require(len(names) == len(set(names)) and all(name in module for name in names), 'raw_model_parameter_shapes_mismatch')
        require(not any(name.startswith(('target_model.', 'embed_tokens.')) for name in names), 'raw_checkpoint_contains_frozen_trainable')
        selected = {name: module[name] for name in names}
        if plan:
            require({'d2t', 't2d'} <= set(module), 'raw_pilot_vocabulary_missing')
            selected.update({key: module[key] for key in ('d2t', 't2d')})
            mapping = module['d2t'] + torch.arange(32000)
            require(module['d2t'].dtype == torch.long and module['t2d'].dtype == torch.bool and module['t2d'].shape == (128256,) and torch.equal(mapping, torch.nonzero(module['t2d'], as_tuple=True)[0]), 'raw_checkpoint_vocabulary_invalid')
            require(semantic_identity({key: module[key] for key in ('d2t', 't2d')}, torch) == plan['vocabulary_identity_sha256'], 'raw_checkpoint_vocabulary_identity_mismatch')
        model_hash = semantic_identity(selected, torch)
        scheduler_hash = semantic_identity(saved['lr_scheduler'], torch)
        optimizer_files = {}
        for name in manifest['files']:
            match = re.search(r'zero_pp_rank_(\d+)_mp_rank_00_optim_states\.pt$', name)
            if match:
                require(int(match.group(1)) not in optimizer_files, 'duplicate_raw_optimizer_partition')
                optimizer_files[int(match.group(1))] = name
        require(set(optimizer_files) == set(range(8)), 'raw_optimizer_partition_coverage_mismatch')
        results, masters = [], []
        for rank in range(8):
            state = torch.load(checkpoint / ('rank-' + str(rank) + '.pt'), map_location='cpu', weights_only=False)
            expected = state['state_identity']
            require(state.get('next_step', manifest.get('global_step')) == manifest.get('next_step', manifest.get('global_step')), 'raw_rank_cursor_mismatch')
            require(model_hash == expected.get('model', expected.get('trainable_model')) and scheduler_hash == expected['scheduler'], 'raw_model_or_scheduler_identity_mismatch')
            value = torch.load(checkpoint / optimizer_files[rank], map_location='cpu', weights_only=False)
            optimizer = value['optimizer_state_dict']
            optimizer_hash = semantic_identity(optimizer, torch)
            require(optimizer_hash == expected.get('optimizer_scaler', expected.get('optimizer_and_scaler')), 'raw_optimizer_adam_scaler_identity_mismatch')
            require(int(optimizer['zero_stage']) == 2 and optimizer['partition_count'] == [8] * len(shapes), 'raw_optimizer_zero_partition_contract_mismatch')
            parts = optimizer['single_partition_of_fp32_groups']
            require(len(parts) == len(shapes) and all(part.dtype == torch.float32 and part.ndim == 1 for part in parts), 'raw_fp32_master_partition_geometry_mismatch')
            masters.append(parts)
            raw_rng = state['rng']
            if not isinstance(raw_rng['numpy'][1], list):
                raw_rng = dict(raw_rng)
                raw_rng['numpy'] = [raw_rng['numpy'][0], raw_rng['numpy'][1].tolist(), *raw_rng['numpy'][2:]]
            rng_hash = semantic_identity(raw_rng, torch)
            if 'rng' in expected:
                require(rng_hash == expected['rng'], 'raw_rng_identity_mismatch')
            require(saved['global_steps'] == expected['global_steps'] and saved['skipped_steps'] == expected['skipped_steps'] == 0 and saved['dp_world_size'] == 8, 'raw_global_counter_mismatch')
            moment_nonzero = {}
            for key, item in optimizer['base_optimizer_state']['state'].items():
                require({'step', 'exp_avg', 'exp_avg_sq'} <= set(item), 'raw_adam_moment_schema_mismatch')
                require(int(item['step']) == saved['global_steps'], 'raw_adam_step_counter_mismatch')
                moment_nonzero[str(key)] = {'exp_avg_nonzero': bool(torch.count_nonzero(item['exp_avg'])), 'exp_avg_sq_nonzero': bool(torch.count_nonzero(item['exp_avg_sq'])), 'step': int(item['step'])}
            if plan:
                require(optimizer['dynamic_loss_scale'] is False and float(optimizer['loss_scaler'].cur_scale) == 128 and optimizer['overflow'] is False and optimizer['clip_grad'] == 0.5, 'raw_static_scaler_or_clipping_mismatch')
                require(moment_nonzero and all(item['exp_avg_nonzero'] and item['exp_avg_sq_nonzero'] for item in moment_nonzero.values()), 'new_qualification_has_zero_adam_moments')
                for group in optimizer['base_optimizer_state']['param_groups']:
                    require(tuple(group['betas']) == (0.9, 0.95) and group['eps'] == 1e-8 and group['weight_decay'] == 0, 'raw_adam_recipe_mismatch')
                    scheduler = plan['recipe']['ds_config']['scheduler']['params']
                    iteration = saved['lr_scheduler']['last_batch_iteration']
                    gamma = iteration / scheduler['warmup_num_steps'] if iteration < scheduler['warmup_num_steps'] else max(0, (scheduler['total_num_steps'] - iteration) / (scheduler['total_num_steps'] - scheduler['warmup_num_steps']))
                    learning_rate = scheduler['warmup_min_lr'] + gamma * (scheduler['warmup_max_lr'] - scheduler['warmup_min_lr'])
                    require(abs(group['lr'] - learning_rate) <= max(1e-15, abs(learning_rate) * 1e-12), 'raw_optimizer_learning_rate_schedule_mismatch')
            results.append({'rank': rank, 'model': model_hash, 'optimizer_scaler': optimizer_hash, 'scheduler': scheduler_hash, 'rng': rng_hash, 'global_steps': saved['global_steps'], 'skipped_steps': saved['skipped_steps'], 'micro_steps_from_rank_checkpoint': expected['micro_steps'], 'adam_moment_nonzero': moment_nonzero})
            del value, optimizer, state
            gc.collect()
        for group_index, group in enumerate(shapes):
            flat = torch.cat([rank[group_index] for rank in masters])
            required_elements = sum(math.prod(shape) for shape in group.values())
            require(flat.numel() == required_elements, 'unqualified_fp32_master_padding_layout')
            offset = 0
            for name, shape in group.items():
                elements = math.prod(shape)
                expected = flat[offset:offset + elements].reshape(shape).to(module[name].dtype)
                require(tuple(module[name].shape) == tuple(shape) and torch.equal(expected, module[name]), 'raw_fp32_master_and_fp16_module_disagree')
                offset += elements
            del flat
        result = {'status': 'passed_raw_state_reconstruction', 'path': str(checkpoint), 'manifest_sha256': digest(checkpoint / 'manifest.json'), 'next_step': manifest.get('next_step', manifest.get('global_step')), 'ranks': results, 'fp32_partition_reassembly_matches_visible_model': True, 'components': ['raw FP16 trainable model', 'all eight raw FP32 master partitions', 'all Adam states and scaler fields', 'raw scheduler', 'raw rank RNG', 'raw engine global/skipped counters', 'rank-saved micro-step cursor'], 'mode': 'pilot_qualification' if plan else 'old_format_reconstruction_probe', 'scientific_training_admitted': False}
        del saved, selected, module, masters
        gc.collect()
        return result

    def raw_qualifications(self, qualification, torch):
        results, by_key = [], {}
        for policy in ('original', 'target'):
            plans = [value for value in self.plans.values() if value['seed'] == 0 and value['cohort'] == 'generalist' and value['policy'] == policy]
            require(len(plans) == 1, 'raw_qualification_plan_coverage_mismatch')
            plan = plans[0]
            for variant in ('uninterrupted', 'native-reference', 'split', 'alternate-pad'):
                latest_path = qualification / policy / variant / 'latest.json'
                if not latest_path.exists():
                    continue
                latest = self.json(str(latest_path))
                checkpoint = Path(latest['path'])
                result = self.raw_checkpoint(checkpoint, torch, plan, latest['manifest_sha256'])
                require(result['next_step'] == 4, 'qualification_raw_final_cursor_mismatch')
                by_key[policy, variant, 4] = result
                results.append(result)
                if variant == 'split':
                    prefix = checkpoint.parent / 'step-2'
                    require((prefix / 'manifest.json').is_file(), 'qualification_raw_split_prefix_missing')
                    result = self.raw_checkpoint(prefix, torch, plan)
                    require(result['next_step'] == 2, 'qualification_raw_prefix_cursor_mismatch')
                    by_key[policy, variant, 2] = result
                    results.append(result)
            if (policy, 'uninterrupted', 4) not in by_key:
                continue
            control = by_key[policy, 'uninterrupted', 4]
            for variant in ('native-reference', 'split', 'alternate-pad'):
                if (policy, variant, 4) in by_key:
                    require(control['ranks'] == by_key[policy, variant, 4]['ranks'], 'raw_native_resume_or_suffix_final_states_differ')
            if (policy, 'split', 2) in by_key:
                for rank, state in enumerate(by_key[policy, 'split', 2]['ranks']):
                    path = qualification / policy / 'uninterrupted/attempts/full' / ('rank-' + str(rank) + '.jsonl')
                    rows = [row for row in v1.read_lines(self.artifact(str(path))) if row['event'] in ('committed_update', 'successful_update') and row['step'] == 1]
                    require(len(rows) == 1, 'raw_prefix_control_trace_missing')
                    expected = rows[0]['state_identity']
                    require(all(state[key] == expected[key] for key in ('model', 'optimizer_scaler', 'scheduler', 'rng', 'global_steps', 'skipped_steps')) and state['micro_steps_from_rank_checkpoint'] == expected['micro_steps'], 'raw_split_prefix_differs_from_uninterrupted_state')
        return {'status': 'passed' if len(results) == 10 else 'partial_evidence' if results else 'unavailable', 'checkpoints': results, 'limitation': 'Frozen teacher/embedding weights are intentionally absent from trainable checkpoints and remain bound to prepared assets. Micro-steps are reconstructed from their dedicated raw rank-state field and checked against observed traces; DeepSpeed model files do not serialize that field.'}

    def stock(self, qualification, torch):
        self.phase = 'independent_export_tensor_and_generation_compatibility'
        results, exported = [], {}
        for policy in ('original', 'target'):
            for rank in range(8):
                path = qualification / policy / 'stock-inference' / ('rank-' + str(rank) + '.json')
                if not path.exists():
                    continue
                receipt = self.json(str(path))
                require(receipt['status'] == 'passed_stock_export_generation' and receipt['rank'] == rank, 'stock_generation_receipt_scope_mismatch')
                require(receipt['plan_identity_sha256'] in self.plans, 'stock_plan_not_in_registered_study')
                plan = self.plans[receipt['plan_identity_sha256']]
                require(plan['policy'] == policy and plan['seed'] == 0 and plan['cohort'] == 'generalist', 'stock_response_policy_mismatch')
                export = Path(receipt['export_path'])
                expected_export = (self.root / 'models').resolve() / 'trained/pilot-qualifications' / (qualification / policy / 'uninterrupted').relative_to(self.root / 'outputs') / 'exports/step-4'
                require(export == expected_export, 'stock_export_namespace_mismatch')
                export_receipt = self.json({'path': str(export / 'export.json'), 'sha256': receipt['export_receipt_sha256']})
                require(export_receipt['plan_identity_sha256'] == plan['identity_sha256'] and export_receipt['state_identity_sha256'] == receipt['export_state_identity_sha256'], 'stock_export_identity_mismatch')
                for name, item in export_receipt['files'].items():
                    self.artifact({'path': str(export / name), **item})
                if str(export) not in exported:
                    initial = self.json(str(qualification / policy / 'uninterrupted/attempts/full/rank-0.initial.json'))
                    complete = self.json(str(qualification / policy / 'uninterrupted/attempts/full/rank-0.complete.json'))
                    require(initial['plan_identity_sha256'] == complete['plan_identity_sha256'] == plan['identity_sha256'], 'stock_raw_update_plan_binding_mismatch')
                    state = torch.load(export / 'pytorch_model.bin', map_location='cpu', weights_only=True)
                    names = [*initial['trainable_names'], 'd2t', 't2d']
                    require(all(name in state for name in names) and not any(name.startswith('target_model.') for name in state), 'stock_export_missing_trainables_or_contains_teacher')
                    subset = {name: state[name] for name in names}
                    actual = semantic_identity(subset, torch)
                    require(actual == export_receipt['state_identity_sha256'] == complete['state_identity']['model'], 'independent_export_tensor_identity_mismatch')
                    require(semantic_identity({key: state[key] for key in ('d2t', 't2d')}, torch) == plan['vocabulary_identity_sha256'], 'independent_export_vocabulary_identity_mismatch')
                    exported[str(export)] = actual
                    del state, subset
                    gc.collect()
                source_files = receipt['stock_source_files']
                require(source_files, 'stock_source_inventory_empty')
                for name, expected in source_files.items():
                    self.artifact({'path': str(self.root / 'repos/EAGLE' / name), 'sha256': expected})
                prompt, speculative, autoregressive = (receipt[name] for name in ('prompt_ids', 'speculative_ids', 'autoregressive_ids'))
                require(prompt and all(type(value) is int and 0 <= value < 128256 for value in [*prompt, *speculative, *autoregressive]), 'stock_token_output_schema_invalid')
                require(len(speculative) > len(prompt) and len(autoregressive) > len(prompt) and speculative[:len(prompt)] == autoregressive[:len(prompt)] == prompt, 'stock_did_not_generate_actual_continuations')
                require(receipt['tokens_equal'] == (speculative == autoregressive) and math.isfinite(receipt['generation_wall_seconds']) and receipt['generation_wall_seconds'] > 0, 'stock_generation_metadata_inconsistent')
                results.append({'policy': policy, 'rank': rank, 'status': 'passed_compatibility_only', 'export_state_sha256': exported[str(export)], 'speculative_continuation_tokens': len(speculative) - len(prompt), 'autoregressive_continuation_tokens': len(autoregressive) - len(prompt), 'tokens_equal': receipt['tokens_equal'], 'stock_source_files': len(source_files)})
        return {'status': 'passed' if len(results) == 16 and len(exported) == 2 else 'partial_evidence' if results else 'unavailable', 'cases': results, 'scope': 'Exact exported trainable/vocabulary identity and genuine stock continuation on one synthetic prompt per rank. Does not require greedy token equality or assert quality, speedup, universal losslessness or a matched stopping benchmark.'}

    def runtime_envelope(self, pre, post, execution_envelope=None):
        self.phase = 'independent_runtime_envelope'
        if pre is None or post is None:
            return {'status': 'unavailable'}
        a, b = self.json(str(pre)), self.json(str(post))
        v1.verified_identity(a)
        v1.verified_identity(b)
        require(a['phase'] == 'pre' and b['phase'] == 'post' and a['status'] == b['status'] == 'cpu_runtime_envelope_observation', 'runtime_envelope_phase_mismatch')
        require(a['study_manifest_sha256'] == b['study_manifest_sha256'], 'runtime_envelope_study_mismatch')
        plan = next(iter(self.plans.values()))
        require(a['study_manifest_sha256'] == digest(Path(plan['study_directory']) / 'manifest.json'), 'runtime_observers_belong_to_another_study')
        require(a['probe_source_sha256'] == b['probe_source_sha256'] == RUNTIME_PROBE_SHA256, 'runtime_observer_source_not_reviewed')
        require(a['python'] == b['python'], 'runtime_interpreter_or_platform_changed')
        for key in ('package_versions', 'module_files', 'shared_environment_libraries', 'source_files', 'upstream_files', 'environment'):
            require(a[key] == b[key], 'runtime_environment_library_changed_between_probes')
        for receipt in (a, b):
            for key in ('module_files', 'shared_environment_libraries', 'source_files', 'upstream_files', 'fused_adam_inventory'):
                for item in receipt[key].values():
                    path = Path(item['path'])
                    require(path.is_file() and path.stat().st_size == item['size_bytes'] and digest(path) == item['sha256'], 'runtime_observed_file_no_longer_matches')
                    self.inventory[str(path)] = {'sha256': item['sha256'], 'size_bytes': item['size_bytes']}
        before, after = a['fused_adam_inventory'], b['fused_adam_inventory']
        require(set(before) <= set(after) and all(after[key] == value for key, value in before.items()), 'existing_fused_adam_artifact_changed_or_disappeared')
        binaries = [key for key in after if key.endswith('.so')]
        require(binaries, 'post_runtime_has_no_fused_adam_binary')
        temporal = None
        if execution_envelope:
            from datetime import datetime
            temporal = self.json(str(execution_envelope))
            require(str(temporal['pre_job']['job_id']) == a['slurm_job_id'] and str(temporal['post_job']['job_id']) == b['slurm_job_id'], 'runtime_envelope_accounting_job_mismatch')
            for key in ('pre_job', 'qualification_job', 'post_job'):
                require(temporal[key]['state'] == 'COMPLETED' and temporal[key]['exit_code'] == '0:0', 'runtime_envelope_job_not_successful')
            times = [datetime.fromisoformat(temporal[key][edge].replace('Z', '+00:00')) for key, edge in (('pre_job', 'end_utc'), ('qualification_job', 'start_utc'), ('qualification_job', 'end_utc'), ('post_job', 'start_utc'))]
            require(all(value.tzinfo is not None for value in times) and times == sorted(times), 'runtime_observer_interval_not_outside_qualification')
            require(str(temporal['qualification_job']['job_id']) in temporal['post_job']['afterok_jobs'] and a['slurm_job_id'] in temporal['qualification_job']['afterok_jobs'], 'runtime_envelope_afterok_dependencies_missing')
        return {'status': 'passed' if temporal else 'partial_evidence', 'temporal_accounting_verified': temporal is not None, 'package_versions': a['package_versions'], 'unchanged_preexisting_fused_adam_files': len(before), 'new_fused_adam_files': sorted(set(after) - set(before)), 'new_compilation_observed': bool(set(after) - set(before)), 'limitations': ['CPU pre/post observations establish the shared environment envelope, not GPU-node system executable/driver binary identity or absence of an adversarial transient change.', 'Actual GPU backend flags and hardware bindings remain the source-bound worker/scan observations.', 'A newly created compiled extension is reported as new and is not described as preexisting unchanged.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--raw-checkpoint', type=Path)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--runtime-pre', type=Path)
    parser.add_argument('--runtime-post', type=Path)
    parser.add_argument('--execution-envelope', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'v2_qualification_audit_requires_cpu_slurm')
    if args.study:
        path = args.study / 'manifest.json' if args.study.is_dir() else args.study
        root = Path(json.loads(path.read_bytes())['common']['project_root']).resolve()
        require(args.qualification is not None, 'qualification_directory_required')
        if args.runtime_pre is None and (args.qualification / 'runtime/pre.json').is_file():
            args.runtime_pre = args.qualification / 'runtime/pre.json'
        if args.runtime_post is None and (args.qualification / 'runtime/post.json').is_file():
            args.runtime_post = args.qualification / 'runtime/post.json'
        if args.execution_envelope is None and (args.qualification / 'runtime/interval-accounting.json').is_file():
            args.execution_envelope = args.qualification / 'runtime/interval-accounting.json'
    else:
        require(args.root is not None and args.raw_checkpoint is not None and args.manifest_sha256 is not None, 'raw_probe_requires_root_checkpoint_and_manifest_identity')
        root = args.root.resolve()
    output = args.output.resolve() if args.output.suffix == '.json' else args.output.resolve() / 'report.json'
    require(output.is_relative_to(root / 'outputs') and not output.exists(), 'choose_fresh_v2_report_under_outputs')
    import torch
    import deepspeed
    torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    audit, start = Audit(root), time.monotonic()
    try:
        if args.study:
            request = audit.automatic_request(args.study.resolve(), args.qualification.resolve())
            schedules = audit.schedules(request['study_manifest'])
            gradients = audit.gradients(request['gradient_cases'], torch)
            traces = audit.traces(request['state_comparisons'])
            suffix = audit.suffix(args.qualification.resolve(), torch)
            raw = audit.raw_qualifications(args.qualification.resolve(), torch)
            stock = audit.stock(args.qualification.resolve(), torch)
            runtime = audit.runtime_envelope(args.runtime_pre, args.runtime_post, args.execution_envelope)
            checks = {'schedules': schedules['status'], 'ce_gradients': gradients['status'], 'outside_mask_gradients': gradients['outside_mask_status'], **traces['statuses'], 'suffix_probe': suffix['status'], 'raw_zero_state_reconstruction': raw['status'], 'stock_export_compatibility': stock['status'], 'runtime_envelope': runtime['status']}
            report = {'status': 'all_checked_evidence_passed_requires_independent_review' if all(value == 'passed' for value in checks.values()) else 'partial_evidence', 'scope': 'independent_full_cpu_pilot_qualification_analysis_not_admission', 'checks': checks, 'schedules': schedules, 'gradients': gradients, 'state_comparisons': traces, 'suffix': suffix, 'raw_states': raw, 'stock': stock, 'runtime': runtime}
        else:
            raw = audit.raw_checkpoint(args.raw_checkpoint.resolve(), torch, expected_manifest=args.manifest_sha256)
            report = {'status': 'passed_old_format_raw_reconstruction_probe', 'scope': 'format_validation_only_not_new_pilot_qualification', 'raw_state': raw}
        audit.finish()
    except Exception as error:
        report = {'status': 'failed', 'phase': audit.phase, 'error_code': str(error) if isinstance(error, v1.AuditFailure) else type(error).__name__}
    report.update(scientific_training_admitted=False, automatic_admission_emitted=False, artifact_inventory=audit.inventory, artifact_inventory_sha256=v1.identity(audit.inventory), analyzer_v1_sha256=V1_SHA256, analyzer_v2_sha256=digest(Path(__file__)), slurm_job_id=os.environ['SLURM_JOB_ID'], cpu_only=True, wall_seconds=time.monotonic() - start, project_helpers_imported=False, runtime_packages={name: importlib.metadata.version(name) for name in ('torch', 'deepspeed', 'numpy', 'transformers', 'accelerate')})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': digest(output), 'output': str(output)}), flush=True)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
