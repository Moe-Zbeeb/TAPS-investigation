import argparse
import copy
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


V2_SHA256 = '5e11dd792adb840be1e861f6a0f2e235d483b977e7dca3cfef31f78e3f0f4091'
V2_PATH = Path(__file__).with_name('independent_pilot_qualification_audit_v2_20261112.py')
SOURCES = {
    'native_full_body.py': '40109aae2cdd5b06fa601220175569823f94e0e63d69c44942e93854740046c1',
    'native_full_body_queue.py': '1cbd366356bfcfe308aa2cdbd02c4a839b0abcb3b45b6b07db68312c58a95d5b',
    'native_gradients.py': '594ee0ee196f9deceecdfe7af7a2bda6b8ee728f2aa9c222748689755c4f4d1a',
    'native_training.py': '2aad19f7aefddfb70dd02b78d2eaf9709c90834d2308d055a836be83de5fec51',
    'native_recovery.py': '4293c07e350a4bafbb18520050888daf7a7d84a7760c5da9e1e7db7649a34c10',
    'pilot_cache.py': 'f9ad06ae2d754c4485bbd22f3e8b98f65ab9706877f40b8961e9f4f1e706ea5e',
    'pool.py': 'b00f1930c4eb74ab1312426978487162f6105e5c8b2b478ae0d5894b9898b8c7',
}
NAMES = {
    'fc.weight', 'norm.weight', 'lm_head.weight',
    'midlayer.hidden_norm.weight', 'midlayer.input_layernorm.weight',
    'midlayer.post_attention_layernorm.weight',
    'midlayer.self_attn.q_proj.weight', 'midlayer.self_attn.k_proj.weight',
    'midlayer.self_attn.v_proj.weight', 'midlayer.self_attn.o_proj.weight',
    'midlayer.mlp.gate_proj.weight', 'midlayer.mlp.up_proj.weight',
    'midlayer.mlp.down_proj.weight',
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


if digest(V2_PATH) != V2_SHA256:
    raise ValueError('frozen_semantic_hasher_changed')
spec = importlib.util.spec_from_file_location('independent_native_frozen_v2', V2_PATH)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
v1 = v2.v1
require = v1.require


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def verified_plan(value):
    require(identity({key: item for key, item in value.items() if key != 'plan_identity_sha256'}) == value['plan_identity_sha256'], 'native_plan_self_identity_mismatch')
    return value


def rng_identity(value, torch):
    value = dict(value)
    state = value['numpy']
    value['numpy'] = [state[0], state[1].tolist() if not isinstance(state[1], list) else state[1], *state[2:]]
    return v2.semantic_identity(value, torch)


def native_learning_rate(iteration):
    gamma = math.log(iteration + 1) / math.log(12000) if 0 <= iteration < 12000 else 0 if iteration < 0 else max(0.0, (800000 - iteration) / (800000 - 12000))
    return 5e-5 * gamma


class Audit(v2.Audit):
    def runtime_sources(self):
        import deepspeed
        package = Path(deepspeed.__file__).resolve().parent
        require(package.is_relative_to((self.root / 'envs/native-eagle3').resolve()), 'native_audit_runtime_outside_declared_environment')
        expected = {'__init__.py': 'b5c4b7797c761ef30baf7f3fa1f83ed910bd97d935b3702c617f93c89101ac95', 'runtime/config.py': '1ed3a84c5be3afff4efc07687de4113bec82e8df097a9f23de69b041f3cfb52e', 'runtime/engine.py': '3bb0e2af63f93c105d51f120bd56886b7d45e4ad33d9cd192e5c687c8673fde8', 'ops/adam/fused_adam.py': 'e7f24193f52f04572738bd3f84faf94951404e69a6287148955491d5a9beb017', 'runtime/lr_schedules.py': '6482b41cc5bf6d8e883020ca8fa5e7a9b1aa44f27e6f93873d12e07e573945d2'}
        for name, wanted in expected.items():
            path = package / name
            require(path.resolve() == path and digest(path) == wanted, 'native_optimizer_or_scheduler_implementation_not_reviewed')
            self.inventory[str(path)] = {'sha256': wanted, 'size_bytes': path.stat().st_size}
        return expected

    def plans_and_data(self, preparation):
        self.phase = 'native_source_plan_and_data_binding'
        queue = self.json(str(preparation / 'queue-plan.json'))
        require(queue['status'] == 'prepared_corrected_native_qualifications_and200_profile' and queue['scientific_training_admitted'] is False and set(queue['plans']) == {'uninterrupted', 'direct-reference', 'split', 'profile'}, 'native_queue_scope_mismatch')
        self.native_plans, self.preparation = {}, preparation
        for label, descriptor in queue['plans'].items():
            plan = verified_plan(self.json(descriptor))
            policy = plan['config']['full_body_v3']
            require(policy['source_sha256'] == queue['source_sha256'] and all(policy['source_sha256'][name] == value for name, value in SOURCES.items()), 'native_source_not_reviewed')
            require(set(policy['source_sha256']) == {'native_full_body.py', 'native_full_body_queue.py', 'native_gradients.py', 'native_training.py', 'native_recovery.py', 'pilot_cache.py', 'pool.py'}, 'native_source_inventory_incomplete')
            for name, expected in plan['frozen_files'].items():
                require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe_native_frozen_inventory')
                self.artifact({'path': str(Path(plan['run_dir']) / name), 'sha256': expected})
            require(all(plan['frozen_files']['inputs/' + name] == expected for name, expected in policy['source_sha256'].items()), 'native_frozen_source_binding_mismatch')
            profile = label == 'profile'
            require(plan['scope'] == 'native_training' and plan['draft'] is None and plan['config']['initialize_from_released_draft'] is False and plan['config']['world_size'] == 8, 'native_not_eight_rank_scratch')
            require(plan['config']['budget'] == {'optimizer_steps': 200 if profile else 4} and plan['config']['pipeline_stage'] == ('corrected-profile-v3' if profile else 'corrected-qualification-v3'), 'native_stage_budget_mismatch')
            require(plan['config']['recovery'] == {'checkpoint_interval': 100 if profile else 2, 'stop_after_optimizer_step': 2 if label == 'split' else None}, 'native_recovery_protocol_mismatch')
            require(policy['maximum_attempted_optimizer_steps'] == (512 if profile else 32) and policy['capture_gradient_partitions'] is (not profile) and policy['checkpoint_mode'] == ('off_reference' if label == 'direct-reference' else 'non_reentrant') and policy['rotary_capacity'] == 2054, 'native_corrected_policy_mismatch')
            require(str(Path(plan['run_dir']) / 'inputs/native_full_body.py') in plan['command'] and plan['native_recipe']['epochs'] == 40, 'native_wrapper_or_epoch_contract_mismatch')
            qa = self.json(policy['vocabulary_audit'])
            parent = verified_plan(self.json(policy['data_parent_plan']))
            require(qa['status'] == 'passed' and qa['vocabulary_uses_train_only'] is True and policy['vocabulary_audit']['sha256'] == parent['frozen_files']['inputs/vocabulary-audit.json'], 'native_training_only_vocabulary_provenance_mismatch')
            self.artifact(policy['vocabulary_cache'])
            require(policy['vocabulary_cache']['sha256'] == parent['frozen_files']['work/cache.pt'] == plan['frozen_files']['work/cache.pt'], 'native_vocabulary_cache_binding_mismatch')
            for role in ('train', 'validation'):
                require(parent[role]['sha256'] == policy['full_data_sha256'][role], 'native_full_data_parent_mismatch')
                if profile:
                    require(plan[role]['sha256'] == policy['full_data_sha256'][role], 'native_profile_not_full_corpus')
            tokenization = self.json(str(Path(plan['run_dir']) / 'tokenization-audit.json'))
            require(tokenization['status'] == 'passed' and tokenization['plan_identity_sha256'] == plan['plan_identity_sha256'], 'native_actual_tokenization_audit_missing')
            if not profile:
                train = tokenization['splits']['train']
                require(train['rows_before_filter'] == train['rows_after_official_filter'] == 64 and train['minimum_length'] == train['maximum_length'] == 2048, 'native_maximum_shape_not_exercised')
            for name, item in plan['target']['files'].items():
                self.artifact({'path': str(Path(plan['target']['path']) / name), **item})
            self.native_plans[label] = plan
        require(all(plan['config']['full_body_v3']['full_data_sha256'] == self.native_plans['profile']['config']['full_body_v3']['full_data_sha256'] for plan in self.native_plans.values()), 'native_cohort_data_bindings_differ')
        return queue

    def streams(self, label):
        self.phase = 'native_actual_attempted_window_and_callback_audit'
        plan = self.native_plans[label]
        directory = Path(plan['run_dir'])
        launch = self.json(str(directory / 'launch.json'))
        require(launch['status'] == 'complete' and launch['plan_identity_sha256'] == plan['plan_identity_sha256'], 'native_launch_not_complete')
        results = []
        for rank in range(8):
            status = self.json(str(directory / ('rank-' + str(rank) + '.json')))
            require(status['status'] == 'bounded_complete' and status['rank'] == rank and status['world_size'] == 8 and status['plan_identity_sha256'] == plan['plan_identity_sha256'] and status['successful_optimizer_steps'] == (200 if label == 'profile' else 4), 'native_rank_not_at_registered_successful_budget')
            windows, backwards, tokens = [], 0, []
            attempts = sorted((directory / 'attempts').glob('attempt-*'), key=lambda path: int(path.name.split('-')[1]))
            require(len(attempts) == (2 if label == 'split' else 1), 'native_attempt_coverage_mismatch')
            for attempt in attempts:
                observed = self.json(str(attempt / ('full-body-rank-' + str(rank) + '.json')))
                require(observed['status'] == 'completed_corrected_native_execution_requires_independent_evidence' and observed['source_sha256'] == plan['config']['full_body_v3']['source_sha256'] and observed['plan_identity_sha256'] == plan['plan_identity_sha256'], 'native_observer_completion_binding_mismatch')
                rotary = observed['rotary_cache']
                require(rotary['previous_capacity'] == 2048 and rotary['primed_capacity'] == 2054 and rotary['construction_device'] == 'cpu' and rotary['construction_dtype'] == 'torch.float32' and rotary['existing_prefix_unchanged'] is True and rotary['frozen_after_priming'] is True, 'native_immutable_rotary_recipe_mismatch')
                require(observed['policy']['mode'] == plan['config']['full_body_v3']['checkpoint_mode'] and observed['policy']['adapter_sha256'] == SOURCES['native_gradients.py'], 'native_runtime_checkpoint_policy_mismatch')
                require(observed['actual_gpu_backend']['gradient_accumulation_steps'] == 2 and observed['actual_gpu_backend']['matmul_tf32'] is True and observed['actual_gpu_backend']['capability'] == [8, 0], 'native_actual_backend_geometry_mismatch')
                require(0 < observed['cuda_max_allocated_bytes'] <= observed['cuda_max_reserved_bytes'] <= observed['actual_gpu_backend']['device_memory_bytes'], 'native_memory_observation_invalid')
                rows = list(v1.read_lines(self.artifact(str(attempt / ('full-body-rank-' + str(rank) + '.jsonl')))))
                initial = attempt / ('initial-masters-rank-' + str(rank) + '.pt')
                self.artifact({'path': str(initial), 'sha256': observed['initial_master_snapshot_sha256']})
                legacy_rows = list(v1.read_lines(self.artifact(str(attempt / ('steps-rank-' + str(rank) + '.jsonl')))))
                require([row['global_step'] for row in legacy_rows] == [row['global_step'] for row in rows], 'native_scalar_journal_window_coverage_mismatch')
                for row, legacy in zip(rows, legacy_rows):
                    require(legacy['skipped_steps'] == row['skipped_steps'] and legacy['optimizer_step'] == row['successful_updates'] and legacy['update_skipped'] == row['update_skipped'] and math.isfinite(legacy['loss']), 'native_scalar_or_counter_journals_inconsistent')
                    row['native_last_microbatch_loss'] = legacy['loss']
                    row['native_global_grad_norm'] = legacy['global_grad_norm']
                    row['_attempt_directory'] = str(attempt)
                for row in rows:
                    require(len(row['actual_microbatches']) == len(row['parameter_gradient_coverage_per_backward']) == 2, 'native_not_two_microbatches_per_optimizer_window')
                    for batch in row['parameter_gradient_coverage_per_backward']:
                        require(set(batch) == NAMES and all(item['callbacks'] > 0 for item in batch.values()), 'native_backward_missing_named_parameter')
                    if not row['update_skipped']:
                        require(all(all(batch[name]['finite'] is True for batch in row['parameter_gradient_coverage_per_backward']) and any(batch[name]['nonzero'] is True for batch in row['parameter_gradient_coverage_per_backward']) for name in NAMES), 'native_applied_update_has_nonfinite_or_zero_body')
                    backwards += 2
                    tokens.extend(row['actual_microbatches'])
                windows.extend(rows)
            require([row['global_step'] for row in windows] == list(range(1, status['native_global_steps'] + 1)), 'native_attempted_global_window_sequence_incomplete')
            require(sum(row['update_skipped'] for row in windows) == status['skipped_steps'] and status['native_global_steps'] - status['skipped_steps'] == status['successful_optimizer_steps'], 'native_skipped_successful_budget_inconsistent')
            previous_skips = 0
            for row in windows:
                require(row['skipped_steps'] == previous_skips + int(row['update_skipped']) and row['successful_updates'] == row['global_step'] - row['skipped_steps'], 'native_per_window_scaler_skip_counter_inconsistent')
                previous_skips = row['skipped_steps']
            require(len(windows) <= plan['config']['full_body_v3']['maximum_attempted_optimizer_steps'], 'native_engineering_attempt_ceiling_exceeded')
            results.append({'rank': rank, 'windows': windows, 'status': status, 'backwards': backwards, 'microbatch_hash_sequence': tokens, 'attempt_directories': list(map(str, attempts))})
        return results

    def raw_checkpoint(self, plan, checkpoint, torch):
        self.phase = 'native_raw_dynamic_state_reconstruction'
        self.current_checkpoint = str(checkpoint)
        manifest = self.json(str(checkpoint / 'manifest.json'))
        require(manifest['plan_identity_sha256'] == plan['plan_identity_sha256'] and manifest['world_size'] == 8, 'native_raw_checkpoint_plan_world_mismatch')
        for name, expected in manifest['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe_native_raw_inventory')
            self.artifact({'path': str(checkpoint / name), 'sha256': expected})
        model_names = [name for name in manifest['files'] if name.endswith('model_states.pt')]
        require(len(model_names) == 1, 'native_raw_model_coverage_mismatch')
        model = torch.load(checkpoint / model_names[0], map_location='cpu', weights_only=False)
        config_path = Path(plan['workdir']) / 'ds_config.json'
        require(config_path.resolve() == config_path and plan['frozen_files']['work/ds_config.json'] == '02b432995b51d8fa1cce7685b7430de7ce14de0419a6729cd9ff0c324c1bec0b', 'native_config_path_or_frozen_source_changed')
        require(type(model['ds_config']) is str and model['ds_config'] == str(config_path), 'native_saved_config_filename_not_exact_registered_path')
        expected_ds = self.json({'path': str(config_path), 'sha256': plan['frozen_files']['work/ds_config.json']})
        require(expected_ds['optimizer']['type'] == 'AdamW' and expected_ds['optimizer']['params']['adam_w_mode'] is True and expected_ds['gradient_accumulation_steps'] == 2 and expected_ds['fp16']['loss_scale'] == 0, 'native_frozen_recipe_contract_changed')
        shapes = model['param_shapes']
        require({name for group in shapes for name in group} == NAMES and sum(len(group) for group in shapes) == 13, 'native_raw_trainable_inventory_mismatch')
        model_hash = v2.semantic_identity({name: model['module'][name] for name in NAMES}, torch)
        scheduler_hash = v2.semantic_identity(model['lr_scheduler'], torch)
        expected_steps = model['global_steps'] - model['skipped_steps']
        require(model['lr_scheduler'] == {'last_batch_iteration': expected_steps - 1}, 'native_log_scheduler_cursor_mismatch')
        optimizer_files = {}
        for name in manifest['files']:
            match = re.search(r'zero_pp_rank_(\d+)_mp_rank_00_optim_states\.pt$', name)
            if match:
                require(int(match.group(1)) not in optimizer_files, 'native_duplicate_optimizer_rank')
                optimizer_files[int(match.group(1))] = name
        require(set(optimizer_files) == set(range(8)), 'native_optimizer_rank_coverage_mismatch')
        masters, states, moments = [], [], defaultdict(lambda: {'elements': 0, 'first_nonzero': 0, 'second_nonzero': 0, 'master_changed_elements': 0, 'master_delta_squared': 0.0})
        for rank in range(8):
            rank_state = torch.load(checkpoint / ('rank-' + str(rank) + '.pt'), map_location='cpu', weights_only=False)
            raw = torch.load(checkpoint / optimizer_files[rank], map_location='cpu', weights_only=False)['optimizer_state_dict']
            actual = {'trainable_model': model_hash, 'optimizer_and_scaler': v2.semantic_identity(raw, torch), 'scheduler': scheduler_hash, 'global_steps': model['global_steps'], 'skipped_steps': model['skipped_steps'], 'micro_steps': rank_state['state_identity']['micro_steps'], 'learning_rates': [float(group['lr']) for group in raw['base_optimizer_state']['param_groups']]}
            expected = rank_state['state_identity']
            require(all(actual[key] == expected[key] for key in actual if key != 'learning_rates') and all(abs(x - y) <= max(1e-15, abs(y) * 1e-12) for x, y in zip(actual['learning_rates'], expected['learning_rates'])), 'native_raw_model_optimizer_scaler_scheduler_counter_identity_mismatch')
            require(raw['dynamic_loss_scale'] is True and int(raw['zero_stage']) == 2 and raw['partition_count'] == [8] * len(shapes), 'native_dynamic_scaler_or_zero_geometry_mismatch')
            rate = native_learning_rate(expected_steps - 1)
            require(all(abs(value - rate) <= max(1e-15, abs(rate) * 1e-12) for value in actual['learning_rates']), 'native_actual_log_learning_rate_mismatch')
            parts = raw['single_partition_of_fp32_groups']
            require(len(parts) == len(shapes), 'native_master_group_count_mismatch')
            masters.append(parts)
            baseline_path = Path(plan['run_dir']) / 'attempts/attempt-1' / ('initial-masters-rank-' + str(rank) + '.pt')
            initial = torch.load(self.artifact(str(baseline_path)), map_location='cpu', weights_only=False)
            require(v2.semantic_identity(initial['param_slice_mappings'], torch) == v2.semantic_identity(raw['param_slice_mappings'], torch), 'native_initial_master_named_partition_mapping_changed')
            require(len(initial['masters']) == len(parts), 'native_initial_master_group_count_mismatch')
            for group_index, group in enumerate(shapes):
                mapping = raw['param_slice_mappings'][group_index]
                params = raw['base_optimizer_state']['param_groups'][group_index]['params']
                settings = raw['base_optimizer_state']['param_groups'][group_index]
                require(tuple(settings['betas']) == (0.9, 0.95) and settings['eps'] == 1e-8 and settings['weight_decay'] == 0, 'native_saved_adam_hyperparameters_mismatch')
                require(len(params) == 1, 'native_optimizer_group_not_flat')
                state = raw['base_optimizer_state']['state'][params[0]]
                require(int(state['step']) == expected_steps and bool(torch.isfinite(state['exp_avg']).all()) and bool(torch.isfinite(state['exp_avg_sq']).all()) and bool((state['exp_avg_sq'] >= 0).all()), 'native_raw_adam_counter_or_moment_invalid')
                master, base = parts[group_index], initial['masters'][group_index]
                require(master.dtype == base.dtype == torch.float32 and master.shape == base.shape == state['exp_avg'].shape == state['exp_avg_sq'].shape, 'native_raw_master_moment_geometry_mismatch')
                require(sum(math.prod(shape) for shape in group.values()) == 8 * master.numel() and raw['group_paddings'][group_index] == 0, 'native_raw_named_partition_layout_not_registered')
                wanted, offset = {}, 0
                for name, shape in group.items():
                    count = math.prod(shape)
                    left, right = max(offset, rank * master.numel()), min(offset + count, (rank + 1) * master.numel())
                    if left < right:
                        wanted[name] = (left - rank * master.numel(), right - left)
                    offset += count
                require(set(mapping) == set(wanted), 'native_raw_named_fragment_parameter_coverage_mismatch')
                for name, address in mapping.items():
                    require(name in group and type(address.start) is int and type(address.numel) is int and 0 <= address.start and address.start + address.numel <= state['exp_avg'].numel(), 'native_named_fragment_invalid')
                    require((address.start, address.numel) == wanted[name], 'native_raw_named_fragment_offset_mismatch')
                    start, stop = address.start, address.start + address.numel
                    moments[name]['elements'] += address.numel
                    moments[name]['first_nonzero'] += int(torch.count_nonzero(state['exp_avg'][start:stop]))
                    moments[name]['second_nonzero'] += int(torch.count_nonzero(state['exp_avg_sq'][start:stop]))
                    delta = master[start:stop].double() - base[start:stop].double()
                    require(bool(torch.isfinite(delta).all()), 'native_master_change_nonfinite')
                    moments[name]['master_changed_elements'] += int(torch.count_nonzero(delta))
                    moments[name]['master_delta_squared'] += float(delta.square().sum())
                    del delta
            states.append({'rank': rank, **actual, 'rng': rng_identity(rank_state['rng'], torch), 'cursor': rank_state['cursor'], 'sampler': rank_state['sampler'], 'epoch_metrics_identity': v2.semantic_identity(rank_state['epoch_metrics'], torch), 'token_counts': rank_state['token_counts']})
            del raw, rank_state, initial
            gc.collect()
        for group_index, group in enumerate(shapes):
            flat = torch.cat([parts[group_index] for parts in masters])
            require(flat.numel() == sum(math.prod(shape) for shape in group.values()), 'native_master_partition_padding_not_qualified')
            offset = 0
            for name, shape in group.items():
                size = math.prod(shape)
                require(torch.equal(flat[offset:offset + size].reshape(shape).to(model['module'][name].dtype), model['module'][name]), 'native_reassembled_masters_disagree_with_model')
                require(moments[name]['elements'] == size and moments[name]['first_nonzero'] > 0 and moments[name]['second_nonzero'] > 0, 'native_named_parameter_missing_raw_adam_learning')
                if expected_steps == 200:
                    require(moments[name]['master_changed_elements'] > 0 and moments[name]['master_delta_squared'] > 0, 'native_profile_has_unchanged_named_fp32_master')
                offset += size
            del flat
        for value in moments.values():
            value['fp32_master_change_l2_from_attempt1_initialization'] = math.sqrt(value.pop('master_delta_squared'))
        result = {'path': str(checkpoint), 'manifest_sha256': digest(checkpoint / 'manifest.json'), 'ranks': states, 'moments': dict(moments), 'next_cursor': manifest['cursor'], 'global_steps': model['global_steps'], 'successful_steps': expected_steps, 'skipped_steps': model['skipped_steps'], 'all13_named_raw_moments_nonzero': True, 'all13_profile_masters_changed': expected_steps == 200, 'raw_master_reassembly_matches_model': True, 'saved_ds_config_representation': 'exact_original_filename', 'saved_ds_config_path': model['ds_config'], 'frozen_ds_config_sha256': plan['frozen_files']['work/ds_config.json']}
        del model, masters
        gc.collect()
        return result

    def run(self, preparation, torch):
        runtime_sources = self.runtime_sources()
        queue = self.plans_and_data(preparation)
        streams = {label: self.streams(label) for label in self.native_plans}
        self.phase = 'native_full_partition_direct_reference_and_restore_parity'
        for label in ('direct-reference', 'split'):
            for a, b in zip(streams['uninterrupted'], streams[label]):
                require(a['microbatch_hash_sequence'] == b['microbatch_hash_sequence'] and len(a['windows']) == len(b['windows']), 'native_actual_attempted_input_sequence_differs')
                for left, right in zip(a['windows'], b['windows']):
                    keys = ('global_step', 'skipped_steps', 'successful_updates', 'update_skipped', 'scaled_gradient_partition_sha256', 'learning_rates_before_step', 'loss_scale_before_step', 'loss_scale_after_step', 'state_identity', 'rng_identity', 'native_last_microbatch_loss', 'native_global_grad_norm')
                    require(all(left[key] == right[key] for key in keys), 'native_reference_or_restore_full_step_state_differs')
        self.phase = 'native_retained_initial_and_pre_step_tensor_replay'
        initial_hashes = {}
        gradient_files = 0
        for label, values in streams.items():
            for value in values:
                rank = value['rank']
                initial_path = Path(value['attempt_directories'][0]) / ('initial-masters-rank-' + str(rank) + '.pt')
                initial = torch.load(self.artifact(str(initial_path)), map_location='cpu', weights_only=False)
                actual = v2.semantic_identity(initial, torch)
                require(rank not in initial_hashes or initial_hashes[rank] == actual, 'native_scratch_initial_masters_differ_between_trajectories')
                initial_hashes[rank] = actual
                del initial
                if label == 'profile':
                    continue
                for row in value['windows']:
                    path = Path(row['_attempt_directory']) / ('gradients-rank-' + str(rank) + '-global-' + str(row['global_step'] - 1) + '.pt')
                    raw = torch.load(self.artifact(str(path)), map_location='cpu', weights_only=True)
                    require(v2.semantic_identity(raw, torch) == row['scaled_gradient_partition_sha256'], 'native_retained_gradient_partition_hash_mismatch')
                    if not row['update_skipped']:
                        require(all(bool(torch.isfinite(part).all()) for parts in raw.values() for part in parts if torch.is_tensor(part)), 'native_applied_raw_gradient_partition_nonfinite')
                    gradient_files += 1
                    del raw
                gc.collect()
        checkpoints, finals = [], {}
        for label, plan in self.native_plans.items():
            latest = self.json(str(Path(plan['trained_models']['resolved_path']) / 'recovery/latest.json'))
            require(latest['plan_identity_sha256'] == plan['plan_identity_sha256'], 'native_latest_checkpoint_plan_mismatch')
            path = Path(latest['path'])
            require(digest(path / 'manifest.json') == latest['manifest_sha256'], 'native_latest_checkpoint_changed')
            value = self.raw_checkpoint(plan, path, torch)
            require(value['successful_steps'] == (200 if label == 'profile' else 4), 'native_raw_successful_budget_mismatch')
            finals[label] = value
            checkpoints.append({'label': label, **value})
        require(finals['uninterrupted']['ranks'] == finals['direct-reference']['ranks'] == finals['split']['ranks'], 'native_raw_final_reference_or_restore_states_differ')
        prefixes = {}
        for label in ('uninterrupted', 'split'):
            plan = self.native_plans[label]
            recovery = Path(plan['trained_models']['resolved_path']) / 'recovery'
            candidates = []
            for path in recovery.glob('attempt-*-step-*'):
                if (path / 'manifest.json').is_file():
                    item = self.json(str(path / 'manifest.json'))
                    candidates.append((item['global_step'], path))
            require(len(candidates) == 2, 'native_two_and_four_raw_checkpoint_coverage_mismatch')
            prefix = self.raw_checkpoint(plan, min(candidates)[1], torch)
            require(prefix['successful_steps'] == 2, 'native_raw_prefix_not_two_successful_steps')
            prefixes[label] = prefix
        require(prefixes['uninterrupted']['ranks'] == prefixes['split']['ranks'], 'native_raw_two_step_restart_prefix_differs')
        return {'status': 'partial_evidence', 'scope': 'independent_native_full_body_raw_state_and_window_replay_not_admission', 'checks': {'all13_backward_dynamic_scaler_windows': 'passed', 'full_native_pre_step_gradient_and_state_parity': 'passed', 'eight_rank_four_vs_two_plus_two_final_raw_state': 'passed', 'all13_named_raw_adam': 'passed', 'maximum_shape_2048_2054': 'passed', 'complete_full_corpus200_profile': 'passed', 'all13_profile_master_changes': 'passed', 'stock_export_compatibility': 'unavailable_in_this_version', 'runtime_envelope': 'unavailable_in_this_version'}, 'raw_checkpoints': checkpoints, 'raw_prefixes': prefixes, 'reviewed_runtime_sources': runtime_sources, 'initial_master_partition_identities_by_rank': initial_hashes, 'retained_gradient_partition_files_replayed': gradient_files, 'window_summary': {label: [{'rank': value['rank'], 'attempted_windows': len(value['windows']), 'actual_backwards': value['backwards'], 'successful_updates': value['status']['successful_optimizer_steps'], 'skipped_steps': value['status']['skipped_steps']} for value in values] for label, values in streams.items()}, 'limitations': ['Raw sampler/RNG/epoch metrics and all attempted windows are compared; source/data qualification is separate from stock export and full runtime closure.', 'Native full masks are compared against a genuine direct-function native reference. Full-network FP64 derivatives are not recomputed from raw input/target tensors in this analyzer.', 'Stock export and full runtime-envelope replay remain explicitly unavailable in this analyzer version.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'native_independent_audit_requires_cpu_slurm')
    queue = json.loads((args.preparation / 'queue-plan.json').read_bytes())
    root = Path(json.loads(Path(queue['plans']['uninterrupted']['path']).read_bytes())['project_root']).resolve()
    output = args.output.resolve()
    require(output.is_relative_to(root / 'outputs') and not output.exists(), 'choose_fresh_native_audit_output')
    import torch
    import deepspeed
    torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    audit, start = Audit(root), time.monotonic()
    try:
        report = audit.run(args.preparation.resolve(), torch)
        audit.finish()
    except Exception as error:
        report = {'status': 'failed', 'phase': audit.phase, 'error_code': str(error) if isinstance(error, v1.AuditFailure) else type(error).__name__, 'checkpoint': getattr(audit, 'current_checkpoint', None)}
    report.update(scientific_training_admitted=False, automatic_admission_emitted=False, artifact_inventory=audit.inventory, artifact_inventory_sha256=identity(audit.inventory), analyzer_sha256=digest(Path(__file__)), semantic_hasher_v2_sha256=V2_SHA256, predecessor_analyzer_sha256='21683ef5a41505cb73d9db59570a55752835309e65995fc283cfb4f7edefb709', verifier_change_scope='Native CLI saves the exact original config filename; require that path and frozen02bJSON bytes plus pinned DSCLI/config-reader/engine sources. Preserve all raw tensor, numerical, dynamic-scaler, gradient, data and budget checks.', slurm_job_id=os.environ['SLURM_JOB_ID'], wall_seconds=time.monotonic() - start, cpu_only=True, project_helpers_imported=False, runtime_packages={name: importlib.metadata.version(name) for name in ('torch', 'deepspeed', 'numpy', 'transformers', 'accelerate')})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': digest(output)}), flush=True)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
