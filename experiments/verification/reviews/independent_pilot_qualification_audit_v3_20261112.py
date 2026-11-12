import argparse
from collections import defaultdict
import gc
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
CHECKPOINT_SHA256 = '7a05799e678085df1b81d512bed679c35053fb6225ea90c96661306f93140f6c'
V2_PATH = Path(__file__).with_name('independent_pilot_qualification_audit_v2_20261112.py')


def digest(path):
    import hashlib
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


if digest(V2_PATH) != V2_SHA256:
    raise ValueError('frozen_v2_analyzer_changed')
spec = importlib.util.spec_from_file_location('frozen_independent_pilot_v2', V2_PATH)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
v1 = v2.v1
require = v1.require

NAMES = {
    'fc.weight', 'norm.weight', 'lm_head.weight',
    'midlayer.hidden_norm.weight', 'midlayer.input_layernorm.weight',
    'midlayer.post_attention_layernorm.weight',
    'midlayer.self_attn.q_proj.weight', 'midlayer.self_attn.k_proj.weight',
    'midlayer.self_attn.v_proj.weight', 'midlayer.self_attn.o_proj.weight',
    'midlayer.mlp.gate_proj.weight', 'midlayer.mlp.up_proj.weight',
    'midlayer.mlp.down_proj.weight',
}


def validate_callbacks(value, sizes):
    require(set(value) == NAMES == set(sizes), 'named_hook_parameter_inventory_mismatch')
    for name, item in value.items():
        require(type(item['callbacks']) is int and item['callbacks'] > 0, 'named_hook_missing_callback')
        require(item['finite'] is True and item['nonzero'] is True, 'named_hook_nonfinite_or_zero')
        count = item['maximum_callback_nonzero_elements']
        require(type(count) is int and 0 < count <= sizes[name], 'named_hook_nonzero_count_invalid')
        require(type(item['sum_finite_callback_l2']) in (int, float) and math.isfinite(item['sum_finite_callback_l2']) and item['sum_finite_callback_l2'] > 0, 'named_hook_gradient_norm_invalid')


class Audit(v2.Audit):
    def checkpoint_policies(self, qualification):
        self.phase = 'corrected_checkpoint_policy_and_named_hooks'
        results = []
        for policy in ('original', 'target'):
            plans = [plan for plan in self.plans.values() if plan['policy'] == policy and plan['cohort'] == 'generalist' and plan['seed'] == 0]
            require(len(plans) == 1, 'corrected_policy_plan_coverage_mismatch')
            plan = plans[0]
            require(plan['recipe']['checkpoint_mode'] == 'non_reentrant' and 'native_gradients.py' in plan['source_sha256'], 'corrected_checkpoint_policy_not_registered')
            sizes = None
            for variant, attempts in (('uninterrupted', ('full',)), ('native-reference', ('reference',)), ('split', ('prefix', 'restore')), ('alternate-pad', ('alternate',))):
                for attempt in attempts:
                    directory = qualification / policy / variant / 'attempts' / attempt
                    for rank in range(8):
                        initial_path = directory / ('rank-' + str(rank) + '.initial.json')
                        complete_path = directory / ('rank-' + str(rank) + '.complete.json')
                        journal_path = directory / ('rank-' + str(rank) + '.jsonl')
                        if not all(path.exists() for path in (initial_path, complete_path, journal_path)):
                            continue
                        initial, complete = self.json(str(initial_path)), self.json(str(complete_path))
                        require(initial['plan_identity_sha256'] == complete['plan_identity_sha256'] == plan['identity_sha256'], 'corrected_policy_plan_binding_mismatch')
                        require(set(initial['trainable_names']) == NAMES and len(initial['trainable_names']) == 13, 'intended_trainable_names_changed')
                        if sizes is None:
                            ancestor = Path(plan['ancestor']['path']) / 'pytorch_model.bin'
                            self.artifact({'path': str(ancestor), **plan['ancestor']['files']['pytorch_model.bin']})
                            import torch
                            released = torch.load(ancestor, map_location='cpu', weights_only=True)
                            sizes = {name: released[name].numel() for name in NAMES}
                            del released
                            gc.collect()
                        reference = variant == 'native-reference'
                        expected_mode = 'off_reference' if reference else 'non_reentrant'
                        require(initial['reference'] is reference and complete['reference'] is reference and initial['qualification'] is True and complete['qualification'] is True, 'direct_reference_or_qualification_flag_mismatch')
                        require(initial['checkpoint_policy'] == complete['checkpoint_policy'], 'checkpoint_policy_changed_during_attempt')
                        observed = initial['checkpoint_policy']
                        require(observed['mode'] == expected_mode and observed['nested_tensor_outputs_supported'] is True, 'checkpoint_policy_did_not_use_registered_branch')
                        require(observed['adapter_sha256'] == plan['source_sha256']['native_gradients.py'] and Path(observed['adapter_path']) == Path(plan['source_directory']) / 'native_gradients.py', 'checkpoint_adapter_source_binding_mismatch')
                        self.artifact({'path': observed['adapter_path'], 'sha256': observed['adapter_sha256']})
                        require(observed['torch_checkpoint_sha256'] == CHECKPOINT_SHA256, 'torch_checkpoint_implementation_not_reviewed')
                        self.artifact({'path': observed['torch_checkpoint_path'], 'sha256': CHECKPOINT_SHA256})
                        require(complete['all_thirteen_trainables_received_finite_nonzero_gradients_each_update'] is True, 'named_parameter_participation_completion_missing')
                        require(type(complete['cuda_max_allocated_bytes']) is int and type(complete['cuda_max_reserved_bytes']) is int and 0 < complete['cuda_max_allocated_bytes'] <= complete['cuda_max_reserved_bytes'], 'corrected_path_memory_observation_invalid')
                        rows = [row for row in v1.read_lines(self.artifact(str(journal_path))) if row['event'] in ('committed_update', 'successful_update')]
                        expected = list(range(complete['start_step'], complete['stop_step']))
                        require([row['step'] for row in rows] == expected and expected == (list(range(2, 4)) if attempt == 'restore' else list(range(2)) if attempt == 'prefix' else list(range(4))), 'corrected_path_actual_update_coverage_mismatch')
                        for row in rows:
                            validate_callbacks(row['parameter_gradient_coverage'], sizes)
                        results.append({'policy': policy, 'variant': variant, 'attempt': attempt, 'rank': rank, 'mode': expected_mode, 'steps': expected, 'all13_named_hooks_finite_nonzero': True, 'cuda_max_allocated_bytes': complete['cuda_max_allocated_bytes'], 'cuda_max_reserved_bytes': complete['cuda_max_reserved_bytes']})
        return {'status': 'passed' if len(results) == 80 else 'partial_evidence' if results else 'unavailable', 'attempt_ranks': results, 'scope': 'Actual per-parameter pre-reduction callback observations in every qualification update, source-bound to the explicit nonreentrant adapter or direct-function reference. These observations complement independently reconstructed raw moments and masters; they do not independently recompute the entire network derivative.'}

    def raw_checkpoint(self, checkpoint, torch, plan=None, expected_manifest=None):
        result = super().raw_checkpoint(checkpoint, torch, plan, expected_manifest)
        if plan is None:
            return result
        self.phase = 'all13_raw_named_adam_and_master_update_reconstruction'
        manifest = self.json(str(checkpoint / 'manifest.json'))
        model_files = [name for name in manifest['files'] if name.endswith('model_states.pt')]
        require(len(model_files) == 1, 'named_raw_model_inventory_mismatch')
        model = torch.load(checkpoint / model_files[0], map_location='cpu', weights_only=False)
        shapes = model['param_shapes']
        expected_sizes = {name: math.prod(shape) for group in shapes for name, shape in group.items()}
        require(set(expected_sizes) == NAMES and sum(len(group) for group in shapes) == 13, 'raw_named_trainable_inventory_mismatch')
        ancestor_path = Path(plan['ancestor']['path']) / 'pytorch_model.bin'
        self.artifact({'path': str(ancestor_path), **plan['ancestor']['files']['pytorch_model.bin']})
        ancestor = torch.load(ancestor_path, map_location='cpu', weights_only=True)
        optimizer_files = {}
        for name in manifest['files']:
            match = re.search(r'zero_pp_rank_(\d+)_mp_rank_00_optim_states\.pt$', name)
            if match:
                require(int(match.group(1)) not in optimizer_files, 'named_raw_duplicate_rank')
                optimizer_files[int(match.group(1))] = name
        require(set(optimizer_files) == set(range(8)), 'named_raw_rank_coverage_mismatch')
        totals = {name: {'elements': 0, 'first_moment_nonzero': 0, 'second_moment_nonzero': 0, 'fp32_master_changed_elements': 0, 'fp32_master_delta_squared': 0.0, 'fragments': []} for name in NAMES}
        for rank in range(8):
            raw = torch.load(checkpoint / optimizer_files[rank], map_location='cpu', weights_only=False)['optimizer_state_dict']
            require(len(raw['param_slice_mappings']) == len(shapes) == len(raw['single_partition_of_fp32_groups']), 'named_raw_group_count_mismatch')
            for group_index, group in enumerate(shapes):
                parameters = raw['base_optimizer_state']['param_groups'][group_index]['params']
                require(len(parameters) == 1, 'named_raw_flat_group_schema_mismatch')
                state = raw['base_optimizer_state']['state'][parameters[0]]
                first, second = state['exp_avg'], state['exp_avg_sq']
                master = raw['single_partition_of_fp32_groups'][group_index]
                require(first.shape == second.shape == master.shape and first.ndim == 1 and master.dtype == first.dtype == second.dtype == torch.float32, 'named_raw_moment_master_geometry_mismatch')
                require(bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all()) and bool(torch.isfinite(master).all()) and bool((second >= 0).all()), 'named_raw_nonfinite_or_negative_second_moment')
                elements = sum(math.prod(shape) for shape in group.values())
                require(elements == 8 * master.numel() and raw['group_paddings'][group_index] == 0, 'named_raw_partition_or_padding_changed')
                partition_start, partition_end = rank * master.numel(), (rank + 1) * master.numel()
                mapping = raw['param_slice_mappings'][group_index]
                expected_mapping, offset = {}, 0
                for name, shape in group.items():
                    count = math.prod(shape)
                    left, right = max(offset, partition_start), min(offset + count, partition_end)
                    if left < right:
                        expected_mapping[name] = (left - partition_start, right - left, left - offset)
                    offset += count
                require(set(mapping) == set(expected_mapping), 'named_raw_fragment_name_coverage_mismatch')
                for name, (start, count, parameter_start) in expected_mapping.items():
                    address = mapping[name]
                    require(set(vars(address)) == {'start', 'numel'} and type(address.start) is int and type(address.numel) is int and (address.start, address.numel) == (start, count), 'named_raw_fragment_offsets_mismatch')
                    stop = start + count
                    baseline = ancestor[name].to(model['module'][name].dtype).reshape(-1)[parameter_start:parameter_start + count].float()
                    delta = master[start:stop].double() - baseline.double()
                    fragment = {'rank': rank, 'group': group_index, 'local_start': start, 'parameter_start': parameter_start, 'elements': count, 'first_moment_nonzero': int(torch.count_nonzero(first[start:stop])), 'second_moment_nonzero': int(torch.count_nonzero(second[start:stop])), 'fp32_master_changed_elements': int(torch.count_nonzero(delta)), 'fp32_master_delta_squared': float(delta.square().sum()), 'adam_step': int(state['step'])}
                    require(fragment['adam_step'] == result['next_step'], 'named_raw_adam_step_mismatch')
                    total = totals[name]
                    for key in ('elements', 'first_moment_nonzero', 'second_moment_nonzero', 'fp32_master_changed_elements', 'fp32_master_delta_squared'):
                        total[key] += fragment[key]
                    total['fragments'].append(fragment)
                    del delta, baseline
            del raw
            gc.collect()
        for name, total in totals.items():
            require(total['elements'] == expected_sizes[name], 'named_raw_full_parameter_element_coverage_mismatch')
            require(total['first_moment_nonzero'] > 0 and total['second_moment_nonzero'] > 0 and total['fp32_master_changed_elements'] > 0 and math.isfinite(total['fp32_master_delta_squared']) and total['fp32_master_delta_squared'] > 0, 'named_trainable_has_no_saved_adam_or_master_update')
            total['fp32_master_delta_l2_from_fp16_initialized_released_ancestor'] = math.sqrt(total.pop('fp32_master_delta_squared'))
        result['all13_named_adam_and_master_update'] = {'status': 'passed', 'parameters': dict(sorted(totals.items())), 'baseline': 'Released ancestor tensors cast to actual initialized FP16 module dtype, then converted to FP32. Checks source-bound initialized-master updates at steps2/4; does not require each individual FP16 module element to change.'}
        del model, ancestor
        gc.collect()
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--runtime-pre', type=Path)
    parser.add_argument('--runtime-post', type=Path)
    parser.add_argument('--execution-envelope', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'v3_qualification_audit_requires_cpu_slurm')
    study = args.study / 'manifest.json' if args.study.is_dir() else args.study
    root = Path(json.loads(study.read_bytes())['common']['project_root']).resolve()
    qualification = args.qualification.resolve()
    output = args.output.resolve() if args.output.suffix == '.json' else args.output.resolve() / 'report.json'
    require(output.is_relative_to(root / 'outputs') and not output.exists(), 'choose_fresh_v3_report_under_outputs')
    import torch
    import deepspeed
    torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    audit, start = Audit(root), time.monotonic()
    try:
        request = audit.automatic_request(args.study.resolve(), qualification)
        schedules = audit.schedules(request['study_manifest'])
        callbacks = audit.checkpoint_policies(qualification)
        gradients = audit.gradients(request['gradient_cases'], torch)
        traces = audit.traces(request['state_comparisons'])
        suffix = audit.suffix(qualification, torch)
        raw = audit.raw_qualifications(qualification, torch)
        stock = audit.stock(qualification, torch)
        pre = args.runtime_pre or qualification / 'runtime/pre.json'
        post = args.runtime_post or qualification / 'runtime/post.json'
        interval = args.execution_envelope or qualification / 'runtime/interval-accounting.json'
        runtime = audit.runtime_envelope(pre if pre.is_file() else None, post if post.is_file() else None, interval if interval.is_file() else None)
        checks = {'schedules': schedules['status'], 'ce_gradients': gradients['status'], 'outside_mask_gradients': gradients['outside_mask_status'], **traces['statuses'], 'suffix_probe': suffix['status'], 'raw_zero_state_reconstruction': raw['status'], 'all13_named_hooks_and_corrected_policy': callbacks['status'], 'all13_named_adam_and_fp32_updates': 'passed' if raw['status'] == 'passed' and all(value.get('all13_named_adam_and_master_update', {}).get('status') == 'passed' for value in raw['checkpoints']) else 'partial_evidence', 'stock_export_compatibility': stock['status'], 'runtime_envelope': runtime['status']}
        report = {'status': 'all_checked_evidence_passed_requires_independent_review' if all(value == 'passed' for value in checks.values()) else 'partial_evidence', 'scope': 'independent_corrected_full_body_cpu_pilot_qualification_analysis_not_admission', 'checks': checks, 'schedules': schedules, 'checkpoint_and_named_hook_coverage': callbacks, 'gradients': gradients, 'state_comparisons': traces, 'suffix': suffix, 'raw_states': raw, 'stock': stock, 'runtime': runtime}
        audit.finish()
    except Exception as error:
        report = {'status': 'failed', 'phase': audit.phase, 'error_code': str(error) if isinstance(error, v1.AuditFailure) else type(error).__name__}
    report.update(scientific_training_admitted=False, automatic_admission_emitted=False, artifact_inventory=audit.inventory, artifact_inventory_sha256=v1.identity(audit.inventory), analyzer_v1_sha256=v2.V1_SHA256, analyzer_v2_sha256=V2_SHA256, analyzer_v3_sha256=digest(Path(__file__)), slurm_job_id=os.environ['SLURM_JOB_ID'], cpu_only=True, wall_seconds=time.monotonic() - start, project_helpers_imported=False, runtime_packages={name: importlib.metadata.version(name) for name in ('torch', 'deepspeed', 'numpy', 'transformers', 'accelerate')})
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
