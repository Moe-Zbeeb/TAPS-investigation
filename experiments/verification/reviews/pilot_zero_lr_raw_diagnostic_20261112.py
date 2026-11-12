import argparse
from collections import defaultdict
import gc
import hashlib
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


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


if digest(V2_PATH) != V2_SHA256:
    raise ValueError('frozen_hasher_changed')
spec = importlib.util.spec_from_file_location('zero_lr_frozen_v2', V2_PATH)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
require = v2.require


def differences(left, right, torch):
    require(left.dtype == right.dtype and left.shape == right.shape and left.is_floating_point(), 'tensor_difference_geometry_mismatch')
    left, right = left.contiguous(), right.contiguous()
    bits = {torch.float16: torch.int16, torch.float32: torch.int32}[left.dtype]
    different = left.view(bits) != right.view(bits)
    zeros = (left == 0) & (right == 0)
    return {'elements': left.numel(), 'numeric_changed': int(torch.count_nonzero(left != right)), 'bitwise_changed': int(torch.count_nonzero(different)), 'signed_zero_only_bit_changes': int(torch.count_nonzero(different & zeros)), 'all_finite': bool(torch.isfinite(left).all() and torch.isfinite(right).all())}


class Audit(v2.Audit):
    def prefix(self, plan, torch):
        path = (self.root / 'models').resolve() / 'trained/pilot-qualifications' / ('pilot-qualifications/' + plan['study_id'] + '/' + plan['policy'] + '/split/checkpoints/step-2')
        manifest = self.json(str(path / 'manifest.json'))
        require(manifest['next_step'] == 2 and manifest['plan_identity_sha256'] == plan['identity_sha256'] and manifest['world_size'] == 8, 'prefix_manifest_scope_mismatch')
        for name, expected in manifest['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe_prefix_inventory')
            self.artifact({'path': str(path / name), 'sha256': expected})
        model_file = next(name for name in manifest['files'] if name.endswith('model_states.pt'))
        saved = torch.load(path / model_file, map_location='cpu', weights_only=False)
        ancestor_path = Path(plan['ancestor']['path']) / 'pytorch_model.bin'
        self.artifact({'path': str(ancestor_path), **plan['ancestor']['files']['pytorch_model.bin']})
        ancestor = torch.load(ancestor_path, map_location='cpu', weights_only=True)
        shapes = saved['param_shapes']
        names = {name for group in shapes for name in group}
        require(len(names) == 13, 'prefix_not_thirteen_trainables')
        visible = {}
        for name in sorted(names):
            actual = saved['module'][name]
            visible[name] = {'dtype': str(actual.dtype), 'shape': list(actual.shape), **differences(actual, ancestor[name].to(actual.dtype), torch)}
        vocabulary = {}
        for name in ('d2t', 't2d'):
            a, b = saved['module'][name], ancestor[name]
            vocabulary[name] = {'dtype': str(a.dtype), 'shape': list(a.shape), 'dtype_equal': a.dtype == b.dtype, 'shape_equal': a.shape == b.shape, 'tensor_equal': torch.equal(a, b), 'bitwise_identity_equal': v2.semantic_identity(a, torch) == v2.semantic_identity(b, torch)}
        optimizer_files = {}
        for name in manifest['files']:
            match = re.search(r'zero_pp_rank_(\d+)_mp_rank_00_optim_states\.pt$', name)
            if match:
                require(int(match.group(1)) not in optimizer_files, 'duplicate_prefix_optimizer_rank')
                optimizer_files[int(match.group(1))] = name
        require(set(optimizer_files) == set(range(8)), 'prefix_optimizer_rank_coverage_mismatch')
        master = defaultdict(lambda: {'elements': 0, 'numeric_changed': 0, 'bitwise_changed': 0, 'signed_zero_only_bit_changes': 0, 'all_finite': True, 'first_moment_nonzero': 0, 'second_moment_nonzero': 0, 'fp32_master_delta_squared': 0.0, 'master_cast_visible_bitwise_changes': 0, 'fragments': []})
        for rank in range(8):
            raw = torch.load(path / optimizer_files[rank], map_location='cpu', weights_only=False)['optimizer_state_dict']
            for group_index, group in enumerate(shapes):
                partition = raw['single_partition_of_fp32_groups'][group_index]
                require(partition.dtype == torch.float32 and partition.ndim == 1 and 8 * partition.numel() == sum(math.prod(shape) for shape in group.values()) and raw['group_paddings'][group_index] == 0, 'prefix_master_partition_geometry_mismatch')
                params = raw['base_optimizer_state']['param_groups'][group_index]['params']
                require(len(params) == 1, 'prefix_flat_optimizer_group_mismatch')
                state = raw['base_optimizer_state']['state'][params[0]]
                mapping, expected, offset = raw['param_slice_mappings'][group_index], {}, 0
                for name, shape in group.items():
                    count = math.prod(shape)
                    left, right = max(offset, rank * partition.numel()), min(offset + count, (rank + 1) * partition.numel())
                    if left < right:
                        expected[name] = (left - rank * partition.numel(), right - left, left - offset)
                    offset += count
                require(set(mapping) == set(expected), 'prefix_fragment_name_mismatch')
                for name, (start, count, parameter_start) in expected.items():
                    address = mapping[name]
                    require((address.start, address.numel) == (start, count), 'prefix_fragment_offsets_mismatch')
                    dtype = saved['module'][name].dtype
                    initial = ancestor[name].to(dtype).reshape(-1)[parameter_start:parameter_start + count].float()
                    actual = partition[start:start + count]
                    entry = differences(actual, initial, torch)
                    delta = actual.double() - initial.double()
                    entry.update(rank=rank, group=group_index, local_start=start, parameter_start=parameter_start, first_moment_nonzero=int(torch.count_nonzero(state['exp_avg'][start:start + count])), second_moment_nonzero=int(torch.count_nonzero(state['exp_avg_sq'][start:start + count])), fp32_master_delta_squared=float(delta.square().sum()), adam_step=int(state['step']))
                    stored = saved['module'][name].reshape(-1)[parameter_start:parameter_start + count]
                    entry['master_cast_visible_bitwise_changes'] = differences(actual.to(dtype), stored, torch)['bitwise_changed']
                    value = master[name]
                    for key in ('elements', 'numeric_changed', 'bitwise_changed', 'signed_zero_only_bit_changes', 'first_moment_nonzero', 'second_moment_nonzero', 'fp32_master_delta_squared', 'master_cast_visible_bitwise_changes'):
                        value[key] += entry[key]
                    value['all_finite'] = value['all_finite'] and entry['all_finite']
                    value['fragments'].append(entry)
                    del initial, actual, delta
            del raw
            gc.collect()
        for name, value in master.items():
            require(value['elements'] == saved['module'][name].numel(), 'prefix_named_element_coverage_mismatch')
            value['fp32_master_numeric_delta_l2'] = math.sqrt(value.pop('fp32_master_delta_squared'))
        baseline = {name: ancestor[name].to(saved['module'][name].dtype) for name in names}
        baseline.update({name: ancestor[name] for name in ('d2t', 't2d')})
        actual_model = {name: saved['module'][name] for name in baseline}
        result = {'policy': plan['policy'], 'checkpoint': str(path), 'manifest_sha256': digest(path / 'manifest.json'), 'registered_plan_identity_sha256': plan['identity_sha256'], 'raw_global_steps': saved['global_steps'], 'raw_skipped_steps': saved['skipped_steps'], 'raw_scheduler': saved['lr_scheduler'], 'initial_dtype_cast_model_sha256': v2.semantic_identity(baseline, torch), 'raw_prefix_model_sha256': v2.semantic_identity(actual_model, torch), 'visible_trainable_differences': visible, 'fp32_master_and_adam_differences': dict(sorted(master.items())), 'vocabulary': vocabulary}
        result['all13_numeric_masters_unchanged'] = all(value['numeric_changed'] == 0 for value in master.values())
        result['all_master_bit_changes_are_signed_zero'] = all(value['bitwise_changed'] == value['signed_zero_only_bit_changes'] for value in master.values())
        result['all_visible_bit_changes_are_signed_zero'] = all(value['bitwise_changed'] == value['signed_zero_only_bit_changes'] for value in visible.values())
        result['all13_named_adam_first_second_nonzero'] = all(value['first_moment_nonzero'] > 0 and value['second_moment_nonzero'] > 0 for value in master.values())
        result['all_master_casts_bitwise_match_visible'] = all(value['master_cast_visible_bitwise_changes'] == 0 for value in master.values())
        del saved, ancestor, baseline, actual_model
        gc.collect()
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'zero_lr_raw_diagnostic_requires_cpu_slurm')
    study = args.study / 'manifest.json' if args.study.is_dir() else args.study
    manifest = json.loads(study.read_bytes())
    root = Path(manifest['common']['project_root']).resolve()
    require(args.output.resolve().is_relative_to(root / 'outputs') and not args.output.exists(), 'choose_fresh_zero_lr_raw_report')
    import torch
    import deepspeed
    torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    audit, start = Audit(root), time.monotonic()
    try:
        v2.v1.verified_identity(audit.json(str(study)))
        results = []
        for descriptor in manifest['plans']:
            plan = v2.v1.verified_identity(audit.json(descriptor))
            if plan['seed'] == 0 and plan['cohort'] == 'generalist':
                results.append(audit.prefix(plan, torch))
        require(len(results) == 2, 'raw_prefix_diagnostic_policy_coverage_mismatch')
        audit.finish()
        report = {'status': 'raw_zero_lr_prefix_mechanism_diagnostic_complete_not_admission', 'prefixes': results}
    except Exception as error:
        report = {'status': 'failed', 'error_code': str(error) if isinstance(error, v2.v1.AuditFailure) else type(error).__name__}
    report.update(scientific_training_admitted=False, automatic_admission_emitted=False, artifact_inventory=audit.inventory, artifact_inventory_sha256=v2.v1.identity(audit.inventory), diagnostic_source_sha256=digest(Path(__file__)), semantic_hasher_v2_sha256=V2_SHA256, slurm_job_id=os.environ['SLURM_JOB_ID'], wall_seconds=time.monotonic() - start, cpu_only=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': digest(args.output)}), flush=True)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
