import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re


def require(value, code):
    if not value:
        raise ValueError(code)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'named_adam_coverage_requires_cpu_slurm')
    require(not args.output.exists(), 'named_adam_output_must_be_fresh')
    import torch
    import deepspeed

    torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    checkpoint = args.checkpoint.resolve()
    manifest_path = checkpoint / 'manifest.json'
    require(digest(manifest_path) == args.manifest_sha256, 'checkpoint_manifest_changed')
    manifest = json.loads(manifest_path.read_bytes())
    inventory = {'manifest.json': args.manifest_sha256}
    for name, descriptor in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe_checkpoint_file')
        expected = descriptor if isinstance(descriptor, str) else descriptor['sha256']
        require(digest(checkpoint / name) == expected, 'raw_checkpoint_file_changed')
        inventory[name] = expected
    files = [name for name in manifest['files'] if name.endswith('model_states.pt')]
    require(len(files) == 1 and manifest['world_size'] == 8, 'expected_single_model_eight_rank_zero2')
    model = torch.load(checkpoint / files[0], map_location='cpu', weights_only=False)
    shapes = model['param_shapes']
    expected = {name: math.prod(shape) for group in shapes for name, shape in group.items()}
    require(len(expected) == 13 and {'fc.weight', 'norm.weight', 'lm_head.weight'} <= set(expected), 'unexpected_native_trainable_parameter_set')
    fragments = defaultdict(list)
    optimizer_files = {}
    for name in manifest['files']:
        match = re.search(r'zero_pp_rank_(\d+)_mp_rank_00_optim_states\.pt$', name)
        if match:
            optimizer_files[int(match.group(1))] = name
    require(set(optimizer_files) == set(range(8)), 'optimizer_rank_coverage_incomplete')
    for rank in range(8):
        raw = torch.load(checkpoint / optimizer_files[rank], map_location='cpu', weights_only=False)['optimizer_state_dict']
        require(int(raw['zero_stage']) == 2 and len(raw['param_slice_mappings']) == len(shapes), 'zero2_named_fragment_schema_mismatch')
        base = raw['base_optimizer_state']
        for group, mapping in enumerate(raw['param_slice_mappings']):
            parameters = base['param_groups'][group]['params']
            require(len(parameters) == 1, 'expected_one_flat_master_per_group')
            state = base['state'][parameters[0]]
            first, second = state['exp_avg'], state['exp_avg_sq']
            require(first.shape == second.shape and first.ndim == 1, 'adam_flat_moment_geometry_mismatch')
            for name, address in mapping.items():
                require(name in expected and set(vars(address)) == {'start', 'numel'}, 'unknown_named_fragment')
                start, count = address.start, address.numel
                require(type(start) is int and type(count) is int and 0 <= start < first.numel() and 0 < count <= first.numel() - start, 'named_fragment_range_invalid')
                stop = start + count
                fragments[name].append({'rank': rank, 'group': group, 'local_start': start, 'elements': count, 'first_moment_nonzero': int(torch.count_nonzero(first[start:stop])), 'second_moment_nonzero': int(torch.count_nonzero(second[start:stop])), 'adam_step': int(state['step'])})
        del raw
    coverage = {}
    for name, size in expected.items():
        items = fragments[name]
        require(sum(item['elements'] for item in items) == size, 'named_parameter_fragments_do_not_cover_full_shape')
        coverage[name] = {'elements': size, 'first_moment_nonzero': sum(item['first_moment_nonzero'] for item in items), 'second_moment_nonzero': sum(item['second_moment_nonzero'] for item in items), 'fragments': items}
    upstream_zero = all(value['first_moment_nonzero'] == value['second_moment_nonzero'] == 0 for name, value in coverage.items() if name.startswith(('fc.', 'midlayer.')))
    downstream_live = all(value['first_moment_nonzero'] > 0 and value['second_moment_nonzero'] > 0 for name, value in coverage.items() if name.startswith(('norm.', 'lm_head.')))
    require(all(digest(checkpoint / name) == expected for name, expected in inventory.items()), 'checkpoint_changed_during_named_coverage')
    report = {'status': 'raw_named_adam_coverage_complete', 'scope': 'actual_saved_optimizer_moment_coverage_per_named_trainable_not_a_fresh_gradient_execution', 'checkpoint': str(checkpoint), 'manifest_sha256': args.manifest_sha256, 'global_steps': model['global_steps'], 'trainable_parameters': coverage, 'fc_midlayer_all_moments_zero': upstream_zero, 'outer_norm_and_head_have_nonzero_moments': downstream_live, 'head_only_pattern_detected': upstream_zero and downstream_live, 'all_files_stable_before_after': True, 'artifact_inventory': inventory, 'probe_source_sha256': digest(Path(__file__)), 'torch_version': torch.__version__, 'deepspeed_version': deepspeed.__version__, 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'scientific_training_admitted': False, 'jobs_or_active_sources_modified': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'head_only_pattern_detected': report['head_only_pattern_detected'], 'report_sha256': digest(args.output)}), flush=True)


if __name__ == '__main__':
    main()
