import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import time


def require(value, code):
    if not value:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def semantic_identity(value, torch):
    result = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            value = item.detach().cpu().contiguous()
            result.update(canonical({'dtype': str(value.dtype), 'shape': list(value.shape)}))
            result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
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
            require(set(vars(item)) == {'numel', 'start'}, 'unknown_fragment_address_schema')
            result.update(canonical([type(item).__module__, type(item).__qualname__]))
            visit(vars(item))
        elif item is None or type(item) in (str, int, float, bool):
            result.update(canonical([type(item).__name__, item]))
        else:
            raise ValueError('unsupported_semantic_state_type')
    visit(value)
    return result.hexdigest()


def schema(value, torch, depth=0):
    kind = type(value).__module__ + '.' + type(value).__qualname__
    if torch.is_tensor(value):
        return {'type': kind, 'dtype': str(value.dtype), 'shape': list(value.shape), 'semantic_sha256': semantic_identity(value, torch)}
    if isinstance(value, str):
        return {'type': kind, 'length': len(value), 'sha256': hashlib.sha256(value.encode()).hexdigest()}
    if value is None or type(value) in (int, float, bool):
        return {'type': kind, 'value': value if not isinstance(value, float) or math.isfinite(value) else 'nonfinite'}
    if depth >= 7:
        return {'type': kind, 'depth_limit': True, 'length': len(value) if hasattr(value, '__len__') else None}
    if isinstance(value, dict):
        entries = {}
        for key, child in value.items():
            label = str(key)
            require(len(label) < 256, 'unexpected_long_checkpoint_dictionary_key')
            entries[label] = schema(child, torch, depth + 1)
        return {'type': kind, 'length': len(value), 'entries': entries}
    if isinstance(value, (list, tuple)):
        return {'type': kind, 'length': len(value), 'items': [schema(child, torch, depth + 1) for child in value[:16]], 'truncated': len(value) > 16}
    if hasattr(value, '__dict__'):
        return {'type': kind, 'attributes': schema(vars(value), torch, depth + 1)}
    return {'type': kind}


def candidates(value, torch, path='', depth=0):
    if not isinstance(value, dict) or depth > 3:
        return
    try:
        yield {'path': path or '<root>', 'semantic_sha256': semantic_identity(value, torch)}
    except ValueError:
        pass
    for key, child in value.items():
        if isinstance(child, dict):
            yield from candidates(child, torch, (path + '.' if path else '') + str(key), depth + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--optimizer-files', type=int, default=2)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'probe_requires_cpu_slurm')
    require(re.fullmatch('[0-9a-f]{64}', args.manifest_sha256), 'manifest_sha256_required')
    require(not args.output.exists(), 'probe_output_must_be_fresh')
    import torch
    import deepspeed
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    start = time.monotonic()
    checkpoint = args.checkpoint.resolve()
    manifest_path = checkpoint / 'manifest.json'
    require(digest(manifest_path) == args.manifest_sha256, 'checkpoint_manifest_changed')
    manifest = json.loads(manifest_path.read_bytes())
    inventory = {'manifest.json': {'sha256': args.manifest_sha256, 'size_bytes': manifest_path.stat().st_size}}
    for name, item in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe_manifest_file')
        expected = item['sha256'] if isinstance(item, dict) else item
        path = checkpoint / name
        require(digest(path) == expected, 'checkpoint_file_changed')
        inventory[name] = {'sha256': expected, 'size_bytes': path.stat().st_size}
    rank_states = {}
    for rank in range(manifest['world_size']):
        path = checkpoint / ('rank-' + str(rank) + '.pt')
        state = torch.load(path, map_location='cpu', weights_only=False)
        expected = state['state_identity']
        entry = {'state_identity': expected, 'schema': schema(state, torch)}
        rng = state['rng']
        if isinstance(rng['numpy'][1], list):
            entry['independent_rng_sha256'] = semantic_identity(rng, torch)
        rank_states[rank] = entry
        del state
    file_reports = {}
    optimizers = 0
    for name in sorted(manifest['files']):
        if not name.endswith('.pt') or name.startswith('rank-'):
            continue
        is_optimizer = name.endswith('optim_states.pt')
        if is_optimizer and optimizers >= args.optimizer_files:
            continue
        if not is_optimizer and not name.endswith('model_states.pt'):
            continue
        value = torch.load(checkpoint / name, map_location='cpu', weights_only=False)
        entry = {'schema': schema(value, torch), 'candidate_hashes': list(candidates(value, torch))}
        if isinstance(value, dict) and isinstance(value.get('module'), dict):
            module = value['module']
            entry['module_semantic_sha256'] = semantic_identity(module, torch)
            without_vocab = {key: tensor for key, tensor in module.items() if key not in ('d2t', 't2d')}
            entry['module_without_vocabulary_sha256'] = semantic_identity(without_vocab, torch)
        match = re.search(r'zero_pp_rank_(\d+)', name)
        if match:
            rank = int(match.group(1))
            expected = rank_states[rank]['state_identity']
            optimizer_identity = expected.get('optimizer_scaler', expected.get('optimizer_and_scaler'))
            entry['rank'] = rank
            entry['expected_optimizer_identity'] = optimizer_identity
            entry['matching_optimizer_candidate_paths'] = [item['path'] for item in entry['candidate_hashes'] if item['semantic_sha256'] == optimizer_identity]
        file_reports[name] = entry
        optimizers += int(is_optimizer)
        del value
        gc.collect()
    source = Path(inspect.getfile(DeepSpeedZeroOptimizer))
    require(all(digest(checkpoint / name) == item['sha256'] for name, item in inventory.items()), 'checkpoint_changed_during_probe')
    report = {'status': 'format_probe_complete_not_qualification', 'checkpoint': str(checkpoint), 'manifest_sha256': args.manifest_sha256, 'manifest_fields': sorted(manifest), 'runtime': {name: importlib.metadata.version(name) for name in ('torch', 'deepspeed', 'numpy', 'transformers', 'accelerate')}, 'deepspeed_optimizer_source': {'path': str(source), 'sha256': digest(source)}, 'world_size': manifest['world_size'], 'rank_states': rank_states, 'raw_files': file_reports, 'artifact_inventory': inventory, 'all_files_rehashed_before_after': True, 'probe_source_sha256': digest(Path(__file__)), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'wall_seconds': time.monotonic() - start, 'no_tensor_values_logged': True, 'scientific_training_admitted': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': digest(args.output), 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
