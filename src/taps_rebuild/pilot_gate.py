import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import native_training as native, pool
from .pilot_cache import read, records, require, safe_positions, verified_corpus


def verify_cache(cache, rank=None):
    cache = Path(cache)
    receipt = read(cache / 'completion.json')
    require(receipt['status'] == 'complete_gate_scan_required' and receipt['pilot_training_admitted'] is False and receipt['world_size'] == 8 and receipt['length'] == 2048, 'Paired cache is not complete')
    identity = dict(receipt)
    require(identity.pop('identity_sha256') == pool.identity(identity), 'Paired cache receipt identity changed')
    names = receipt['files'] if rank is None else ['rank-' + str(rank) + '.' + suffix for suffix in ('ids', 'masks', 'jsonl')]
    for name in names:
        item, path = receipt['files'][name], cache / name
        require(path.stat().st_size == item['size_bytes'] and pool.digest(path) == item['sha256'], 'Paired cache bytes changed: ' + name)
    return receipt


def upstream(root):
    path = Path(root) / 'repos/EAGLE/eagle/traineagle3'
    require(all(pool.digest(path / name) == expected for name, expected in native.UPSTREAM.items()), 'Pinned native source changed')
    sys.path.insert(0, str(path))
    return path


def backend(torch):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {'torch': torch.__version__, 'cuda': torch.version.cuda, 'matmul_tf32': False, 'cudnn_tf32': False, 'attention': 'unchanged pinned custom LlamaAttention', 'target_dtype': 'float16', 'batch': 1, 'length': 2048, 'attention_mask': 'all ones', 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def devices():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    require(visible == [str(index) for index in range(8)], 'Pilot requires ordered physical devices0..7')
    command = ['/usr/bin/nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv,noheader,nounits']
    inventory = [line.strip().split(', ') for line in subprocess.check_output(command, text=True, timeout=20).splitlines()]
    require(len(inventory) == 8 and [row[0] for row in inventory] == visible and len({row[1] for row in inventory}) == 8 and all('A100' in row[2] and int(row[3]) > 80000 for row in inventory), 'Pilot hardware inventory differs')
    return inventory


def worker(root, cache, output):
    require(os.environ.get('SLURM_JOB_ID') and int(os.environ['WORLD_SIZE']) == 8, 'Eight-GPU Slurm launch is required')
    import numpy as np
    import torch

    root, cache, output = Path(root).resolve(), Path(cache).resolve(), Path(output).resolve()
    require(output.is_relative_to(root / 'data'), 'Eligibility masks belong under project data/')
    rank = int(os.environ['LOCAL_RANK'])
    require(rank == int(os.environ['RANK']) and torch.cuda.device_count() == 8, 'Pilot scan rank mapping differs')
    torch.cuda.set_device(rank)
    runtime, hardware = backend(torch), devices()
    receipt = verify_cache(cache, rank)
    assets = read(root / 'outputs/preparation/assets.json')
    target = native.model_record(assets, 'llama31_instruct_8b', verify_content=rank == 0)
    ancestor = native.model_record(assets, 'eagle3_llama31_8b', verify_content=rank == 0)
    require(target == receipt['target'] and ancestor == receipt['ancestor'], 'Eligibility model inventory differs from cache')
    upstream(root)
    state = torch.load(Path(ancestor['path']) / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    native.validate_vocabulary(torch, state['d2t'], state['t2d'], 128256, 32000)
    vocabulary = native.tensor_state_identity({key: state[key] for key in ('d2t', 't2d')}, torch)
    t2d = state['t2d'].to(rank)
    del state
    output.mkdir(parents=True, exist_ok=True)
    count = receipt['rank_examples'][rank]
    bound = {'cache_identity_sha256': receipt['identity_sha256'], 'target_identity_sha256': target['model_identity_sha256'], 'ancestor_identity_sha256': ancestor['model_identity_sha256'], 'vocabulary_identity_sha256': vocabulary, 'source_sha256': pool.digest(__file__), 'native_helper_sha256': pool.digest(native.__file__), 'upstream': native.UPSTREAM, 'runtime': runtime, 'hardware': hardware, 'rank': rank, 'examples': count}
    identity = pool.identity(bound)
    prefix = output / ('rank-' + str(rank))
    with prefix.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress_path, completion = prefix.with_suffix('.progress.json'), prefix.with_suffix('.complete.json')
        if completion.exists():
            report = read(completion)
            require(report['binding_identity_sha256'] == identity and pool.digest(prefix.with_suffix('.gates')) == report['gates_sha256'], 'Completed eligibility scan changed')
            return report
        done, seconds = 0, 0.0
        if progress_path.exists():
            progress = read(progress_path)
            require(progress['binding_identity_sha256'] == identity, 'Eligibility scan cannot resume across changed bindings')
            done, seconds = progress['completed_examples'], progress['target_forward_wall_seconds']
        ids = np.memmap(cache / ('rank-' + str(rank) + '.ids'), mode='r', dtype='<i4', shape=(count, 2, 2048))
        gates = np.memmap(prefix.with_suffix('.gates'), mode='r+' if progress_path.exists() else 'w+', dtype='u1', shape=(count, 2, 2048))
        model = importlib.import_module('modeling_llama_kv').LlamaForCausalLM.from_pretrained(target['path'], torch_dtype=torch.float16).to(rank).eval()
        require(not model.training and all(not module.training for module in model.modules()), 'Teacher is not in evaluation mode')
        attention = torch.ones((1, 2048), dtype=torch.long, device=rank)
        for index in range(done, count):
            torch.cuda.synchronize(rank)
            started = time.monotonic()
            with torch.no_grad():
                for policy in (0, 1):
                    inputs = torch.as_tensor(np.array(ids[index, policy], dtype='int64'), device=rank)[None]
                    result = model(input_ids=inputs, attention_mask=attention)
                    require(bool(torch.isfinite(result.logits).all()), 'Nonfinite teacher gate scan')
                    argmax = result.logits[0, 1:].argmax(-1)
                    gates[index, policy, :-1] = t2d[argmax].cpu().numpy()
                    gates[index, policy, -1] = 0
                    del result
            torch.cuda.synchronize(rank)
            seconds += time.monotonic() - started
            if (index + 1) % 32 == 0 or index + 1 == count:
                gates.flush()
                with prefix.with_suffix('.gates').open('rb') as stream:
                    os.fsync(stream.fileno())
                pool.atomic_json(progress_path, {'binding_identity_sha256': identity, 'completed_examples': index + 1, 'target_forward_wall_seconds': seconds})
                print(json.dumps({'rank': rank, 'complete': index + 1, 'total': count}), flush=True)
        report = {'status': 'complete_eligibility_scan', 'binding': bound, 'binding_identity_sha256': identity, 'gates_sha256': pool.digest(prefix.with_suffix('.gates')), 'target_forward_wall_seconds': seconds, 'completed_examples': count, 'training_admitted': False, 'hidden_states_cached': False}
        pool.atomic_json(completion, report)
        return report


def provenance(root, cache, config_path):
    config = read(config_path)
    receipt = read(Path(cache) / 'completion.json')
    original_config = read(receipt['config_path'])
    require(pool.digest(receipt['config_path']) == receipt['config_sha256'], 'Original cache configuration changed')
    relaxed = json.loads(json.dumps(config))
    for item in relaxed['corpora']:
        item.pop('generated_completion_sha256')
    require(relaxed == original_config or config == original_config, 'Provenance audit changed cache selection recipe')
    proofs, counts, seen = [], {}, set()
    for descriptor in config['corpora']:
        original, generated, bound = verified_corpus(Path(root), descriptor, receipt['target']['model_identity_sha256'])
        proofs.extend(bound)
        for split in ('train', 'dev', 'test'):
            for row in records(generated / (split + '.jsonl')):
                require(row['split'] == split and row['example_id'] not in seen and row['response_regime'] == 'target_generated' and row['original_response_complete'] is True, 'Generated rows have duplicate IDs, physical split or regime disagreement')
                native.native_row(row, canonical_messages=True, split=split)
                seen.add(row['example_id'])
                key = str(generated) + ':' + split
                counts[key] = counts.get(key, 0) + 1
    require(all(pool.digest(item['path']) == item['sha256'] for item in proofs), 'Provenance audit input changed while reading')
    report = {'status': 'passed', 'scope': 'strengthened provenance audit of immutable pre-eligibility cache', 'config_path': str(Path(config_path).resolve()), 'config_sha256': pool.digest(config_path), 'cache_identity_sha256': receipt['identity_sha256'], 'cache_preparation_source_sha256': receipt['adapter_sha256'], 'audit_source_sha256': pool.digest(__file__), 'proofs': proofs, 'generated_split_counts': counts, 'training_admitted': False}
    report['identity_sha256'] = pool.identity(report)
    return report


def finalize_unlocked(root, cache, output, config):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Eligibility summary requires CPU Slurm')
    import numpy as np

    root, cache, output = Path(root).resolve(), Path(cache).resolve(), Path(output).resolve()
    require(output.is_relative_to(root / 'data'), 'Eligibility masks belong under project data/')
    receipt = verify_cache(cache)
    audit = provenance(root, cache, config)
    completion = output / 'completion.json'
    if completion.exists():
        previous = read(completion)
        identity = dict(previous)
        require(identity.pop('identity_sha256') == pool.identity(identity) and previous['shared_binding']['source_sha256'] == pool.digest(__file__) and previous['provenance_audit']['config_sha256'] == pool.digest(config), 'Completed eligibility summary binding changed')
        require(all(pool.digest(output / name) == item['sha256'] for name, item in previous['files'].items()), 'Completed eligibility summary artifact changed')
        require(previous['status'] == 'eligible_for_schedule_preparation', 'Eligibility population is insufficient')
        return previous
    scans = [read(output / ('rank-' + str(rank) + '.complete.json')) for rank in range(8)]
    require(all(item['status'] == 'complete_eligibility_scan' and item['binding']['rank'] == rank and item['binding']['cache_identity_sha256'] == receipt['identity_sha256'] for rank, item in enumerate(scans)), 'Eligibility rank coverage differs')
    shared = [{key: value for key, value in item['binding'].items() if key not in ('rank', 'examples')} for item in scans]
    require(all(value == shared[0] for value in shared), 'Eligibility ranks disagree on immutable bindings')
    require(shared[0]['source_sha256'] == pool.digest(__file__) and shared[0]['native_helper_sha256'] == pool.digest(native.__file__) and shared[0]['upstream'] == native.UPSTREAM and shared[0]['target_identity_sha256'] == receipt['target']['model_identity_sha256'] and shared[0]['ancestor_identity_sha256'] == receipt['ancestor']['model_identity_sha256'], 'Eligibility scan source/model binding differs from current pinned contract')
    populations = {quota: {domain: [0] * 8 for domain in ('chat', 'math', 'code')} for quota in (64, 32)}
    split_counts = {quota: {} for quota in (64, 32)}
    pool_files = {}
    for rank, scan in enumerate(scans):
        prefix = output / ('rank-' + str(rank))
        require(pool.identity(scan['binding']) == scan['binding_identity_sha256'] and pool.digest(prefix.with_suffix('.gates')) == scan['gates_sha256'], 'Eligibility scan identity changed')
        count = receipt['rank_examples'][rank]
        require(scan['completed_examples'] == scan['binding']['examples'] == count and prefix.with_suffix('.gates').stat().st_size == count * 2 * 2048, 'Eligibility scan rank coverage/geometry differs')
        gates = np.memmap(prefix.with_suffix('.gates'), mode='r', dtype='u1', shape=(count, 2, 2048))
        masks = np.memmap(cache / ('rank-' + str(rank) + '.masks'), mode='r', dtype='u1', shape=(count, 2, 2048))
        require(bool(((gates == 0) | (gates == 1)).all()) and bool(((masks == 0) | (masks == 1)).all()), 'Eligibility/cache mask is not binary')
        path = output / ('rank-' + str(rank) + '.eligible.jsonl')
        temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
        observed = 0
        with temporary.open('xb') as stream:
            for index, row in enumerate(records(cache / ('rank-' + str(rank) + '.jsonl'))):
                require(row['rank'] == rank and row['index'] == index and len(row['lengths']) == 2 and all(type(value) is int and 0 < value <= 2048 for value in row['lengths']), 'Cache rank index/lengths differ')
                observed += 1
                eligible = [safe_positions((masks[index, policy] & gates[index, policy]).tolist(), row['lengths'][policy]) for policy in (0, 1)]
                row['eligible_positions'] = eligible
                stream.write(native.canonical(row) + b'\n')
                for quota in (64, 32):
                    if min(map(len, eligible)) >= quota:
                        key = row['domain'] + ':' + row['split']
                        split_counts[quota][key] = split_counts[quota].get(key, 0) + 1
                        if row['split'] == 'train':
                            populations[quota][row['domain']][rank] += 1
            stream.flush()
            os.fsync(stream.fileno())
        require(observed == count, 'Cache metadata row count differs')
        if path.exists():
            require(pool.digest(path) == pool.digest(temporary), 'Previously published eligibility pool differs')
            temporary.unlink()
        else:
            temporary.replace(path)
        pool_files[path.name] = {'sha256': pool.digest(path), 'size_bytes': path.stat().st_size}
        for artifact in (prefix.with_suffix('.gates'), prefix.with_suffix('.complete.json')):
            pool_files[artifact.name] = {'sha256': pool.digest(artifact), 'size_bytes': artifact.stat().st_size}
    quota = next((value for value in (64, 32) if all(count >= 128 for domain in populations[value].values() for count in domain)), None)
    report = {'status': 'eligible_for_schedule_preparation' if quota else 'blocked_insufficient_common_eligibility', 'cache_identity_sha256': receipt['identity_sha256'], 'scan_bindings': [item['binding_identity_sha256'] for item in scans], 'shared_binding': shared[0], 'files': pool_files, 'train_populations': populations, 'split_counts': split_counts, 'quota': quota, 'minimum_per_domain_rank': 128, 'gate_rule': 'native_mask[s] and t2d[argmax_full_vocab(target_logits_original[s+1])] and6<=s and s+1<real_length', 'training_admitted': False, 'target_forward_rank_seconds': [item['target_forward_wall_seconds'] for item in scans], 'provenance_audit': audit}
    report['identity_sha256'] = pool.identity(report)
    require(all(pool.digest(output / name) == item['sha256'] for name, item in pool_files.items()), 'Eligibility artifacts changed during finalization')
    pool.atomic_json(output / 'completion.json', report)
    require(quota is not None, 'Insufficient common paired effective contexts; do not launch training')
    return report


def finalize(root, cache, output, config):
    output = Path(output)
    require(output.is_dir(), 'Eligibility output does not exist')
    with (output / 'finalize.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return finalize_unlocked(root, cache, output, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('worker', 'finalize'))
    for name in ('root', 'cache', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    operation = worker if args.action == 'worker' else finalize
    options = [args.config] if args.action == 'finalize' else []
    require(args.action != 'finalize' or args.config is not None, 'Finalization requires the pinned provenance configuration')
    print(json.dumps(operation(args.root, args.cache, args.output, *options), indent=2))


if __name__ == '__main__':
    main()
