import argparse
import ast
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from .pool import allocation, atomic_json, digest, identity, terminate_children


MODULES = ['__init__.py', 'engineering.py', 'pool.py', 'metrics.py', 'data.py', 'learning.py', 'composition.py', 'backends/__init__.py', 'backends/official.py', 'backends/pooled.py']


def comparison(tokens, tokenizer, limit):
    stops = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')}
    end = next((i + 1 for i, token in enumerate(tokens) if token in stops), len(tokens))
    return tokens[:min(end, limit)]


def run_components(root, run_id, rank, torch, config_path=None):
    from . import composition, data, learning
    started = time.perf_counter()
    trained = []
    for mode in ['separate_heads', 'lora_output']:
        config = json.loads(Path(config_path or root / 'configs/engineering-training.json').read_text())
        config['run_id'] = run_id + '-gpu' + str(rank) + '-' + mode.replace('_', '-')
        config['seed'] += rank
        config['model']['expert_mode'] = mode
        first = learning.train(config, root, device='cuda:0', stop_after_steps=1)
        result = learning.train(config, root, device='cuda:0', resume='latest')
        if result['status'] != 'complete' or not result['budget_eligibility']['eligible']:
            raise RuntimeError('Engineering trainer did not complete after checkpoint resume')
        trained.append({'mode': mode, 'first': first, 'resumed': result})
        gc.collect()
        torch.cuda.empty_cache()
    drafts = [
        {'expert_id': 'a', 'root_token': 1, 'root_source': 'target_greedy', 'nodes': [{'path': [2], 'log_probability': -0.2}, {'path': [2, 3], 'log_probability': -0.5}]},
        {'expert_id': 'b', 'root_token': 1, 'root_source': 'target_greedy', 'nodes': [{'path': [2], 'log_probability': -0.3}, {'path': [4], 'log_probability': -0.4}]},
    ]
    tree = composition.pooled_budget_tree(drafts, root_token=1, root_source='target_greedy', budget=4, score_mode='uncalibrated_diagnostic')
    q = torch.tensor([.05, .15, .3, .5], device='cuda:0')
    generator = torch.Generator(device='cuda:0').manual_seed(20261109 + rank)
    counts = [0] * 4
    for _ in range(1000):
        sampled = composition.sample_target_children(q, [0, 0, 1], generator=generator)
        counts[int(sampled['token'])] += 1
    distribution_error = sum(abs(count / 1000 - probability) for count, probability in zip(counts, [.05, .15, .3, .5])) / 2
    if distribution_error > .1:
        raise RuntimeError('GPU conditional sampler sanity check differs excessively from target')
    tree_q = torch.tensor([.05, .1, .2, .3, .35], device='cuda:0')
    verified = composition.verify_tree(tree, {index: tree_q for index in range(len(tree['tokens']))}, greedy=True)
    base = {'w': torch.zeros(4, 4, device='cuda:0')}
    experts = {'a': {'w': torch.ones(4, 4, device='cuda:0')}, 'b': {'w': torch.full((4, 4), 2., device='cuda:0')}}
    merges = {}
    for method in ['linear', 'task_arithmetic', 'ties', 'dare']:
        result = composition.merge_weights(base, experts, ancestor_identity='a' * 64, expert_ancestors={'a': 'a' * 64, 'b': 'a' * 64}, method=method, density=.5, seed=20261109 + rank)
        if not torch.isfinite(result['state_dict']['w']).all().item():
            raise RuntimeError('GPU weight merge produced nonfinite values')
        merges[method] = {'sum': float(result['state_dict']['w'].sum().item()), 'metadata': result['metadata']}

    class TinyTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            result = [1]
            for item in messages:
                result += [2 if item['role'] == 'user' else 3]
                result += [4 + ord(char) % 27 for char in item['content']]
                result += [31]
            if add_generation_prompt:
                result += [3]
            return result

    class TinyTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 8)
            self.head = torch.nn.Linear(8, 32)

        def forward(self, tokens, **kwargs):
            hidden = self.embedding(tokens)
            return SimpleNamespace(hidden_states=(hidden, hidden * 2, hidden * 3), logits=self.head(hidden))

    example = data.feature_tensors(torch, TinyTokenizer(), TinyTeacher().cuda().eval(), [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'ok'}], 0, [0, 1, 2], 32, True)
    if example['features'].shape[-1] != 24 or not torch.equal(example['labels'][:-1], example['input_ids'][1:]):
        raise RuntimeError('GPU teacher feature alignment differs from the declared next-token contract')
    torch.cuda.synchronize()
    return {'status': 'passed', 'training': trained, 'pooled_tree': tree, 'target_tree_verification': verified, 'conditional_counts': counts, 'conditional_empirical_tv': distribution_error, 'sampling_scope': '1000-draw GPU engineering sanity check, not a full-model distribution test', 'weight_merges': merges, 'feature_shapes': {key: list(value.shape) for key, value in example.items() if hasattr(value, 'shape')}, 'elapsed_seconds': time.perf_counter() - started}


def run_models(root, manifest, torch):
    from .backends.official import OfficialBackend
    from .backends.pooled import PooledEagle3Backend
    assets, sources = manifest['assets'], manifest['sources']
    target, draft = assets['models']['llama31_instruct_8b'], assets['models']['eagle3_llama31_8b']
    source = sources['sources']['EAGLE']
    backend = OfficialBackend('eagle3', source['path'], target['path'], draft['path'])
    prompt = backend.tokenizer.apply_chat_template([{'role': 'system', 'content': source['system_prompt']['text']}, {'role': 'user', 'content': 'Explain why two plus two equals four in one short sentence.'}], tokenize=False, add_generation_prompt=True)
    ids = backend.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda:0')
    records = {}
    reference = None
    try:
        for mode in ['target_only', 'speculative']:
            torch.manual_seed(20261109)
            torch.cuda.manual_seed_all(20261109)
            result = backend.generate(ids, temperature=0, max_new_tokens=32, generation_mode=mode)
            tokens = result['output_ids'][0, ids.shape[1]:].cpu().tolist()
            useful = comparison(tokens, backend.tokenizer, 32)
            reference = useful if reference is None else reference
            if useful != reference:
                raise RuntimeError('Released EAGLE-3 greedy output differs from same-backend target-only decoding')
            records[mode] = {key: value for key, value in result.items() if key != 'output_ids'}
            records[mode]['output_token_ids'] = tokens
            records[mode]['comparison_token_ids'] = useful
    finally:
        backend.close()
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    pooled = PooledEagle3Backend(source['path'], target['path'], {'a': draft['path'], 'b': draft['path']}, target_identity=target['model_identity_sha256'], draft_identities={'a': draft['model_identity_sha256'], 'b': draft['model_identity_sha256']}, expert_target_identities={'a': target['model_identity_sha256'], 'b': target['model_identity_sha256']}, engineering_uncalibrated=True)
    try:
        torch.manual_seed(20261109)
        torch.cuda.manual_seed_all(20261109)
        result = pooled.generate(ids, temperature=0, max_new_tokens=32)
        tokens = result['output_ids'][0, ids.shape[1]:].cpu().tolist()
        useful = comparison(tokens, pooled.tokenizer, 32)
        if useful != reference:
            raise RuntimeError('Native two-drafter pooled greedy output differs from target-only decoding')
        records['pooled_two_identical_drafts'] = {key: value for key, value in result.items() if key != 'output_ids'}
        records['pooled_two_identical_drafts'].update(output_token_ids=tokens, comparison_token_ids=useful)
    finally:
        pooled.close()
    return {'status': 'passed', 'target_identity': target['model_identity_sha256'], 'draft_identity': draft['model_identity_sha256'], 'source_revision': source['revision'], 'records': records, 'scope': 'Released-model engineering sanity check only; no published reproduction or specialist-quality claim'}


def worker(root, directory, manifest_path, rank, models):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError('Engineering worker requires one allocated visible CUDA device')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['source_hashes'].items():
        if digest(directory / 'source' / name) != expected:
            raise RuntimeError('Engineering source snapshot differs')
    if digest(directory / 'inputs/engineering-training.json') != manifest['training_config_sha256']:
        raise RuntimeError('Engineering training configuration snapshot differs')
    torch.cuda.set_device(0)
    torch.manual_seed(20261109 + rank)
    torch.cuda.reset_peak_memory_stats()
    output = {'status': 'running', 'scope': 'engineering_not_scientific_results', 'rank': rank, 'manifest_sha256': digest(manifest_path), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__)}
    path = directory / ('worker-' + str(rank) + '.json')
    atomic_json(path, output)
    try:
        output['components'] = run_components(root, manifest['run_id'], rank, torch, directory / 'inputs/engineering-training.json')
        if models:
            output['models'] = run_models(root, manifest, torch)
        output.update(status='passed', peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        output.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        atomic_json(path, output)


def controller(root, run_id, expected_gpus, models):
    hardware = allocation(expected_gpus)
    directory = root / 'outputs/engineering' / run_id
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / 'source/src/taps_rebuild'
    source.mkdir(parents=True)
    hashes = {}
    for name in MODULES:
        original, destination = root / 'src/taps_rebuild' / name, source / name
        contents = original.read_bytes()
        ast.parse(contents)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
        hashes[str(destination.relative_to(directory / 'source'))] = hashlib.sha256(contents).hexdigest()
    assets = json.loads((root / 'outputs/preparation/assets.json').read_text()) if models else None
    sources = json.loads((root / 'outputs/preparation/sources.json').read_text()) if models else None
    if models and (assets['status'] != 'complete' or sources['status'] != 'complete'):
        raise RuntimeError('Released-model GPU smoke requires complete verified preparation')
    config = json.loads((root / 'configs/reproduction.json').read_text())
    (directory / 'inputs').mkdir()
    shutil.copyfile(root / 'configs/engineering-training.json', directory / 'inputs/engineering-training.json')
    manifest = {'run_id': run_id, 'scope': 'engineering_not_scientific_results', 'source_hashes': hashes, 'assets': assets, 'sources': sources, 'hardware': hardware, 'training_config_sha256': digest(root / 'configs/engineering-training.json'), 'models_enabled': models}
    manifest_path = directory / 'manifest.json'
    atomic_json(manifest_path, manifest)
    python = str(Path(config['legacy_runtime_root']) / 'envs/taps-eagle/bin/python')
    children, logs = [], []
    previous = {}
    def interrupted(number, frame):
        raise RuntimeError('Engineering controller interrupted by signal ' + str(number))
    for number in [signal.SIGTERM, signal.SIGHUP, signal.SIGINT]:
        previous[number] = signal.signal(number, interrupted)
    try:
        for binding in hardware['device_bindings']:
            rank = binding['shard_index']
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=binding['token'], PYTHONPATH=str(directory / 'source/src'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            command = [python, '-B', '-m', 'taps_rebuild.engineering', '--root', str(root), '--run-id', run_id, '--expected-gpus', str(expected_gpus), '--worker', str(rank)]
            if models:
                command.append('--models')
            log = (directory / ('worker-' + str(rank) + '.log')).open('wb')
            logs.append(log)
            children.append(subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        pending = set(children)
        while pending:
            for child in list(pending):
                code = child.poll()
                if code is None:
                    continue
                pending.remove(child)
                if code:
                    raise RuntimeError('Engineering GPU worker failed: ' + str(code))
            if pending:
                time.sleep(.25)
        workers = [json.loads((directory / ('worker-' + str(rank) + '.json')).read_text()) for rank in range(expected_gpus)]
        if any(item['status'] != 'passed' for item in workers):
            raise RuntimeError('Engineering worker receipts are incomplete')
        atomic_json(directory / 'summary.json', {'status': 'passed', 'scope': 'engineering_not_scientific_results', 'gpu_count': expected_gpus, 'workers': workers, 'manifest_sha256': digest(manifest_path)})
        print(json.dumps({'status': 'passed', 'gpu_count': expected_gpus, 'path': str(directory)}), flush=True)
    except BaseException as error:
        terminate_children(children)
        atomic_json(directory / 'summary.json', {'status': 'failed', 'scope': 'engineering_not_scientific_results', 'error_type': type(error).__name__, 'error': str(error), 'gpu_count': expected_gpus, 'manifest_sha256': digest(manifest_path)})
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        for handle in logs:
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--expected-gpus', type=int, default=8)
    parser.add_argument('--models', action='store_true')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if not args.run_id or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for char in args.run_id):
        raise ValueError('Run ID must be a simple experiment name')
    root = args.root.resolve()
    if args.worker is None:
        controller(root, args.run_id, args.expected_gpus, args.models)
    else:
        directory = root / 'outputs/engineering' / args.run_id
        worker(root, directory, directory / 'manifest.json', args.worker, args.models)


if __name__ == '__main__':
    main()
