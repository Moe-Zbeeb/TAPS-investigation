import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


RUNS = {'eagle3': 'official-eagle3-20261109-v2', 'eagle2': 'official-eagle2-hass-20261109-v1', 'hass': 'official-eagle2-hass-20261109-v1'}
BENCHMARKS = ('mt_bench', 'humaneval', 'gsm8k', 'alpaca', 'cnn_dm')


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


def read(path):
    return json.loads(Path(path).read_bytes())


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def seed_for(*values):
    return int(identity(list(values))[:8], 16)


def hash_partition(token, salt):
    return hashlib.sha256(canonical([20261109, salt, token])).digest()[-1] & 1


def sampling_seed(method, mode, prompt_index, repeat):
    require(method in RUNS and mode in ('target_only', 'speculative') and 0 <= prompt_index < 10 and 0 <= repeat < 1024, 'Invalid independent seed coordinates')
    return 100000000 + list(RUNS).index(method) * 1000000 + ('target_only', 'speculative').index(mode) * 100000 + prompt_index * 1024 + repeat


def continuation_events(tokens, stops):
    values = [hash_partition(tokens[position - 1] if position <= len(tokens) else 'END', salt) for position in (2, 8, 16, 32) for salt in (0, 1)]
    values.extend(int(any(token in stops for token in tokens[:position])) for position in (8, 16, 32))
    grams = [tuple(tokens[index:index + 4]) for index in range(max(0, len(tokens) - 3))]
    values.append(int(len(grams) != len(set(grams))))
    return values


def conditional_bound(trials):
    return math.sqrt(math.log(2 * 960 / 0.025) / (2 * trials))


def sampling_bound(trials):
    return math.sqrt(math.log(2 * 36 / 0.025) / trials)


def prepare(root, destination):
    from taps_rebuild.runner import verify_provenance

    require(os.environ.get('SLURM_JOB_ID'), 'Preparation requires Slurm')
    require(not destination.exists(), 'Preparation output already exists')
    group = root / 'outputs/reference-groups/official-20261110-v1'
    group_manifest, collection = read(group / 'manifest.json'), read(group / 'collection.json')
    require(digest(group / 'manifest.json') == collection['manifest_sha256'], 'Reference collection manifest changed')
    raw_hashes = {item['raw']['path']: item['raw']['sha256'] for run in group_manifest['input_runs'] for item in run['worker_proofs']}
    document = {'scope': 'validation_only_no_training_admission', 'methods': {}, 'baseline_collection_sha256': digest(group / 'collection.json'), 'protocol_sha256': digest(Path(__file__).with_name('PREREGISTRATION.md')), 'sampling_proposal_sha256': digest(Path(__file__).with_name('TEMPERATURE_ONE_AUDIT_PLAN.md')), 'driver_sha256': digest(__file__)}
    with ExitStack() as locks:
        for method, run_id in RUNS.items():
            directory = root / 'outputs/runs' / run_id
            lock = locks.enter_context((directory / 'pool.lock').open('r+b'))
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            manifest, config, sources, source, models, provenance = verify_provenance(root, directory / 'inputs/config.json', directory / 'manifest.json', method)
            cases = {}
            for path in sorted((directory / 'raw').glob(method + '-shard*.jsonl')):
                require(digest(path) == raw_hashes[str(path)], 'Original raw generation file changed')
                with path.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        if row['temperature'] != 0 or row['repeat'] != 0:
                            continue
                        key = identity([row['benchmark'], row['question_id'], row['turn_id']])
                        case = cases.setdefault(key, {'benchmark': row['benchmark'], 'question_id': row['question_id'], 'turn_id': row['turn_id'], 'input_token_ids': row['input_token_ids'], 'input_sha256': row['input_sha256'], 'seed': row['seed'], 'original_case_ids': {}})
                        require(case['input_token_ids'] == row['input_token_ids'] and case['input_sha256'] == identity(row['input_token_ids']), 'Original paired input mismatch')
                        require(row['generation_mode'] not in case['original_case_ids'], 'Duplicate original generation')
                        case['original_case_ids'][row['generation_mode']] = row['case_id']
            cases = sorted(cases.values(), key=lambda x: (x['benchmark'], x['question_id'], x['turn_id']))
            require(len(cases) == (480 if method == 'eagle3' else 320), 'Incomplete greedy reference panel')
            require(all(set(case['original_case_ids']) == {'target_only', 'speculative'} for case in cases), 'Incomplete original paired coverage')
            prompts = []
            for benchmark in BENCHMARKS:
                descriptor = sources['benchmarks'][benchmark]
                require(digest(descriptor['path']) == descriptor['sha256'], 'Benchmark bytes changed')
                rows = [json.loads(line) for line in Path(descriptor['path']).read_text().splitlines()]
                chosen = sorted(rows, key=lambda x: identity([20261109, benchmark, x['question_id']]))[:2]
                prompts.extend({'benchmark': benchmark, 'question_id': row['question_id'], 'text': row['turns'][0], 'benchmark_sha256': descriptor['sha256']} for row in chosen)
            document['methods'][method] = {'run_id': run_id, 'run_manifest': str(directory / 'manifest.json'), 'run_config': str(directory / 'inputs/config.json'), 'provenance': provenance, 'config': config, 'definition': config['methods'][method], 'source': source, 'models': models, 'authored_source_root': manifest['authored_source_root'], 'greedy_cases': cases, 'sampling_prompts': prompts}
    destination.mkdir(parents=True)
    atomic(destination / 'cases.json', document)
    atomic(destination / 'completion.json', {'status': 'complete', 'cases_sha256': digest(destination / 'cases.json'), 'greedy_pairs': 1120, 'sampling_prompts_per_method': 10, 'created_at_utc': datetime.now(timezone.utc).isoformat()})


def backend_for(selected, method, precision):
    import torch
    from taps_rebuild.backends.official import OfficialBackend

    config, definition = selected['config'], selected['definition']
    backend = OfficialBackend(method, selected['source']['path'], selected['models']['target']['path'], selected['models']['drafter']['path'], total_token=config['total_token'], depth=definition['depth'], top_k=config['top_k'])
    if precision == 'float32':
        backend.close()
        arguments = dict(base_model_path=selected['models']['target']['path'], ea_model_path=selected['models']['drafter']['path'], total_token=config['total_token'], depth=definition['depth'], top_k=config['top_k'], torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map={'': 0}, local_files_only=True)
        if method != 'hass':
            arguments['use_eagle3'] = method == 'eagle3'
        backend.model = backend.module.EaModel.from_pretrained(**arguments).eval()
    return backend


def generate(backend, ids, seed, temperature, cap, mode):
    import numpy
    import torch
    from taps_rebuild.runner import decode_completion, seed_runtime

    seed_runtime(seed, torch, numpy)
    result = backend.generate(torch.tensor([ids]), temperature=temperature, max_new_tokens=cap, max_length=2048, generation_mode=mode)
    raw = result['output_ids'][0, len(ids):].detach().cpu().tolist()
    require(all(type(token) is int and 0 <= token < backend.model.base_model.config.vocab_size for token in raw), 'Invalid generated token')
    _, comparison = decode_completion(backend.tokenizer, raw, cap)
    return {'seed': seed, 'output_token_ids': raw, 'comparison_token_ids': comparison, 'cycles': result['cycles'], 'wall_ms': result['wall_total_ms'], 'backend_metadata': {k: v for k, v in result.items() if k not in {'output_ids', 'cycles'}}}


def prompt_ids(backend, selected, prompt):
    from taps_rebuild.runner import encode

    messages = [{'role': 'system', 'content': selected['source']['system_prompt']['text']}, {'role': 'user', 'content': prompt['text']}]
    return encode(backend.tokenizer, messages)[0].tolist()


def capture_cases(backend, ids, seed):
    captures = []
    original_tree, original_posterior = backend.module.tree_decoding, backend.module.evaluate_posterior
    state = {'cycle': 0}

    def tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices):
        state['cycle'] += 1
        state['prefix'] = input_ids.detach().clone()
        return original_tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices)

    def posterior(logits, candidates, logits_processor):
        if state['cycle'] in (1, 3):
            captures.append({'cycle': state['cycle'], 'prefix': state['prefix'], 'logits': logits.detach().clone(), 'candidates': candidates.detach().clone()})
        return original_posterior(logits, candidates, logits_processor)

    backend.module.tree_decoding, backend.module.evaluate_posterior = tree, posterior
    try:
        generate(backend, ids, seed, 1, 32, 'speculative')
    finally:
        backend.module.tree_decoding, backend.module.evaluate_posterior = original_tree, original_posterior
    return captures


def ar_reference(backend, prefix):
    import torch

    with torch.inference_mode():
        backend.module.reset_tree_mode(backend.model)
        backend.model.current_length_data.zero_()
        ar = backend.model.base_model(prefix, past_key_values=backend.model.past_key_values, use_cache=True).logits[0, -1]
        weights = torch.softmax(ar, dim=0)
        q = weights.double() / weights.double().sum()
    return weights, q


def conditional_case(backend, capture, output, name, seed, trials):
    import numpy
    import torch
    from taps_rebuild.runner import seed_runtime, stop_ids

    candidates, logits = capture['candidates'], capture['logits']
    root = int(candidates[0, 0])
    require(bool((candidates[:, 0] == root).all()), 'Candidate roots differ')
    prefix = torch.cat((capture['prefix'], torch.tensor([[root]], device=capture['prefix'].device)), dim=1)
    weights, q = ar_reference(backend, prefix)
    with torch.inference_mode():
        tree_weights = torch.softmax(logits[0, 0], dim=0).double()
        tree_weights /= tree_weights.sum()
    vocabulary = len(q)
    order = torch.argsort(q, descending=True, stable=True)
    masks = torch.zeros((8, vocabulary), dtype=torch.bool, device=q.device)
    for index, count in enumerate((1, 8, 32, 128)):
        masks[index, order[:count]] = True
    stops = stop_ids(backend.tokenizer)
    masks[4, list(stops)] = True
    children = sorted({int(x) for x in candidates[:, 1].tolist() if 0 <= int(x) < vocabulary})
    masks[5, children] = True
    for salt in (0, 1):
        masks[6 + salt] = torch.tensor([hash_partition(token, salt) for token in range(vocabulary)], dtype=torch.bool, device=q.device)
    expected = (masks.double() * q).sum(dim=1).cpu().tolist()
    artifact = output / (name + '.pt')
    torch.save({'prefix': prefix.cpu(), 'candidates': candidates.cpu(), 'tree_logits': logits.cpu(), 'native_ar_weights': weights.cpu(), 'normalized_ar_q': q.cpu(), 'event_masks': masks.cpu()}, artifact)
    histograms, checks = {}, {}
    started = time.perf_counter()
    with torch.inference_mode():
        for sampler in ('posterior', 'target_only'):
            seed_runtime(seed_for(seed, sampler), torch, numpy)
            tokens = []
            for _ in range(trials):
                if sampler == 'posterior':
                    best, accepted, residual = backend.module.evaluate_posterior(logits, candidates, lambda _, values: values)
                    fallback = int(torch.multinomial(residual, 1))
                    token = int(candidates[int(best), 1]) if int(accepted) >= 1 else fallback
                else:
                    token = int(torch.multinomial(weights, 1))
                require(0 <= token < vocabulary, 'Invalid conditional sample')
                tokens.append(token)
            histogram = torch.bincount(torch.tensor(tokens, device=q.device), minlength=vocabulary)
            histograms[sampler] = histogram.cpu().tolist()
            counts = (masks.long() * histogram).sum(dim=1).cpu().tolist()
            radius = conditional_bound(trials)
            checks[sampler] = [{'count': count, 'expected': probability, 'simultaneous_error_upper_bound': abs(count / trials - probability) + radius, 'passed': abs(count / trials - probability) + radius <= 0.02} for count, probability in zip(counts, expected)]
    return {'cycle': capture['cycle'], 'trials_per_sampler': trials, 'capture': artifact.name, 'capture_sha256': digest(artifact), 'ar_prefix_sha256': identity(prefix[0].cpu().tolist()), 'tree_vs_ar_full_vocabulary_tv': float((tree_weights - q).abs().sum() / 2), 'checks': checks, 'histograms': histograms, 'passed': all(event['passed'] for events in checks.values() for event in events), 'wall_seconds': time.perf_counter() - started}


def worker(args):
    import torch
    from taps_rebuild.runner import require_gpu_slurm, stop_ids, verify_provenance

    require_gpu_slurm()
    prepared = read(args.prepared / 'cases.json')
    require(digest(__file__) == prepared['driver_sha256'], 'Frozen driver differs')
    selected = prepared['methods'][args.method]
    verify_provenance(args.root, selected['run_config'], selected['run_manifest'], args.method)
    backend = backend_for(selected, args.method, 'float32' if args.phase == 'greedy' else 'float16')
    metadata = {'method': args.method, 'phase': args.phase, 'worker': args.index, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(), 'torch': torch.__version__, 'prepared_sha256': digest(args.prepared / 'cases.json'), 'provenance': selected['provenance'], 'gpu': torch.cuda.get_device_name(0)}
    completed, failures = 0, 0
    path = args.output / f'{args.phase}-{args.method}-{args.index}.jsonl'
    require(not path.exists(), 'Worker output already exists')
    try:
        with path.open('x') as stream:
            if args.phase == 'greedy':
                for index, case in enumerate(selected['greedy_cases']):
                    if index % args.expected_gpus != args.index:
                        continue
                    outputs = {mode: generate(backend, case['input_token_ids'], case['seed'], 0, 512, mode) for mode in ('target_only', 'speculative')}
                    equal = outputs['target_only']['comparison_token_ids'] == outputs['speculative']['comparison_token_ids']
                    row = {**metadata, 'case': case, 'precision': 'float32', 'equal': equal, 'outputs': outputs}
                    stream.write(canonical(row).decode() + '\n')
                    stream.flush()
                    completed += 1
                    failures += int(not equal)
                    atomic(path.with_suffix('.progress.json'), {'completed': completed, 'failures': failures})
            elif args.phase == 'sampling':
                stops = stop_ids(backend.tokenizer)
                task_index = 0
                for prompt_index, prompt in enumerate(selected['sampling_prompts']):
                    ids = prompt_ids(backend, selected, prompt)
                    for repeat in range(args.sampling_repeats):
                        for mode in ('target_only', 'speculative'):
                            assigned = (task_index // 2) % args.expected_gpus == args.index
                            task_index += 1
                            if not assigned:
                                continue
                            seed = sampling_seed(args.method, mode, prompt_index, repeat)
                            result = generate(backend, ids, seed, 1, 32, mode)
                            row = {**metadata, 'prompt_index': prompt_index, 'prompt': prompt, 'input_token_ids': ids, 'input_sha256': identity(ids), 'repeat': repeat, 'mode': mode, 'events': continuation_events(result['comparison_token_ids'], stops), **result}
                            stream.write(canonical(row).decode() + '\n')
                            completed += 1
                            if completed % 16 == 0:
                                stream.flush()
                                atomic(path.with_suffix('.progress.json'), {'completed': completed})
            elif args.phase == 'conditional':
                for index, prompt in enumerate(selected['sampling_prompts']):
                    if index % args.expected_gpus != args.index:
                        continue
                    ids = prompt_ids(backend, selected, prompt)
                    captures = capture_cases(backend, ids, seed_for(20261109, 'capture', args.method, prompt['benchmark'], prompt['question_id']))
                    failures += int([case['cycle'] for case in captures] != [1, 3])
                    for capture in captures:
                        name = f'capture-{args.method}-{index}-{capture["cycle"]}'
                        result = conditional_case(backend, capture, args.output, name, seed_for(20261110, args.method, index, capture['cycle']), args.conditional_trials)
                        stream.write(canonical({**metadata, 'prompt': prompt, 'prompt_index': index, **result}).decode() + '\n')
                        stream.flush()
                        completed += 1
                        failures += int(not result['passed'])
                        atomic(path.with_suffix('.progress.json'), {'completed': completed, 'failures': failures})
            os.fsync(stream.fileno())
        atomic(path.with_suffix('.receipt.json'), {**metadata, 'status': 'complete', 'completed': completed, 'failed_checks': failures, 'raw_sha256': digest(path)})
    finally:
        backend.close()


def health(args):
    import torch

    require(os.environ.get('SLURM_JOB_ID'), 'Health check requires an allocation')
    require(torch.cuda.device_count() == args.expected_gpus, 'CUDA device count differs from full grant')
    devices = []
    for index in range(args.expected_gpus):
        with torch.cuda.device(index):
            x = torch.ones((256, 256), device=f'cuda:{index}', dtype=torch.float16)
            require(bool(torch.isfinite(x @ x).all()), 'CUDA matrix check failed')
            torch.cuda.synchronize(index)
            p = torch.cuda.get_device_properties(index)
            require(p.total_memory >= 75 * 1024 ** 3, 'This validation queue requires 80GB-class GPUs')
            devices.append({'logical_device': index, 'name': p.name, 'memory_bytes': p.total_memory, 'uuid': str(getattr(p, 'uuid', 'unavailable'))})
    print(json.dumps({'status': 'passed', 'devices': devices, 'torch': torch.__version__}), flush=True)


def hardware(expected):
    from taps_rebuild.attach import fields, gpu_count

    job_id = os.environ.get('SLURM_JOB_ID', '')
    require(job_id.isdigit(), 'Controller requires Slurm')
    job = fields(subprocess.check_output(['/usr/local/bin/scontrol', 'show', 'job', '-o', job_id], text=True, timeout=30))
    node = socket.gethostname().split('.')[0]
    require(job.get('JobState') == 'RUNNING' and job.get('NodeList') == node and job.get('NumNodes') == '1' and job.get('UserId', '').endswith('(' + str(os.getuid()) + ')'), 'Allocation is not owned on this node')
    require(gpu_count(job.get('AllocTRES') or job['TRES']) == expected, 'Allocation GPU count differs')
    inventory = fields(subprocess.check_output(['/usr/local/bin/scontrol', 'show', 'node', '-o', node], text=True, timeout=30))
    require(gpu_count(inventory['CfgTRES']) == expected, 'Full-node GPU ownership required')
    tokens = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if tokens == [''] and node == 'deep-chungus-10' and expected == 8:
        tokens = [str(index) for index in range(8)]
    require(len(tokens) == expected and len(set(tokens)) == expected and all(token.strip() for token in tokens), 'Invalid assigned CUDA device list')
    return {'job': job, 'node': inventory, 'tokens': tokens}


def summarize(output, phase, expected, repeats, trials):
    raw = sorted(output.glob(phase + '-*.jsonl'))
    counts, seen, events, failed = {}, set(), {}, 0
    for path in raw:
        receipt = read(path.with_suffix('.receipt.json'))
        require(receipt['status'] == 'complete' and receipt['raw_sha256'] == digest(path), 'Worker proof missing or changed')
        failed += receipt['failed_checks']
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                method = row['method']
                counts[method] = counts.get(method, 0) + 1
                if phase == 'greedy':
                    key = (method, identity(row['case']))
                elif phase == 'sampling':
                    key = (method, row['prompt_index'], row['repeat'], row['mode'])
                    totals = events.setdefault((method, row['mode']), [0] * 12)
                    for index, value in enumerate(row['events']):
                        totals[index] += value
                else:
                    key = (method, row['prompt_index'], row['cycle'])
                require(key not in seen, 'Duplicate audit case')
                seen.add(key)
    required = {'eagle3': 480, 'eagle2': 320, 'hass': 320} if phase == 'greedy' else {method: 20 * repeats if phase == 'sampling' else 20 for method in RUNS}
    complete = counts == required and len(raw) == 3 * expected
    bounds = {}
    if phase == 'sampling' and complete:
        radius = sampling_bound(10 * repeats)
        for method in RUNS:
            differences = [abs(a - b) / (10 * repeats) for a, b in zip(events[(method, 'target_only')], events[(method, 'speculative')])]
            bounds[method] = [{'difference': difference, 'simultaneous_error_upper_bound': difference + radius, 'passed': difference + radius <= 0.06} for difference in differences]
        failed += sum(not event['passed'] for values in bounds.values() for event in values)
    report = {'scope': 'validation_only_no_training_admission', 'phase': phase, 'execution_complete': complete, 'counts': counts, 'failed_checks': failed, 'passed_scoped_checks': complete and failed == 0, 'sampling_event_bounds': bounds, 'conditional_trials': trials if phase == 'conditional' else None, 'training_admission': False, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    atomic(output / (phase + '-summary.json'), report)
    return report


def controller(args):
    from taps_rebuild.pool import worker_environment

    prepared = read(args.prepared / 'cases.json')
    require(digest(__file__) == prepared['driver_sha256'], 'Driver changed after preparation')
    require(digest(args.prepared / 'cases.json') == read(args.prepared / 'completion.json')['cases_sha256'], 'Prepared cases changed')
    allocation = hardware(args.expected_gpus)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'controller.lock').open('x+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(args.output / 'manifest.json', {'prepared_sha256': digest(args.prepared / 'cases.json'), 'driver_sha256': digest(__file__), 'hardware': allocation, 'phases': args.phases, 'sampling_repeats': args.sampling_repeats, 'conditional_trials': args.conditional_trials})
        python = Path(prepared['methods']['eagle3']['config']['legacy_runtime_root']) / 'envs/taps-eagle/bin/python'
        environment = worker_environment(allocation['tokens'][0], Path(prepared['methods']['eagle3']['authored_source_root']), args.root, python)
        environment['CUDA_VISIBLE_DEVICES'] = ','.join(allocation['tokens'])
        health_result = subprocess.run([str(python), '-B', __file__, '--action', 'health', '--expected-gpus', str(args.expected_gpus)], env=environment, capture_output=True, text=True, timeout=180)
        atomic(args.output / 'health.json', {'returncode': health_result.returncode, 'stdout': health_result.stdout, 'stderr': health_result.stderr})
        require(health_result.returncode == 0, 'Full-node CUDA health check failed')
        for phase in args.phases.split(','):
            for method, selected in prepared['methods'].items():
                python = Path(selected['config']['legacy_runtime_root']) / 'envs' / ('taps-' + selected['definition']['profile']) / 'bin/python'
                children = []
                for index, token in enumerate(allocation['tokens']):
                    environment = worker_environment(token, Path(selected['authored_source_root']), args.root, python)
                    environment['PATH'] = str(python.parent) + ':/usr/local/bin:/usr/bin:/bin'
                    command = [str(python), '-B', __file__, '--action', 'worker', '--root', str(args.root), '--prepared', str(args.prepared), '--output', str(args.output), '--phase', phase, '--method', method, '--index', str(index), '--expected-gpus', str(args.expected_gpus), '--sampling-repeats', str(args.sampling_repeats), '--conditional-trials', str(args.conditional_trials)]
                    log = (args.output / f'{phase}-{method}-{index}.log').open('xb')
                    child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),), start_new_session=True)
                    children.append((child, log))
                codes = []
                try:
                    while any(child.poll() is None for child, _ in children):
                        require(not any(child.poll() not in (None, 0) for child, _ in children), 'A validation worker failed')
                        time.sleep(1)
                    codes = [child.returncode for child, _ in children]
                finally:
                    for child, log in children:
                        if child.poll() is None:
                            child.terminate()
                    for child, log in children:
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=10)
                        log.close()
                require(not any(codes), 'Worker wave failed: ' + repr((phase, method, codes)))
                print(json.dumps({'completed_wave': method, 'phase': phase}), flush=True)
            report = summarize(args.output, phase, args.expected_gpus, args.sampling_repeats, args.conditional_trials)
        atomic(args.output / 'completion.json', {'status': 'complete', 'meaning': 'All worker waves completed operationally; consult each phase summary for coverage and statistical outcomes', 'phases': args.phases, 'training_admission': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['prepare', 'controller', 'worker', 'health'], required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-gpus', type=int, default=8)
    parser.add_argument('--phases', default='greedy,conditional')
    parser.add_argument('--phase', choices=['greedy', 'sampling', 'conditional'])
    parser.add_argument('--method', choices=list(RUNS))
    parser.add_argument('--index', type=int)
    parser.add_argument('--sampling-repeats', type=int, default=1024)
    parser.add_argument('--conditional-trials', type=int, default=65536)
    args = parser.parse_args()
    require(1 <= args.sampling_repeats <= 1024 and 1 <= args.conditional_trials <= 65536, 'Invalid sampling budget')
    if args.action == 'prepare':
        prepare(args.root, args.output)
    elif args.action == 'worker':
        worker(args)
    elif args.action == 'health':
        health(args)
    else:
        controller(args)


if __name__ == '__main__':
    main()
