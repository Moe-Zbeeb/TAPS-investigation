import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import torch
from transformers import AutoTokenizer


ROOT = Path('/mnt/nfs/home/mohamadzbib/projects/taps-rebuild')
BUNDLE = ROOT / 'outputs/preparation/pipeline-validation-20261111-v2'
DESTINATION = ROOT / 'outputs/pipeline-validation/independent-review-20261111'
METHODS = ('eagle3', 'eagle2', 'hass')
PREPARED = 'd3f236a73ab18940bd28f76112210449a2f6ef797061d50a1ab98fc059e0a865'
NUMERICAL_CASES = '782dc425270ff366f2fe64f15f9034a7d403194f49eacb8957f5339a43fa5c9f'
EXPECTED_SOURCE = {'admission_v2.py': 'befe4958005a00df7512e2e238c81d80b8e51bf6ba4ce124d09b45c16e5fcd0b', 'numerical_validation.py': 'cb0546c268c09d73e15f62bde6d55a9a3437d26cce24ef464c14a9138d84629a', 'native_training.py': '2aad19f7aefddfb70dd02b78d2eaf9709c90834d2308d055a836be83de5fec51'}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def partition(token, salt):
    return hashlib.sha256(canonical([20261109, salt, token])).digest()[-1] & 1


def journal(directory, phase, inventory):
    paths = sorted(directory.glob(phase + '-*.jsonl'))
    require(len(paths) == 24, phase + ': missing journals')
    for path in paths:
        receipt_path = path.with_suffix('.receipt.json')
        receipt = read(receipt_path)
        raw_hash = digest(path)
        require(receipt['status'] == 'complete' and receipt['raw_sha256'] == raw_hash, phase + ': receipt/raw mismatch')
        count = 0
        with path.open() as stream:
            for line in stream:
                count += 1
                yield json.loads(line)
        require(receipt['completed'] == count and digest(path) == raw_hash, phase + ': journal count/content changed')
        inventory.append({'path': str(path), 'sha256': raw_hash, 'receipt_sha256': digest(receipt_path)})


def events(tokens, stops):
    result = [partition(tokens[position - 1] if len(tokens) >= position else 'END', salt) for position in (2, 8, 16, 32) for salt in (0, 1)]
    result.extend(int(any(token in stops for token in tokens[:position])) for position in (8, 16, 32))
    grams = [tuple(tokens[index:index + 4]) for index in range(max(0, len(tokens) - 3))]
    return result + [int(len(grams) > len(set(grams)))]


def numerical_case(captures, case):
    pair, probe = case['pair'], case['divergence']
    expected = torch.tensor(pair['target_only']['input_token_ids'] + pair['target_only']['output_token_ids'][:probe])
    require(set(captures) == {route + '-' + str(repeat) for route in ('target_only', 'speculative', 'clean') for repeat in range(2)}, 'Numerical capture coverage')
    vectors, winners = {}, {}
    for key, capture in captures.items():
        route = key.rsplit('-', 1)[0]
        vector = capture['logits']
        require(vector.dtype == torch.float32 and vector.shape == (128256,) and bool(torch.isfinite(vector).all()), 'Numerical vector dimensions/values')
        require(capture['prefix'].dtype == torch.long and torch.equal(capture['prefix'], expected), 'Numerical prefix')
        arena = capture['cache_before']
        lengths = arena['logical_lengths']
        require(arena['shapes'] == [[64, 1, 8, 2048, 128]] and arena['dtypes'] == ['torch.float32'] and len(arena['pointers']) == 1, 'Numerical cache geometry')
        require(lengths.shape == (64,) and lengths.dtype == torch.long and bool((lengths == lengths[0]).all()), 'Numerical cache lengths')
        winners[key] = int(vector.argmax())
        vectors[key] = vector
        if route == 'clean':
            after = capture['cache_after_lengths']
            require(bool((lengths == 0).all()) and after.shape == (64,) and after.dtype == torch.long and bool((after == len(expected)).all()), 'Clean cache reset/full prefix')
            require(not set(arena['pointers']).intersection(capture['inherited_pointers']) and capture['tree_mask'] is None and capture['position_ids'] is None, 'Clean cache inherited state')
        elif route == 'target_only':
            require(int(lengths[0]) == len(expected) - 1 and capture['tree_mask'] is None, 'Incremental AR cache')
            require(capture['emitted_token_id'] == winners[key] == pair[route]['output_token_ids'][probe], 'AR actual decision')
        else:
            prefix, candidates, tree, retrieve = (capture[name] for name in ('tree_prefix', 'candidates', 'tree_candidates', 'retrieve_indices'))
            require(prefix.ndim == 1 and torch.equal(prefix, expected[:len(prefix)]) and int(lengths[0]) == len(prefix), 'Tree committed prefix/cache')
            require(tree.ndim == 2 and tree.shape[0] == 1 and retrieve.shape == candidates.shape and retrieve.dtype == torch.long and bool(((retrieve >= -1) & (retrieve < tree.shape[1])).all()), 'Tree retrieval')
            padded = torch.cat((tree[0], torch.tensor([-1], dtype=torch.long)))
            require(torch.equal(padded[retrieve], candidates), 'Tree candidate retrieval')
            best, offset = capture['best_row'], capture['offset']
            require(0 <= best < len(candidates) and 1 <= offset <= min(capture['accepted'] + 1, candidates.shape[1]), 'Tree probe offset')
            require(torch.equal(torch.cat((prefix, candidates[best, :offset])), expected) and bool((candidates[:, 0] == capture['root_token_id']).all()), 'Tree selected path')
            mask, positions = capture['tree_mask'], capture['position_ids'].reshape(-1)
            nodes, path = tree.shape[1], retrieve[best, :offset]
            require(mask.shape == (1, 1, nodes, nodes) and bool(((mask == 0) | (mask == 1)).all()) and positions.shape == (nodes,), 'Tree mask dimensions')
            require(bool((path >= 0).all()) and torch.equal(positions[path], torch.arange(len(prefix), len(expected))), 'Tree path positions')
            for depth, node in enumerate(path.tolist()):
                visible = set(torch.nonzero(mask[0, 0, node], as_tuple=False).reshape(-1).tolist())
                require(visible == set(path[:depth + 1].tolist()), 'Cross-branch tree visibility')
            require(capture['emitted_token_id'] == winners[key] == pair[route]['output_token_ids'][probe], 'Tree actual decision')
    for route in ('target_only', 'speculative', 'clean'):
        require(torch.equal(vectors[route + '-0'], vectors[route + '-1']), 'Unstable repeated vector')
    maximum = max(float((vectors[a] - vectors[b]).abs().max()) for a, b in itertools.combinations(vectors, 2))
    require(maximum <= 0.0001 and winners['clean-0'] == winners['target_only-0'], 'Numerical epsilon/clean winner')
    known = probe == 200
    if known:
        require(pair['target_only']['input_sha256'] == '189dfe7739c780b54e5b3bff318e66ea42422d6aa94eafd46ea28d093c1aacf9' and winners['target_only-0'] == 8798 and winners['speculative-0'] == 3177, 'Known numerical site')
        for vector in vectors.values():
            values, indices = vector.topk(3)
            require(set(indices[:2].tolist()) == {8798, 3177} and float(values[0] - values[1]) <= .0002 and float(values[1] - values[2]) > .0002, 'Isolated disputed near tie')
    else:
        require(probe == 32 and len(set(winners.values())) == 1, 'Control changed winner')
    return {'known_site': known, 'maximum_pairwise_infinity_delta': maximum, 'winners': winners}


def main():
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU Slurm only')
    torch.set_num_threads(8)
    configs = [read(path) for path in (BUNDLE / 'configs').glob('*.json')]
    configs = [value for value in configs if isinstance(value, dict) and {'prepared', 'primary', 'screen', 'numerical_output'} <= set(value)]
    require(len(configs) == 1, 'Ambiguous frozen replay configuration')
    config = configs[0]
    p = lambda key: ROOT / config[key]
    prepared = read(p('prepared'))
    require(digest(p('prepared')) == PREPARED, 'Registered cases changed')
    for name, expected in EXPECTED_SOURCE.items():
        require(digest(BUNDLE / 'src/taps_rebuild' / name) == expected, 'Final reviewed source differs: ' + name)
    inputs, stops, vocabulary, hashes = {}, {}, {}, {}
    for method in METHODS:
        selected = prepared['methods'][method]
        tokenizer = AutoTokenizer.from_pretrained(selected['models']['target']['path'], local_files_only=True)
        eos = tokenizer.eos_token_id
        stops[method] = {token for token in (list(eos) if isinstance(eos, (list, tuple)) else [eos]) + [tokenizer.convert_tokens_to_ids('<|eot_id|>')] if type(token) is int and token >= 0}
        vocabulary[method] = len(tokenizer)
        for index, prompt in enumerate(selected['sampling_prompts']):
            text = tokenizer.apply_chat_template([{'role': 'system', 'content': selected['source']['system_prompt']['text']}, {'role': 'user', 'content': prompt['text']}], tokenize=False, add_generation_prompt=True)
            inputs[method, index] = tokenizer([text], add_special_tokens=False)['input_ids'][0]
        if vocabulary[method] not in hashes:
            hashes[vocabulary[method]] = torch.tensor([[partition(token, salt) for token in range(vocabulary[method])] for salt in (0, 1)], dtype=torch.bool)
    inventory, conditional, conditional_keys = [], [], set()
    radius = math.sqrt(math.log(2 * 960 / .025) / (2 * 65536))
    for row in journal(p('primary'), 'conditional', inventory):
        method, index, cycle = row['method'], row['prompt_index'], row['cycle']
        key = (method, index, cycle)
        require(key not in conditional_keys and method in METHODS and 0 <= index < 10 and cycle in (1, 3), 'Conditional coverage')
        conditional_keys.add(key)
        require(row['provenance'] == prepared['methods'][method]['provenance'] and row['prepared_sha256'] == PREPARED and row['prompt'] == prepared['methods'][method]['sampling_prompts'][index], 'Conditional provenance/prompt')
        path = p('primary') / row['capture']
        require(path.parent.resolve() == p('primary').resolve() and digest(path) == row['capture_sha256'], 'Conditional capture hash')
        capture = torch.load(path, map_location='cpu', weights_only=True)
        q, weights, candidates, prefix = (capture[name] for name in ('normalized_ar_q', 'native_ar_weights', 'candidates', 'prefix'))
        size = vocabulary[method]
        require(q.shape == weights.shape == (size,) and bool(torch.isfinite(q).all()) and bool((q >= 0).all()) and abs(float(q.sum()) - 1) < 1e-10, 'Conditional reference law')
        require(bool(torch.isfinite(weights).all()) and bool((weights >= 0).all()) and float(weights.double().sum()) > 0 and torch.allclose(q, weights.double() / weights.double().sum(), rtol=5e-14, atol=1e-18), 'Independent saved AR normalization')
        require(prefix[0, :len(inputs[key[:2]])].tolist() == inputs[key[:2]] and int(prefix[0, -1]) == int(candidates[0, 0]) and bool((candidates[:, 0] == candidates[0, 0]).all()), 'Conditional prompt/root identity')
        masks = torch.zeros((8, size), dtype=torch.bool)
        order = torch.argsort(q, descending=True, stable=True)
        for event, count in enumerate((1, 8, 32, 128)):
            masks[event, order[:count]] = True
        masks[4, sorted(stops[method])] = True
        masks[5, sorted({int(token) for token in candidates[:, 1] if 0 <= int(token) < size})] = True
        masks[6:] = hashes[size]
        require(torch.equal(masks, capture['event_masks']), 'Conditional event definitions')
        probabilities = (masks.double() * q).sum(dim=1)
        for sampler in ('posterior', 'target_only'):
            histogram = row['histograms'][sampler]
            require(row['trials_per_sampler'] == 65536 and len(histogram) == size and all(type(value) is int and value >= 0 for value in histogram) and sum(histogram) == 65536, 'Conditional sample budget')
            counts = (masks.long() * torch.tensor(histogram)).sum(dim=1)
            for event in range(8):
                bound = abs(int(counts[event]) / 65536 - float(probabilities[event])) + radius
                require(bound <= .02, 'Registered conditional event failed')
                summary = row['checks'][sampler][event]
                require(summary['count'] == int(counts[event]) and abs(summary['expected'] - float(probabilities[event])) < 1e-12 and abs(summary['simultaneous_error_upper_bound'] - bound) < 1e-12, 'Conditional summary reconstruction')
                conditional.append(bound)
        inventory.append({'path': str(path), 'sha256': row['capture_sha256']})
    require(len(conditional_keys) == 60 and len(conditional) == 960, 'Incomplete conditional population')
    sampling_keys, totals = set(), {(method, mode): [0] * 12 for method in METHODS for mode in ('target_only', 'speculative')}
    for row in journal(p('screen'), 'sampling', inventory):
        method, index, repeat, mode = (row[name] for name in ('method', 'prompt_index', 'repeat', 'mode'))
        key = method, index, repeat, mode
        require(key not in sampling_keys and method in METHODS and 0 <= index < 10 and 0 <= repeat < 1024 and mode in ('target_only', 'speculative'), 'Sampling coverage')
        sampling_keys.add(key)
        seed = 100000000 + METHODS.index(method) * 1000000 + ('target_only', 'speculative').index(mode) * 100000 + index * 1024 + repeat
        require(row['seed'] == seed and row['input_token_ids'] == inputs[method, index] and row['input_sha256'] == identity(inputs[method, index]) and row['provenance'] == prepared['methods'][method]['provenance'], 'Sampling seed/input/provenance')
        raw = row['output_token_ids']
        require(all(type(token) is int and 0 <= token < vocabulary[method] for token in raw), 'Sampling token IDs')
        terminal = next((offset + 1 for offset, token in enumerate(raw) if token in stops[method]), None)
        require(len(raw) >= 32 or terminal is not None, 'Short unterminated sample')
        tokens = raw[:min(32, terminal if terminal is not None else len(raw))]
        values = events(tokens, stops[method])
        require(tokens == row['comparison_token_ids'] and values == row['events'], 'Sampling events/termination')
        totals[method, mode] = [a + b for a, b in zip(totals[method, mode], values)]
    require(len(sampling_keys) == 61440, 'Incomplete sampling population')
    sampling_bounds = [abs(a - b) / 10240 + math.sqrt(math.log(2 * 36 / .025) / 10240) for method in METHODS for a, b in zip(totals[method, 'target_only'], totals[method, 'speculative'])]
    require(max(sampling_bounds) <= .06, 'Registered end-to-end event failed')
    greedy_keys, failures = set(), []
    registered_greedy = {(method, identity(case)) for method in METHODS for case in prepared['methods'][method]['greedy_cases']}
    stopping_manifest = read(p('stopping_output') / 'manifest.json')
    for row in journal(p('stopping_output'), 'greedy', inventory):
        method, case = row['method'], row['case']
        key = method, identity(case)
        require(key in registered_greedy and key not in greedy_keys and row['precision'] == 'float32' and row['provenance'] == prepared['methods'][method]['provenance'], 'Greedy case/provenance')
        greedy_keys.add(key)
        projected = {}
        for mode in ('target_only', 'speculative'):
            output = row['outputs'][mode]
            reserved = prepared['methods'][method]['config']['total_token'] - 1
            budget = min(512, 2048 - reserved - 10 - len(case['input_token_ids']))
            require(budget >= 1 and output['fixed_output_budget'] == budget and output['reserved_tree_tokens'] == reserved and output['seed'] == case['seed'], 'Predetermined greedy budget/seed')
            raw = output['output_token_ids']
            terminal = next((offset + 1 for offset, token in enumerate(raw) if token in stopping_manifest['stop_ids'][method]), None)
            require(len(raw) >= budget or terminal is not None, 'Short greedy output')
            projected[mode] = raw[:min(budget, terminal if terminal is not None else len(raw))]
            require(projected[mode] == output['comparison_token_ids'], 'Greedy projection')
        equal = projected['target_only'] == projected['speculative']
        require(row['equal'] == equal, 'Greedy exact equality metadata')
        if not equal:
            offset = next((i for i, (a, b) in enumerate(zip(projected['target_only'], projected['speculative'])) if a != b), None)
            failures.append({'method': method, 'benchmark': case['benchmark'], 'question_id': case['question_id'], 'turn_id': case['turn_id'], 'first_difference': offset})
    require(greedy_keys == registered_greedy and failures == [{'method': 'eagle3', 'benchmark': 'mt_bench', 'question_id': 146, 'turn_id': 1, 'first_difference': 200}], 'Unclassified greedy discrepancy')
    numerical_prepared = p('numerical_prepared')
    require(digest(numerical_prepared / 'cases.json') == NUMERICAL_CASES, 'Numerical selected cases changed')
    cases = read(numerical_prepared / 'cases.json')['eagle3']['cases']
    numeric = []
    for index, case in enumerate(cases):
        worker_path = p('numerical_output') / ('worker-' + str(index) + '.json')
        row = read(worker_path)
        path = p('numerical_output') / row['capture']
        require(row['case_index'] == row['worker'] == index and row['cases_sha256'] == NUMERICAL_CASES and row['provenance'] == prepared['methods']['eagle3']['provenance'] and digest(path) == row['capture_sha256'], 'Numerical worker/capture identity')
        for mode in ('target_only', 'speculative'):
            outputs = row['outputs'][mode]
            require(len(outputs) == 2 and outputs[0] == outputs[1] and outputs[0][:case['divergence'] + 1] == case['pair'][mode]['output_token_ids'][:case['divergence'] + 1], 'Numerical native output reproduction')
        numeric.append({'case_index': index, **numerical_case(torch.load(path, map_location='cpu', weights_only=True), case)})
        inventory.append({'path': str(path), 'sha256': row['capture_sha256'], 'worker_sha256': digest(worker_path)})
    require(len(numeric) == 8 and sum(item['known_site'] for item in numeric) == 1, 'Numerical population')
    report = {'status': 'passed', 'review_kind': 'independent_raw_reconstruction', 'scientific_training_authorized': False, 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(), 'uid': os.getuid(), 'script_sha256': digest(__file__), 'source_identities': EXPECTED_SOURCE, 'conditional_states': 60, 'conditional_checks': 960, 'conditional_draws': 7864320, 'conditional_maximum_event_bound': max(conditional), 'sampling_continuations': 61440, 'sampling_checks': 36, 'sampling_maximum_event_bound': max(sampling_bounds), 'greedy_pairs': len(greedy_keys), 'greedy_exact_pairs': len(greedy_keys) - len(failures), 'greedy_failures': failures, 'numerical_cases': numeric, 'artifacts': inventory, 'scope': 'Independent retained raw/tensor reconstruction only; not a new GPU collection, model-content audit, paper-agreement result or scientific admission'}
    report['identity_sha256'] = identity(report)
    path = DESTINATION / 'independent-raw-replay.json'
    with path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('artifacts', 'numerical_cases')}, indent=2))


if __name__ == '__main__':
    main()
