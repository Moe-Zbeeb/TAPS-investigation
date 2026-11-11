from itertools import combinations
from pathlib import Path

from .admission import digest, identity, read, require


EPSILON = 0.0001
DISPUTED = {8798, 3177}
INPUT_SHA256 = '189dfe7739c780b54e5b3bff318e66ea42422d6aa94eafd46ea28d093c1aacf9'
CASES_SHA256 = '782dc425270ff366f2fe64f15f9034a7d403194f49eacb8957f5339a43fa5c9f'


def vector_checks(captures, case, vocabulary):
    import torch

    pair, probe = case['pair'], case['divergence']
    prompt = pair['target_only']['input_token_ids']
    expected = torch.tensor(prompt + pair['target_only']['output_token_ids'][:probe])
    keys = {route + '-' + str(repeat) for route in ('target_only', 'speculative', 'clean') for repeat in range(2)}
    require(set(captures) == keys, 'Incomplete retained vector population')
    winners, vectors = {}, {}
    for key, capture in captures.items():
        vector = capture['logits']
        require(vector.shape == (vocabulary,) and vector.dtype == torch.float32 and bool(torch.isfinite(vector).all()), 'Invalid full FP32 vocabulary vector')
        require(torch.equal(capture['prefix'], expected), 'Numerical probe prefix differs')
        route = key.rsplit('-', 1)[0]
        winner = int(vector.argmax())
        winners[key], vectors[key] = winner, vector
        cache = capture['cache_before']
        require(cache['pointers'] and len(cache['pointers']) == len(cache['shapes']) == len(cache['dtypes']) and all(value == 'torch.float32' for value in cache['dtypes']), 'Missing FP32 cache identity')
        lengths = cache['logical_lengths']
        require(lengths.ndim == 1 and lengths.numel() == 64 and bool((lengths == lengths[0]).all()), 'Inconsistent logical cache lengths')
        if route == 'clean':
            require(bool((lengths == 0).all()) and bool((capture['cache_after_lengths'] == len(expected)).all()), 'Clean reference did not use an empty cache and complete prefix')
            require(not set(cache['pointers']) & set(capture['inherited_pointers']) and capture['tree_mask'] is None and capture['position_ids'] is None, 'Clean reference reused speculative state')
        elif route == 'target_only':
            require(capture['emitted_token_id'] == winner == pair[route]['output_token_ids'][probe] and int(lengths[0]) == len(expected) - 1 and capture['tree_mask'] is None, 'Incremental AR probe identity or selection differs')
        else:
            require(capture['emitted_token_id'] == winner == pair[route]['output_token_ids'][probe], 'Native tree output is not its captured argmax')
            prefix, candidates = capture['tree_prefix'], capture['candidates']
            tree, retrieve = capture['tree_candidates'], capture['retrieve_indices']
            require(prefix.ndim == 1 and torch.equal(prefix, expected[:len(prefix)]) and int(lengths[0]) == len(prefix), 'Tree prefix/cache identity differs')
            require(tree.ndim == 2 and tree.shape[0] == 1 and retrieve.ndim == 2 and candidates.shape == retrieve.shape and bool((retrieve >= -1).all()) and bool((retrieve < tree.shape[1]).all()), 'Invalid tree retrieval dimensions')
            padded = torch.cat((tree[0], torch.tensor([-1], dtype=tree.dtype)))
            require(torch.equal(candidates, padded[retrieve]), 'Candidate paths differ from native retrieval')
            best, offset = capture['best_row'], capture['offset']
            require(0 <= best < len(candidates) and 1 <= offset <= capture['accepted'] + 1 and offset <= candidates.shape[1], 'Invalid selected path offset')
            require(torch.equal(torch.cat((prefix, candidates[best, :offset])), expected) and bool((candidates[:, 0] == capture['root_token_id']).all()), 'Selected tree path/root does not produce the common prefix')
            mask, positions = capture['tree_mask'], capture['position_ids'].reshape(-1)
            nodes = tree.shape[1]
            require(mask.shape == (1, 1, nodes, nodes) and bool(((mask == 0) | (mask == 1)).all()) and positions.shape == (nodes,), 'Tree mask/position dimensions differ')
            path = retrieve[best, :offset]
            require(bool((path >= 0).all()) and torch.equal(positions[path], torch.arange(len(prefix), len(prefix) + offset)), 'Selected path positions differ')
            for depth, node in enumerate(path.tolist()):
                visible = set(torch.nonzero(mask[0, 0, node], as_tuple=False).reshape(-1).tolist())
                require(visible == set(path[:depth + 1].tolist()), 'Selected path mask exposes another tree branch')
    for route in ('target_only', 'speculative', 'clean'):
        require(torch.equal(vectors[route + '-0'], vectors[route + '-1']), 'Repeated full vectors are unstable')
    delta = max(float((vectors[a] - vectors[b]).abs().max()) for a, b in combinations(sorted(vectors), 2))
    require(delta <= EPSILON, 'Full-vector discrepancy exceeds registered epsilon')
    ar, tree, clean = (winners[route + '-0'] for route in ('target_only', 'speculative', 'clean'))
    require(ar == clean, 'Clean full-prefix reference selects another winner')
    failure = pair['target_only']['benchmark'] == 'mt_bench' and pair['target_only']['question_id'] == 146 and pair['target_only']['turn_id'] == 1
    if failure:
        require(probe == 200 and pair['target_only']['input_sha256'] == INPUT_SHA256 and ar == 8798 and tree == 3177, 'Registered numerical mismatch did not reproduce')
        for vector in vectors.values():
            values, indices = vector.topk(3)
            require(set(indices[:2].tolist()) == DISPUTED and float(values[0] - values[1]) <= 2 * EPSILON and float(values[1] - values[2]) > 2 * EPSILON, 'Disputed token pair is not an isolated registered near tie')
    else:
        require(probe == 32 and len(set(winners.values())) == 1, 'A control has a numerical argmax mismatch')
    return {'classified_known_near_tie': failure, 'max_pairwise_full_vector_delta': delta, 'winners': winners, 'top_three': {key: {'ids': vector.topk(3).indices.tolist(), 'logits': vector.topk(3).values.tolist()} for key, vector in vectors.items()}}


def replay(root, prepared, directory, bundle, bind, registered):
    import torch

    prepared, directory, bundle = Path(prepared), Path(directory), Path(bundle)
    manifest = read(directory / 'manifest.json')
    require(manifest['epsilon'] == EPSILON and manifest['retrospective_case_selection'] is True and len(manifest['hardware']['tokens']) == 8, 'Numerical confirmation registration differs')
    require(read(directory / 'completion.json') == {'status': 'complete', 'workers': 8, 'captures': 48, 'training_admission': False}, 'Numerical collection is incomplete')
    require(manifest['cases_sha256'] == digest(prepared / 'cases.json') == CASES_SHA256 and manifest['prepared_manifest_sha256'] == digest(prepared / 'manifest.json'), 'Selected numerical inputs changed')
    for name, expected in manifest['files'].items():
        require(digest(bundle / name) == expected, 'Numerical collector/protocol changed')
        bind(bundle / name, 'numerical_source')
    bind(prepared / 'cases.json', 'numerical_inputs')
    bind(prepared / 'manifest.json', 'numerical_inputs')
    bind(directory / 'manifest.json', 'numerical_manifest')
    bind(directory / 'completion.json', 'numerical_manifest')
    selected = read(prepared / 'cases.json')['eagle3']
    require(len(selected['cases']) == 8, 'Numerical control population differs')
    method = registered['methods']['eagle3']
    require(selected['target_path'] == method['models']['target']['path'] and selected['draft_path'] == method['models']['drafter']['path'] and selected['source_path'] == method['source']['path'] and selected['definition'] == method['definition'] and selected['config'] == method['config'] and selected['run_config'] == method['run_config'] and selected['run_manifest'] == method['run_manifest'], 'Numerical model/source/config differs from registered EAGLE3')
    results = []
    for index, case in enumerate(selected['cases']):
        path = directory / ('worker-' + str(index) + '.json')
        row = read(path)
        require(row['status'] == 'complete' and row['worker'] == row['case_index'] == index and row['probe'] == case['divergence'] and row['cases_sha256'] == manifest['cases_sha256'] and row['script_sha256'] == manifest['files']['driver.py'] and row['protocol_sha256'] == manifest['files']['PREREGISTRATION.md'], 'Numerical worker/source identity differs')
        for mode in ('target_only', 'speculative'):
            outputs = row['outputs'][mode]
            require(len(outputs) == 2 and outputs[0] == outputs[1] and outputs[0][:row['probe'] + 1] == case['pair'][mode]['output_token_ids'][:row['probe'] + 1], 'Native output replay differs from original probe')
        require(row['vocabulary_size'] == 128256 and row['runtime']['dtype'] == 'torch.float32', 'Numerical vocabulary/dtype differs')
        require(row['provenance'] == method['provenance'] and row['cuda_visible_devices'] == manifest['hardware']['device_bindings'][index]['token'] and row['slurm_job_id'] == manifest['hardware']['slurm_job_id'] and row['gpu'] == manifest['hardware']['gpu_inventory'][index]['name'], 'Numerical runtime/hardware/provenance binding differs')
        require(row['runtime']['torch'] and row['runtime']['cuda'] and (index == 0 or row['runtime'] == read(directory / 'worker-0.json')['runtime']), 'Numerical worker runtimes differ')
        capture_path = directory / row['capture']
        require(capture_path.parent.resolve() == directory.resolve() and digest(capture_path) == row['capture_sha256'], 'Retained vectors changed')
        captures = torch.load(capture_path, map_location='cpu', weights_only=True)
        results.append({'worker': index, 'input_sha256': case['pair']['target_only']['input_sha256'], 'probe': case['divergence'], **vector_checks(captures, case, row['vocabulary_size'])})
        bind(path, 'numerical_worker')
        bind(capture_path, 'numerical_vectors')
    require(sum(row['classified_known_near_tie'] for row in results) == 1, 'Numerical classification population differs')
    return {'status': 'passed', 'epsilon': EPSILON, 'results': results, 'scope': 'One source-bound FP32 route-dependent near tie; kernel cause unresolved; divergent suffixes remain unequal', 'manifest_identity': identity(manifest)}
