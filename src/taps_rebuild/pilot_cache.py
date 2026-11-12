import argparse
import ast
from collections import Counter
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3

from . import native_training as native, pool


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def rank_for(example_id, world=8):
    return int(pool.identity(['pilot-rank-v1', example_id])[:16], 16) % world


def safe_positions(mask, length, margin=6):
    return [index for index, value in enumerate(mask) if value and margin <= index and index + 1 < length]


def quantile_positions(eligible, quota, seed, example_id, occurrence):
    require(len(eligible) >= quota > 0, 'Insufficient effective contexts for quota')
    positions = []
    for index in range(quota):
        numerator = int(pool.identity(['pilot-position-v1', seed, example_id, occurrence, index])[:16], 16)
        lower = index * len(eligible) // quota
        upper = (index + 1) * len(eligible) // quota
        position = lower + numerator * (upper - lower) // 2 ** 64
        positions.append(eligible[position])
    require(len(set(positions)) == quota and positions == sorted(positions), 'Stratified contexts are not distinct')
    return positions


class CaptureDataset:
    column_names = ['id', 'conversations']

    def __init__(self):
        self.function = None

    def __getitem__(self, key):
        require(key == 'train', 'Native dataset split changed')
        return self

    def shuffle(self, seed):
        require(seed == 42, 'Native preprocessing shuffle changed')
        return self

    def map(self, function, **options):
        require(self.function is None and options['batched'] is True and options['num_proc'] == 8 and options['remove_columns'] == self.column_names and options['load_from_cache_file'] is False, 'Native preprocessing map contract changed')
        self.function = function
        return self

    def set_format(self, **options):
        require(options == {'type': 'torch'}, 'Native dataset format changed')


def native_builder(source, torch, loader):
    require(pool.digest(source) == native.UPSTREAM['main.py'], 'Pinned native preprocessing source changed')
    tree = ast.parse(Path(source).read_bytes())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_dataset_rank']
    require(len(functions) == 1, 'Native preprocessing definition is ambiguous')
    namespace = {'torch': torch, 'load_dataset': loader, 'train_config': {'max_len': 2048}, '__name__': '__native_dataset_audit__'}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['build_dataset_rank']


def preprocessor(source, tokenizer, torch):
    capture = CaptureDataset()
    builder = native_builder(source, torch, lambda *args, **options: capture)
    require(builder(tokenizer, 'capture-only') is capture and capture.function is not None, 'Native preprocessing capture failed')
    return capture.function


def encode(function, row, torch):
    item = native.native_row(row, canonical_messages=True, split=row['split'])
    value = function({key: [data] for key, data in item.items()})
    count = len(value['input_ids'])
    require(count in (0, 1) and all(len(value[key]) == count for key in ('loss_mask', 'attention_mask')), 'Native single-example output coverage differs')
    if not count:
        return None
    ids, mask, attention = (value[key][0] for key in ('input_ids', 'loss_mask', 'attention_mask'))
    require(ids.ndim == 2 and ids.shape[0] == 1 and ids.shape == mask.shape == attention.shape and 0 < ids.shape[1] <= 2048, 'Native cached sequence geometry differs')
    require(bool(((mask == 0) | (mask == 1)).all()) and bool((attention == 1).all()) and int(mask.sum()) > 0, 'Native cached mask differs')
    return ids[0], mask[0]


def verified_corpus(root, descriptor, target_identity):
    original, generated = (root / descriptor[key] for key in ('original', 'generated'))
    source_path, target_path = original / 'completion.json', generated / 'completion.json'
    source, target = read(source_path), read(target_path)
    require(pool.digest(source_path) == descriptor['original_completion_sha256'] and source['status'] == 'complete' and source['scope'] == 'real_training_corpus_preparation', 'Original pilot corpus identity differs')
    require(target['status'] == 'complete' and target['scope'] == 'full_training_data_preparation' and target['plan_identity_sha256'] == descriptor['teacher_plan_identity_sha256'] and target['target_identity_sha256'] == target_identity and target['response_regime'] == 'target_generated', 'Generated pilot corpus identity differs')
    generation_plan = read(target['plan_path'])
    require(generation_plan['config_sha256'] == source['config_sha256'] and generation_plan['corpus_completion_sha256'] == pool.digest(source_path), 'Original/generated corpus binding differs')
    proofs = []
    for directory, receipt in ((original, source), (generated, target)):
        for split in ('train', 'dev', 'test'):
            path = directory / (split + '.jsonl')
            require(pool.digest(path) == receipt['files'][path.name]['sha256'], 'Canonical pilot corpus bytes changed')
            proofs.append({'path': str(path), 'sha256': receipt['files'][path.name]['sha256']})
    return original, generated, proofs + [{'path': str(source_path), 'sha256': pool.digest(source_path)}, {'path': str(target_path), 'sha256': pool.digest(target_path)}]


def prepare(root, config_path, output):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Pilot pair cache requires CPU Slurm')
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root / 'data') and not output.exists(), 'Choose a fresh pilot cache under project data/')
    config = read(config_path)
    require(config['world_size'] == 8 and config['length'] == 2048 and config['minimum_raw_contexts'] == 32, 'Pilot cache contract changed')
    torch.set_num_threads(1)
    assets = read(root / 'outputs/preparation/assets.json')
    target = native.model_record(assets, config['target_id'], verify_content=True)
    ancestor = native.model_record(assets, config['ancestor_id'], verify_content=True)
    tokenizer = AutoTokenizer.from_pretrained(target['path'], local_files_only=True)
    pad = tokenizer.convert_tokens_to_ids('<|eot_id|>')
    require(type(pad) is int and 0 <= pad < len(tokenizer), 'Native suffix-pad token is invalid')
    source = root / 'repos/EAGLE/eagle/traineagle3/main.py'
    function = preprocessor(source, tokenizer, torch)
    output.mkdir(parents=True)
    counts, proofs, fixtures, evaluation = Counter(), [], [], {}
    fixture_counts = Counter()
    streams = [{key: (output / ('rank-' + str(rank) + '.' + key)).open('xb') for key in ('ids', 'masks', 'jsonl')} for rank in range(8)]
    sizes = [0] * 8
    domains = {}
    try:
        with closing(sqlite3.connect(output / 'pairs.sqlite')) as database:
            database.execute('CREATE TABLE original (id TEXT PRIMARY KEY, domain TEXT, split TEXT, family TEXT, rep TEXT, priority TEXT, payload TEXT)')
            database.execute('CREATE TABLE targets (id TEXT PRIMARY KEY, payload TEXT)')
            for descriptor in config['corpora']:
                original, generated, bound = verified_corpus(root, descriptor, target['model_identity_sha256'])
                proofs.extend(bound)
                for split in ('train', 'dev', 'test'):
                    for row in records(original / (split + '.jsonl')):
                        domain = descriptor.get('domain') or row['domain']
                        require(domain in ('chat', 'math', 'code') and row['split'] == split, 'Pilot domain/split differs')
                        family = row['group_id']
                        require(family not in domains or domains[family] == (domain, split), 'A pilot conversation family crossed domains or splits')
                        domains[family] = domain, split
                        if split == 'test':
                            priority = pool.identity([config['selection_seed'], 'evaluation', family, row['example_id']])
                            previous = evaluation.get((domain, family))
                            if previous is None or priority < previous[0]:
                                evaluation[domain, family] = priority, row
                        if not row['original_response_complete']:
                            counts[domain + ':incomplete_original'] += 1
                            continue
                        identity = row['example_id']
                        database.execute('INSERT INTO original VALUES (?,?,?,?,?,?,?)', (identity, domain, split, family, pool.identity([config['selection_seed'], 'representative', identity]), pool.identity([config['selection_seed'], 'candidate', family]), json.dumps(row)))
                    database.commit()
                for split in ('train', 'dev', 'test'):
                    for row in records(generated / (split + '.jsonl')):
                        database.execute('INSERT INTO targets VALUES (?,?)', (row['example_id'], json.dumps(row)))
                    database.commit()
            selected = 'SELECT id,domain,split,family,priority,payload FROM (SELECT *,ROW_NUMBER() OVER(PARTITION BY domain,family ORDER BY rep,id) AS representative FROM original) WHERE representative=1'
            for domain in ('chat', 'math', 'code'):
                accepted = 0
                candidates = database.execute('SELECT p.id,p.split,p.family,p.payload,t.payload FROM (' + selected + ') AS p LEFT JOIN targets t ON p.id=t.id WHERE p.domain=? ORDER BY p.priority,p.id LIMIT ?', (domain, config['candidate_groups'][domain]))
                for identity, split, family, original_json, generated_json in candidates:
                    counts[domain + ':candidates'] += 1
                    if generated_json is None:
                        counts[domain + ':no_target_completion'] += 1
                        continue
                    original, generated = json.loads(original_json), json.loads(generated_json)
                    mutable = {'messages', 'response_regime', 'original_response_complete'}
                    require({k: v for k, v in original.items() if k not in mutable} == {k: v for k, v in generated.items() if k not in mutable} and [x for x in original['messages'] if x['role'] == 'user'] == [x for x in generated['messages'] if x['role'] == 'user'], 'Paired source fields or user turns differ')
                    encoded = []
                    for policy, row in (('original', original), ('target', generated)):
                        value = encode(function, row, torch)
                        if value is None:
                            counts[domain + ':' + policy + ':native_length_excluded'] += 1
                            break
                        ids, mask = value
                        if len(safe_positions(mask.tolist(), len(ids))) < config['minimum_raw_contexts']:
                            counts[domain + ':' + policy + ':raw_contexts_excluded'] += 1
                            break
                        encoded.append(value)
                    if len(encoded) != 2:
                        continue
                    rank = rank_for(identity)
                    ids = np.full((2, 2048), pad, dtype='<i4')
                    masks = np.zeros((2, 2048), dtype='u1')
                    lengths = []
                    for index, (tokens, mask) in enumerate(encoded):
                        length = len(tokens)
                        ids[index, :length] = tokens.numpy()
                        masks[index, :length] = mask.numpy()
                        lengths.append(length)
                    metadata = {'example_id': identity, 'domain': domain, 'split': split, 'group_id': family, 'rank': rank, 'index': sizes[rank], 'lengths': lengths}
                    streams[rank]['ids'].write(ids.tobytes())
                    streams[rank]['masks'].write(masks.tobytes())
                    streams[rank]['jsonl'].write(native.canonical(metadata) + b'\n')
                    sizes[rank] += 1
                    counts[domain + ':paired_native_eligible'] += 1
                    counts[domain + ':' + split] += 1
                    if fixture_counts[domain] < 8:
                        for policy, row, (tokens, mask) in zip(('original', 'target'), (original, generated), encoded):
                            item = native.native_row(row, canonical_messages=True, split=split)
                            item['id'] = identity + ':' + policy
                            fixtures.append((item, tokens.clone(), mask.clone()))
                        fixture_counts[domain] += 1
                    accepted += 1
                    if accepted >= config['maximum_cached_groups_per_domain']:
                        break
        for per_rank in streams:
            for stream in per_rank.values():
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        for per_rank in streams:
            for stream in per_rank.values():
                stream.close()
    require(all(sizes) and all(counts[domain + ':train'] > 0 for domain in ('chat', 'math', 'code')), 'Pilot pair population is empty')
    fixture_path = output / 'native-parity.jsonl'
    with fixture_path.open('xb') as stream:
        for item, _, _ in fixtures:
            stream.write(native.canonical(item) + b'\n')
    order = load_dataset('json', data_files=str(fixture_path))['train'].shuffle(seed=42)['id']
    exact = native_builder(source, torch, load_dataset)(tokenizer, str(fixture_path))
    expected = {item['id']: (tokens, mask) for item, tokens, mask in fixtures}
    require(len(exact) == len(fixtures), 'Native cache parity coverage differs')
    for identity, value in zip(order, exact):
        tokens, mask = expected[identity]
        require(torch.equal(value['input_ids'][0], tokens) and torch.equal(value['loss_mask'][0], mask) and bool((value['attention_mask'] == 1).all()), 'Captured preprocessing differs from actual pinned dataset builder')
    evaluation_counts = Counter()
    with (output / 'evaluation.jsonl').open('xb') as stream:
        for domain in ('chat', 'math', 'code'):
            choices = sorted((value for (kind, _), value in evaluation.items() if kind == domain), key=lambda value: value[0])
            for _, row in choices:
                messages = [{'role': 'system', 'content': native.system_prompt(source)}, {'role': 'user', 'content': row['messages'][0]['content']}]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                if len(tokens) > config['evaluation_prefix_limit']:
                    continue
                stream.write(native.canonical({'domain': domain, 'example_id': row['example_id'], 'group_id': row['group_id'], 'input_token_ids': tokens, 'text': row['messages'][0]['content']}) + b'\n')
                evaluation_counts[domain] += 1
                if evaluation_counts[domain] == config['evaluation_questions_per_domain']:
                    break
    require(all(evaluation_counts[domain] == config['evaluation_questions_per_domain'] for domain in ('chat', 'math', 'code')), 'Insufficient held-out pilot inference prompts')
    require(all(pool.digest(item['path']) == item['sha256'] for item in proofs), 'Pilot input changed during cache preparation')
    files = {p.name: {'sha256': pool.digest(p), 'size_bytes': p.stat().st_size} for p in output.iterdir() if p.is_file() and p.name != 'pairs.sqlite'}
    report = {'status': 'complete_gate_scan_required', 'scope': 'Paired native token cache; no training admission', 'directory': str(output), 'config_path': str(Path(config_path).resolve()), 'config_sha256': pool.digest(config_path), 'adapter_sha256': pool.digest(__file__), 'native_helper_sha256': pool.digest(native.__file__), 'upstream_main_sha256': native.UPSTREAM['main.py'], 'target': target, 'ancestor': ancestor, 'world_size': 8, 'length': 2048, 'pad_token_id': pad, 'rank_examples': sizes, 'counts': dict(counts), 'evaluation_counts': dict(evaluation_counts), 'native_preprocessing_parity_cases': len(fixtures), 'files': files, 'inputs': proofs, 'mask_quota_selected': False, 'teacher_hidden_states_cached': False, 'pilot_training_admitted': False}
    report['identity_sha256'] = pool.identity(report)
    pool.atomic_json(output / 'completion.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.config, args.output), indent=2))


if __name__ == '__main__':
    main()
