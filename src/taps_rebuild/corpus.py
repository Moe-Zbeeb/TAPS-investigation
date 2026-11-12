import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import unicodedata
import urllib.request

from . import native_training, pool


class CorpusError(ValueError):
    pass


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def text_id(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def canonical_row(row, source, ordinal, filename, allow_prompt_only=False):
    if not isinstance(row, dict):
        raise CorpusError('malformed_record')
    original_id = row.get('id', row.get('prompt_id', str(ordinal)))
    if not isinstance(original_id, (str, int)) or not str(original_id):
        original_id = filename + ':' + str(ordinal)
    messages = row.get('conversations', row.get('messages'))
    if source.get('format') == 'alpaca':
        instruction, context, answer = row.get('instruction'), row.get('input', ''), row.get('output')
        if not all(isinstance(value, str) for value in (instruction, context, answer)) or not instruction.strip() or not answer.strip():
            raise CorpusError('invalid_instruction_pair')
        prompt = instruction + ('\n\n' + context if context.strip() else '')
        messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]
    if not isinstance(messages, list) or not messages:
        raise CorpusError('missing_conversation')
    converted = []
    for item in messages:
        if not isinstance(item, dict):
            raise CorpusError('malformed_turn')
        role = item.get('from', item.get('role'))
        role = {'human': 'user', 'gpt': 'assistant'}.get(role, role)
        value = item.get('value', item.get('content'))
        if role not in ('user', 'assistant') or not isinstance(value, str) or not value.strip():
            raise CorpusError('invalid_role_or_text')
        if any(token in value for token in native_training.SPECIAL):
            raise CorpusError('embedded_template_control_token')
        converted.append({'role': role, 'content': value})
    dropped = converted[0]['role'] == 'assistant'
    if dropped:
        converted = converted[1:]
    complete = bool(converted) and len(converted) % 2 == 0
    if not converted or (not complete and not allow_prompt_only) or any(item['role'] != ('user', 'assistant')[index % 2] for index, item in enumerate(converted)):
        raise CorpusError('incomplete_or_non_alternating_conversation')
    prompts = [normalized(item['content']) for item in converted if item['role'] == 'user']
    prompt_id = pool.identity(prompts)
    root_id = re.sub(r'_[0-9]+$', '', str(original_id)) if source['id'] == 'sharegpt_v3' else str(original_id)
    domain = source.get('domain', 'general_chat')
    if not isinstance(domain, str) or not re.fullmatch('[a-z][a-z0-9_]*', domain):
        raise CorpusError('invalid_domain')
    provenance = {'id': source['id'], 'repo_id': source['repo_id'], 'revision': source['revision'], 'file': filename, 'row': ordinal, 'original_id': str(original_id)}
    if source.get('format') == 'alpaca' and 'source' in row:
        provenance['upstream_source'] = row['source']
    return {'example_id': pool.identity([source['id'], source['revision'], filename, ordinal, str(original_id)]), 'prompt_id': prompt_id, 'first_prompt_id': text_id(converted[0]['content']), 'source_group_id': source['id'] + ':' + root_id, 'domain': domain, 'messages': converted, 'response_regime': 'original' if complete else 'original_prompt_only', 'original_response_complete': complete, 'initial_assistant_removed': dropped, 'source': provenance}


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            previous = self.parent[value]
            self.parent[value] = root
            value = previous
        return root

    def join(self, first, second):
        first, second = self.find(first), self.find(second)
        self.parent[max(first, second)] = min(first, second)


def split_for(group, specification):
    value = int(pool.identity([specification['seed'], group])[:16], 16) / 2 ** 64
    train, dev = specification['train_fraction'], specification['dev_fraction']
    return 'train' if value < train else 'dev' if value < train + dev else 'test'


def selected_groups(connection, groups, selection):
    if selection is None:
        return None
    if type(selection.get('seed')) is not int or not selection.get('max_groups_by_source') or any(type(value) is not int or value < 1 for value in selection['max_groups_by_source'].values()):
        raise CorpusError('Invalid bounded pilot selection')
    candidates, owners = {}, {}
    for group, encoded in connection.execute('SELECT group_key,value FROM records ORDER BY ordinal'):
        row, group = json.loads(encoded), groups.find(group)
        if len(row['messages']) != 2:
            raise CorpusError('Bounded pilot selection requires complete single-turn instruction pairs')
        source = row['source']['id']
        if group in owners and owners[group] != source:
            raise CorpusError('Bounded pilot selection cannot separate a cross-source conversation family')
        owners[group] = source
        candidates.setdefault(source, set()).add(group)
    if set(selection['max_groups_by_source']) - set(candidates):
        raise CorpusError('Pilot selection refers to an absent source')
    selected = set()
    for source, values in candidates.items():
        limit = selection['max_groups_by_source'].get(source, len(values))
        ordered = sorted(values, key=lambda group: (pool.identity([selection['seed'], source, group]), group))
        selected.update(ordered[:limit])
    return selected


def download(source, descriptor, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != descriptor['size_bytes'] or pool.digest(destination) != descriptor['sha256']:
            raise CorpusError('Existing raw dataset differs from pinned bytes')
        return
    url = 'https://huggingface.co/datasets/' + source['repo_id'] + '/resolve/' + source['revision'] + '/' + descriptor['path'] + '?download=true'
    temporary = destination.with_name(destination.name + '.partial')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url + '&taps_attempt=' + str(attempt), headers={'User-Agent': 'TAPS-corpus-preparation'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
                digest = hashlib.sha256()
                total = 0
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    digest.update(block)
                    total += len(block)
                stream.flush()
                os.fsync(stream.fileno())
            if total != descriptor['size_bytes'] or digest.hexdigest() != descriptor['sha256']:
                raise CorpusError('Downloaded source size or SHA256 differs')
            temporary.replace(destination)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def source_rows(path):
    if path.suffix == '.json':
        rows = json.loads(path.read_bytes())
        if not isinstance(rows, list):
            raise CorpusError('ShareGPT JSON must be an array')
        yield from rows
    elif path.suffix == '.parquet':
        import pyarrow.parquet
        for batch in pyarrow.parquet.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    else:
        raise CorpusError('Unregistered source format')


def benchmark_exclusions(root):
    registry = json.loads((root / 'outputs/preparation/sources.json').read_bytes())
    excluded, proofs = set(), []
    for name, item in registry['benchmarks'].items():
        path = Path(item['path'])
        if pool.digest(path) != item['sha256']:
            raise CorpusError('Benchmark source changed')
        for line in path.read_text().splitlines():
            excluded.update(text_id(text) for text in json.loads(line)['turns'])
        proofs.append({'benchmark': name, 'sha256': item['sha256']})
    return excluded, proofs


def prepare(root, config_path, output):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise CorpusError('Corpus preparation requires a CPU Slurm task with hidden GPUs')
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root / 'data') or output.exists():
        raise CorpusError('Choose a fresh output under project data/')
    config = json.loads(Path(config_path).read_bytes())
    split = config['split']
    if not 0 < split['train_fraction'] < 1 or not 0 < split['dev_fraction'] < 1 or abs(sum(split[name] for name in ('train_fraction', 'dev_fraction', 'test_fraction')) - 1) > 1e-12:
        raise CorpusError('Invalid registered split fractions')
    for source in config['sources']:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', source['repo_id']) or not re.fullmatch(r'[0-9a-f]{40}', source['revision']):
            raise CorpusError('Dataset repository and immutable revision required')
        for descriptor in source['files']:
            relative = Path(descriptor['path'])
            if relative.is_absolute() or '..' in relative.parts or not re.fullmatch(r'[0-9a-f]{64}', descriptor['sha256']) or descriptor['size_bytes'] < 1:
                raise CorpusError('Unsafe or unpinned dataset file')
    excluded, benchmark_proofs = benchmark_exclusions(root)
    anchor_excluded = set()
    for descriptor in config.get('exclusion_corpora', []):
        directory = (root / descriptor['path']).resolve()
        completion_path = directory / 'completion.json'
        if not directory.is_relative_to(root / 'data') or pool.digest(completion_path) != descriptor['completion_sha256']:
            raise CorpusError('Pinned conversation-exclusion corpus changed')
        anchor = json.loads(completion_path.read_bytes())
        if anchor['status'] != 'complete' or anchor['scope'] != 'real_training_corpus_preparation':
            raise CorpusError('Conversation-exclusion corpus is not complete canonical input data')
        files = {}
        for split_name in ('train', 'dev', 'test'):
            path = directory / (split_name + '.jsonl')
            if pool.digest(path) != anchor['files'][path.name]['sha256']:
                raise CorpusError('Pinned conversation-exclusion input changed')
            with path.open() as stream:
                for line in stream:
                    for turn in json.loads(line)['messages']:
                        if turn['role'] == 'user':
                            anchor_excluded.add(text_id(turn['content']))
            files[path.name] = anchor['files'][path.name]
        benchmark_proofs.append({'kind': 'canonical_conversation_exclusion', 'path': str(directory), 'completion_sha256': descriptor['completion_sha256'], 'files': files})
    output.mkdir(parents=True)
    pool.atomic_json(output / 'intent.json', {'config_sha256': pool.digest(config_path), 'source_sha256': pool.digest(__file__), 'status': 'preparing', 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'created_at_utc': datetime.now(timezone.utc).isoformat()})
    groups, seen, counts = Groups(), set(), Counter()
    database = output / 'preparation-index.sqlite'
    connection = sqlite3.connect(database)
    connection.execute('CREATE TABLE records (ordinal INTEGER PRIMARY KEY, group_key TEXT, value TEXT)')
    try:
        with (output / 'excluded.jsonl').open('x') as rejected:
            for source in config['sources']:
                for descriptor in source['files']:
                    filename = descriptor['path']
                    raw = root / 'data/raw' / source['id'] / source['revision'] / filename
                    print(json.dumps({'stage': 'download', 'source': source['id'], 'file': filename}), flush=True)
                    download(source, descriptor, raw)
                    for ordinal, original in enumerate(source_rows(raw)):
                        counts[source['id'] + ':read'] += 1
                        try:
                            row = canonical_row(original, source, ordinal, filename, allow_prompt_only=True)
                            if any(text_id(item['content']) in excluded for item in row['messages'] if item['role'] == 'user'):
                                raise CorpusError('exact_reference_benchmark_overlap')
                            if any(text_id(item['content']) in anchor_excluded for item in row['messages'] if item['role'] == 'user'):
                                raise CorpusError('prior_corpus_prompt_overlap')
                            group = 'source:' + row['source_group_id']
                            groups.join(group, 'prompt:' + row['first_prompt_id'])
                            if row['prompt_id'] in seen:
                                raise CorpusError('duplicate_complete_prompt_sequence')
                            seen.add(row['prompt_id'])
                            connection.execute('INSERT INTO records(group_key,value) VALUES (?,?)', (group, json.dumps(row, ensure_ascii=False)))
                            counts[source['id'] + ':accepted_before_split'] += 1
                        except CorpusError as error:
                            counts['excluded:' + str(error)] += 1
                            rejected.write(json.dumps({'source': source['id'], 'file': filename, 'row': ordinal, 'reason': str(error)}) + '\n')
                        if ordinal % 5000 == 0:
                            connection.commit()
                    connection.commit()
                    print(json.dumps({'stage': 'normalized', 'source': source['id'], 'counts': dict(counts)}), flush=True)
        selection = selected_groups(connection, groups, config.get('selection'))
        with ExitStack() as streams:
            handles = {}
            for split in ('train', 'dev', 'test'):
                directory = output / 'original'
                directory.mkdir(exist_ok=True)
                handles[split] = (streams.enter_context((output / (split + '.jsonl')).open('x')), streams.enter_context((directory / (split + '.jsonl')).open('x')))
            split_groups = {}
            for group, value in connection.execute('SELECT group_key,value FROM records ORDER BY ordinal'):
                group = groups.find(group)
                split = split_for(group, config['split'])
                if group in split_groups and split_groups[group] != split:
                    raise CorpusError('Conversation group crossed splits')
                split_groups[group] = split
                row = json.loads(value)
                if selection is not None and group not in selection:
                    counts['selection_excluded:' + row['source']['id']] += 1
                    continue
                row.update(split=split, group_id=pool.identity(group))
                counts['selected_source:' + row['source']['id']] += 1
                handles[split][0].write(native_training.canonical(row).decode() + '\n')
                if row['original_response_complete']:
                    native = native_training.native_row(row, canonical_messages=True, split=split)
                    handles[split][1].write(native_training.canonical(native).decode() + '\n')
                    counts['native_original:' + split] += 1
                else:
                    counts['target_completion_required:' + split] += 1
                counts['split:' + split] += 1
            for first, second in handles.values():
                for stream in (first, second):
                    stream.flush()
                    os.fsync(stream.fileno())
        if any(counts['split:' + split] == 0 for split in ('train', 'dev', 'test')):
            raise CorpusError('Every registered split must be nonempty')
        files = {str(path.relative_to(output)): {'sha256': pool.digest(path), 'size_bytes': path.stat().st_size} for path in output.rglob('*.jsonl')}
        receipt = {'status': 'complete', 'scope': 'real_training_corpus_preparation', 'config_sha256': pool.digest(config_path), 'source_sha256': pool.digest(__file__), 'sources': config['sources'], 'counts': dict(counts), 'files': files, 'benchmark_exclusion_proofs': benchmark_proofs, 'splits_grouped': True, 'response_regime': 'original_and_prompt_only_inputs', 'native_original_files_contain_complete_pairs_only': True, 'target_regeneration_performed': False, 'native_tokenization_filter_performed': False, 'teacher_features': 'online native training; offline prototype tensors not required', 'training_started': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
        if config.get('selection') is not None:
            receipt['selection'] = config['selection']
        pool.atomic_json(output / 'completion.json', receipt)
    finally:
        connection.close()
    database.unlink()
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.config, args.output), indent=2))


if __name__ == '__main__':
    main()
