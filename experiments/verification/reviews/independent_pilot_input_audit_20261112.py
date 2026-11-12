import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unicodedata


PINS = {
    'template': '48d1e400ad8c4e89cb76e2b457e4633f63417a4b570e69ead0505900cf5bebf4',
    'config': '0ce2fbfa26518c6162bc2ba44c41bfdeb0e1a5df8b6bccbd9a1e5e518db3fae1',
    'bundle': '1a2cc6d99dafb8e430cbff504c25c8ae5c81eeb0c924da29e7d583cd5ab2a8d9',
    'adapter': '3657939301561abc17c2312f197e2f598710a68a07968c1defdcc56cbbe94ffd',
    'corpus': 'a058e587d6cb42477e654a043cf58ba1924594f3aee97e08696f85474773ead2',
    'chat': '42f408593e5d5774bd859e75fc11bc73ce69153d105a6b789b555e1ee743604c',
}
SPECIAL = ('<|begin_of_text|>', '<|end_of_text|>', '<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>')


class AuditFailure(Exception):
    pass


def require(value, code):
    if not value:
        raise AuditFailure(code)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def text_identity(value):
    normalized = ' '.join(unicodedata.normalize('NFKC', value).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def json_lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def emit(stage, **counts):
    print(json.dumps({'stage': stage, **counts}, sort_keys=True), flush=True)


class Audit:
    def __init__(self, root):
        self.root = root
        self.inventory = {}
        self.phase = 'initialization'
        self.counts = Counter()

    def pin(self, path, expected=None, size=None):
        path = path.resolve()
        require(path.is_relative_to(self.root), 'input_path_escapes_project')
        require(path.is_file(), 'input_file_missing')
        actual_size = path.stat().st_size
        actual_sha = sha(path)
        require(expected is None or actual_sha == expected, 'input_sha256_mismatch')
        require(size is None or actual_size == size, 'input_size_mismatch')
        key = str(path.relative_to(self.root))
        value = {'sha256': actual_sha, 'size_bytes': actual_size}
        require(key not in self.inventory or self.inventory[key] == value, 'input_identity_changed')
        self.inventory[key] = value
        return path

    def run(self):
        bundle = self.root / 'outputs/preparation/rq1-pilot-inputs-20261112-v1'
        population = self.root / 'data/corpora/rq1-math-code-original-20261112-v1'
        self.phase = 'frozen_source_and_configuration'
        manifest = json.loads(self.pin(bundle / 'bundle.json', PINS['bundle']).read_bytes())
        require(manifest['scope'] == 'Bounded pilot original inputs and target teacher responses only; no pilot training', 'bundle_scope_changed')
        for name, expected in manifest['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe_bundle_inventory')
            self.pin(bundle / name, expected)
        template = json.loads(self.pin(bundle / 'configs/pilot-inputs.json', PINS['template']).read_bytes())
        config = json.loads(self.pin(bundle / 'derived-corpus-config.json', PINS['config']).read_bytes())
        expected_config = json.loads(json.dumps(template))
        expected_config['sources'][1]['files'][0]['sha256'] = '58d136bb7ea8e4b7c5b6ef08831da905d0abfafeb56f8fa519083089dd6a210e'
        expected_config['preparation_adapter'] = {
            'path': str(bundle / 'src/taps_rebuild/pilot_data.py'),
            'sha256': PINS['adapter'],
            'corpus_sha256': PINS['corpus'],
            'source_template_sha256': PINS['template'],
        }
        require(config == expected_config, 'derived_configuration_not_exact_template_derivation')
        completion = json.loads(self.pin(population / 'completion.json').read_bytes())
        require(completion['status'] == 'complete' and completion['scope'] == 'real_training_corpus_preparation', 'population_not_complete')
        require(completion['config_sha256'] == PINS['config'] and completion['source_sha256'] == PINS['corpus'], 'population_source_configuration_mismatch')
        require(completion['sources'] == config['sources'] and completion['selection'] == config['selection'], 'population_policy_mismatch')
        require(completion['target_regeneration_performed'] is False and completion['native_tokenization_filter_performed'] is False and completion['training_started'] is False, 'population_scope_mismatch')
        expected_files = {name + '.jsonl' for name in ('train', 'dev', 'test', 'excluded')}
        expected_files.update('original/' + name + '.jsonl' for name in ('train', 'dev', 'test'))
        require(set(completion['files']) == expected_files, 'population_file_inventory_mismatch')
        for name, descriptor in completion['files'].items():
            self.pin(population / name, descriptor['sha256'], descriptor['size_bytes'])
        self.phase = 'reference_and_prior_chat_exclusions'
        registry = json.loads(self.pin(self.root / 'outputs/preparation/sources.json').read_bytes())
        reference_proofs = {item['benchmark']: item['sha256'] for item in completion['benchmark_exclusion_proofs'] if 'benchmark' in item}
        require(set(reference_proofs) == {'alpaca', 'cnn_dm', 'gsm8k', 'humaneval', 'mt_bench'}, 'reference_panel_inventory_mismatch')
        benchmark_prompts = set()
        reference_rows = reference_turns = 0
        for name, expected in reference_proofs.items():
            item = registry['benchmarks'][name]
            require(item['sha256'] == expected, 'reference_registry_hash_mismatch')
            path = self.pin(Path(item['path']), expected)
            rows = 0
            for row in json_lines(path):
                rows += 1
                for text in row['turns']:
                    benchmark_prompts.add(text_identity(text))
                    reference_turns += 1
            require(rows == 80 and rows == item['rows'], 'reference_panel_row_count_mismatch')
            reference_rows += rows
        anchor_prompts = set()
        anchor_rows = anchor_turns = 0
        anchor_proofs = [item for item in completion['benchmark_exclusion_proofs'] if item.get('kind') == 'canonical_conversation_exclusion']
        require(len(anchor_proofs) == len(config['exclusion_corpora']) == 1, 'anchor_inventory_mismatch')
        for specification in config['exclusion_corpora']:
            directory = self.root / specification['path']
            require(specification['completion_sha256'] == PINS['chat'], 'anchor_completion_pin_mismatch')
            anchor = json.loads(self.pin(directory / 'completion.json', PINS['chat']).read_bytes())
            require(anchor['status'] == 'complete' and anchor['scope'] == 'real_training_corpus_preparation', 'anchor_not_complete')
            proof = anchor_proofs[0]
            require(proof['path'] == str(directory) and proof['completion_sha256'] == PINS['chat'], 'anchor_proof_binding_mismatch')
            for split in ('train', 'dev', 'test'):
                name = split + '.jsonl'
                descriptor = anchor['files'][name]
                require(proof['files'][name] == descriptor, 'anchor_split_proof_mismatch')
                path = self.pin(directory / name, descriptor['sha256'], descriptor['size_bytes'])
                for row in json_lines(path):
                    anchor_rows += 1
                    for turn in row['messages']:
                        if turn['role'] == 'user':
                            anchor_prompts.add(text_identity(turn['content']))
                            anchor_turns += 1
            emit('prior_chat_scanned', rows=anchor_rows, user_turns=anchor_turns, unique_prompts=len(anchor_prompts))
        self.phase = 'raw_source_reconstruction'
        graph = defaultdict(set)
        accepted = []
        seen = set()
        expected_exclusions = []
        for source in config['sources']:
            require(source['format'] == 'alpaca', 'unexpected_source_schema')
            for descriptor in source['files']:
                filename = descriptor['path']
                path = self.pin(self.root / 'data/raw' / source['id'] / source['revision'] / filename, descriptor['sha256'], descriptor['size_bytes'])
                if 'git_blob_sha1' in descriptor:
                    blob = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0')
                    with path.open('rb') as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                            blob.update(block)
                    require(blob.hexdigest() == descriptor['git_blob_sha1'], 'raw_git_blob_mismatch')
                values = json.loads(path.read_bytes())
                require(isinstance(values, list), 'raw_source_not_array')
                for ordinal, original in enumerate(values):
                    self.counts[source['id'] + ':read'] += 1
                    reason = None
                    if not isinstance(original, dict):
                        reason = 'malformed_record'
                    else:
                        instruction = original.get('instruction')
                        context = original.get('input', '')
                        answer = original.get('output')
                        if not all(isinstance(value, str) for value in (instruction, context, answer)) or not instruction.strip() or not answer.strip():
                            reason = 'invalid_instruction_pair'
                    if reason is None:
                        prompt = instruction + ('\n\n' + context if context.strip() else '')
                        if any(token in text for token in SPECIAL for text in (prompt, answer)):
                            reason = 'embedded_template_control_token'
                        prompt_hash = text_identity(prompt)
                        if reason is None and prompt_hash in benchmark_prompts:
                            reason = 'exact_reference_benchmark_overlap'
                        if reason is None and prompt_hash in anchor_prompts:
                            reason = 'prior_corpus_prompt_overlap'
                    if reason is None:
                        original_id = original.get('id', original.get('prompt_id', str(ordinal)))
                        if not isinstance(original_id, (str, int)) or not str(original_id):
                            original_id = filename + ':' + str(ordinal)
                        source_group = source['id'] + ':' + str(original_id)
                        source_node = 'source:' + source_group
                        prompt_node = 'prompt:' + prompt_hash
                        graph[source_node].add(prompt_node)
                        graph[prompt_node].add(source_node)
                        prompt_id = identity([normalized(prompt)])
                        if prompt_id in seen:
                            reason = 'duplicate_complete_prompt_sequence'
                        else:
                            seen.add(prompt_id)
                            provenance = {'id': source['id'], 'repo_id': source['repo_id'], 'revision': source['revision'], 'file': filename, 'row': ordinal, 'original_id': str(original_id)}
                            if 'source' in original:
                                provenance['upstream_source'] = original['source']
                            row = {
                                'example_id': identity([source['id'], source['revision'], filename, ordinal, str(original_id)]),
                                'prompt_id': prompt_id,
                                'first_prompt_id': prompt_hash,
                                'source_group_id': source_group,
                                'domain': source['domain'],
                                'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}],
                                'response_regime': 'original',
                                'original_response_complete': True,
                                'initial_assistant_removed': False,
                                'source': provenance,
                            }
                            accepted.append((source_node, row))
                            self.counts[source['id'] + ':accepted_before_split'] += 1
                    if reason is not None:
                        self.counts['excluded:' + reason] += 1
                        expected_exclusions.append({'source': source['id'], 'file': filename, 'row': ordinal, 'reason': reason})
                emit('raw_source_reconstructed', source_id=source['id'], rows=len(values))
                del values
        self.phase = 'independent_graph_selection_and_splits'
        component = {}
        component_sizes = []
        for node in graph:
            if node in component:
                continue
            reached = set()
            pending = [node]
            while pending:
                current = pending.pop()
                if current in reached:
                    continue
                reached.add(current)
                pending.extend(graph[current] - reached)
            representative = min(reached)
            component.update((item, representative) for item in reached)
            component_sizes.append(len(reached))
        candidates = defaultdict(set)
        group_source = {}
        for node, row in accepted:
            group = component[node]
            source_id = row['source']['id']
            require(group not in group_source or group_source[group] == source_id, 'cross_source_retained_family')
            group_source[group] = source_id
            candidates[source_id].add(group)
        selected_groups = set()
        for source_id, limit in config['selection']['max_groups_by_source'].items():
            priority = sorted(candidates[source_id], key=lambda group: (identity([config['selection']['seed'], source_id, group]), group))
            selected_groups.update(priority[:limit])
        expected_rows = {name: [] for name in ('train', 'dev', 'test')}
        group_splits = {}
        for node, row in accepted:
            group = component[node]
            if group not in selected_groups:
                self.counts['selection_excluded:' + row['source']['id']] += 1
                continue
            value = Fraction(int(identity([config['split']['seed'], group])[:16], 16), 2 ** 64)
            train = Fraction(str(config['split']['train_fraction']))
            dev = Fraction(str(config['split']['dev_fraction']))
            split = 'train' if value < train else 'dev' if value < train + dev else 'test'
            require(group not in group_splits or group_splits[group] == split, 'expected_group_split_conflict')
            group_splits[group] = split
            row.update(split=split, group_id=identity(group))
            expected_rows[split].append(row)
            self.counts['selected_source:' + row['source']['id']] += 1
            self.counts['split:' + split] += 1
            self.counts['native_original:' + split] += 1
        self.phase = 'retained_population_exact_comparison'
        all_ids, all_prompts, actual_groups = set(), set(), {}
        source_split_counts = defaultdict(Counter)
        for split, expected in expected_rows.items():
            actual_count = 0
            original = iter(json_lines(population / 'original' / (split + '.jsonl')))
            for index, actual in enumerate(json_lines(population / (split + '.jsonl'))):
                require(index < len(expected) and actual == expected[index], 'canonical_raw_provenance_or_selection_mismatch')
                require(actual['example_id'] not in all_ids and actual['prompt_id'] not in all_prompts, 'retained_example_or_prompt_duplicate')
                all_ids.add(actual['example_id'])
                all_prompts.add(actual['prompt_id'])
                group_id = actual['group_id']
                require(group_id not in actual_groups or actual_groups[group_id] == split, 'retained_group_crosses_splits')
                actual_groups[group_id] = split
                prompt = actual['messages'][0]['content']
                require(text_identity(prompt) not in benchmark_prompts and text_identity(prompt) not in anchor_prompts, 'retained_exact_prompt_overlap')
                native_expected = {'id': actual['example_id'], 'conversations': [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': actual['messages'][1]['content']}]}
                require(next(original, None) == native_expected, 'native_original_pair_or_order_mismatch')
                source_split_counts[actual['source']['id']][split] += 1
                actual_count += 1
            require(actual_count == len(expected) and next(original, None) is None, 'retained_split_or_native_count_mismatch')
        actual_exclusions = list(json_lines(population / 'excluded.jsonl'))
        require(actual_exclusions == expected_exclusions, 'exclusion_journal_exact_replay_mismatch')
        require(dict(self.counts) == completion['counts'], 'preparation_receipt_counts_mismatch')
        require(len(all_ids) == 40000 and all(self.counts['selected_source:' + source['id']] == 20000 for source in config['sources']), 'registered_pilot_population_size_mismatch')
        require({split: len(rows) for split, rows in expected_rows.items()} == {'train': 39168, 'dev': 383, 'test': 449}, 'registered_pilot_split_count_mismatch')
        self.phase = 'final_input_immutability_rehash'
        for name, descriptor in self.inventory.items():
            path = self.root / name
            require(path.stat().st_size == descriptor['size_bytes'] and sha(path) == descriptor['sha256'], 'input_changed_during_independent_audit')
        self.phase = 'complete'
        return {
            'status': 'passed',
            'scope': 'actual_pilot_input_population_only',
            'counts': dict(self.counts),
            'selected_rows': len(all_ids),
            'selected_groups': len(actual_groups),
            'source_split_counts': {name: dict(value) for name, value in source_split_counts.items()},
            'reference_panel_rows': reference_rows,
            'reference_panel_user_turns': reference_turns,
            'reference_panel_unique_prompts': len(benchmark_prompts),
            'prior_chat_rows': anchor_rows,
            'prior_chat_user_turns': anchor_turns,
            'prior_chat_unique_prompts': len(anchor_prompts),
            'raw_connected_components': len(component_sizes),
            'maximum_raw_component_nodes': max(component_sizes),
            'exact_benchmark_prompt_overlaps': 0,
            'exact_prior_chat_prompt_overlaps': 0,
            'duplicate_selected_examples_or_prompts': 0,
            'group_split_conflicts': 0,
            'raw_prompt_answer_and_provenance_mismatches': 0,
            'selection_split_or_order_mismatches': 0,
            'native_original_pair_mismatches': 0,
            'exclusion_journal_mismatches': 0,
            'all_receipt_counts_match': True,
            'all_inventory_bytes_stable_before_and_after': True,
            'scientific_training_authorized': False,
            'teacher_outputs_reviewed': False,
            'limitations': [
                'Exact normalized reference-panel/prior-chat exclusion only; no semantic, paraphrase, code-clone or full-benchmark contamination claim.',
                'This is the pre-teacher input population. Teacher EOS completion, runtime/model/settings, generated coverage and postgeneration native length QA require separate evidence.',
                'Original upstream license declarations and network ownership were not independently authenticated.',
                'No matched-budget pilot training or completed post-filter data-scale condition is admitted.',
            ],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    directory = root / 'outputs/pilot-input-audit/20261112-v1'
    directory.mkdir(parents=True, exist_ok=True)
    require(not (directory / 'report.json').exists(), 'independent_report_already_exists')
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'independent_audit_requires_cpu_slurm')
    start = time.monotonic()
    audit = Audit(root)
    try:
        report = audit.run()
    except Exception as error:
        report = {'status': 'failed', 'scope': 'actual_pilot_input_population_only', 'phase': audit.phase, 'error_code': str(error) if isinstance(error, AuditFailure) else type(error).__name__, 'scientific_training_authorized': False}
    report.update(
        audit_script_sha256=sha(Path(__file__)),
        slurm_job_id=os.environ['SLURM_JOB_ID'],
        cpu_only=True,
        imports_project_helpers=False,
        wall_seconds=time.monotonic() - start,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact_inventory=audit.inventory,
        artifact_inventory_sha256=identity(audit.inventory),
    )
    temporary = directory / ('report.json.tmp-' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(directory / 'report.json')
    emit('independent_audit_finished', status=report['status'], report_sha256=sha(directory / 'report.json'), wall_seconds=report['wall_seconds'])
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
