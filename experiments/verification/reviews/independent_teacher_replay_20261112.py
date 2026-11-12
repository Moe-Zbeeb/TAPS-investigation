import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time


SPECIAL = ('<|begin_of_text|>', '<|end_of_text|>', '<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>')
REGISTERED = {
    'dbf6acd428977385aa98b2f302241086764f28eb9e1cda1005af0f45ed3331af': {'name': 'chat', 'examples': 540107, 'original': '42f408593e5d5774bd859e75fc11bc73ce69153d105a6b789b555e1ee743604c', 'generated': '94eeea149a025fdc7d08bfde4d9cb91403c89ed929738f9d8f0e86a424e830c6'},
    '97dde81997dabd002466d6952b202d919bad2aba46aa324b4beececabc975961': {'name': 'math_code', 'examples': 40000, 'original': 'c132befc2a0d4be999d42e771a7163ed3dad9a8b6220d09ff0fc344ce7a45d57', 'generated': '9e96390ec1c075367b09dfcdd7978f29937d6a1ff141abdb74b22bab3bc96187'},
}


class ReplayFailure(Exception):
    pass


def require(value, code):
    if not value:
        raise ReplayFailure(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def row_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(data)
    return result.hexdigest()


def rows(path):
    with path.open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n'), 'uncommitted_or_unterminated_jsonl_record')
            yield json.loads(line)


def validate_messages(messages, complete):
    require(isinstance(messages, list) and messages and (not complete or len(messages) % 2 == 0), 'conversation_geometry_invalid')
    for index, turn in enumerate(messages):
        require(set(turn) == {'role', 'content'} and turn['role'] == ('user', 'assistant')[index % 2] and isinstance(turn['content'], str) and turn['content'].strip() and not any(token in turn['content'] for token in SPECIAL), 'conversation_role_or_control_text_invalid')


def native_row(record):
    validate_messages(record['messages'], True)
    return {'id': str(record['example_id']), 'conversations': [{'from': ('human', 'gpt')[index % 2], 'value': turn['content']} for index, turn in enumerate(record['messages'])]}


class Replay:
    def __init__(self, root):
        self.root = root
        self.inventory = {}
        self.phase = 'initialization'
        self.global_ids = set()

    def pin(self, path, expected=None, size=None):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root) or path.is_relative_to((self.root / 'models').resolve()), 'input_outside_registered_project_storage')
        value = {'sha256': digest(path), 'size_bytes': path.stat().st_size}
        require(expected is None or value['sha256'] == expected, 'input_digest_mismatch')
        require(size is None or value['size_bytes'] == size, 'input_size_mismatch')
        require(str(path) not in self.inventory or self.inventory[str(path)] == value, 'input_changed_during_replay')
        self.inventory[str(path)] = value
        return path

    def stop_ids(self, plan):
        directory = Path(plan['target']['path'])
        files = plan['target']['files']
        values = {}
        for name in ('config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json'):
            require(name in files, 'target_stop_metadata_missing')
            values[name] = json.loads(self.pin(directory / name, files[name]['sha256'], files[name]['size_bytes']).read_bytes())
        require(not values['generation_config.json'].get('stop_strings'), 'unregistered_default_string_stop')
        stops = set()
        for config in (values['config.json'], values['generation_config.json']):
            eos = config.get('eos_token_id')
            if type(eos) is int:
                stops.add(eos)
            elif isinstance(eos, list):
                require(all(type(value) is int for value in eos), 'invalid_eos_configuration')
                stops.update(eos)
        eos = values['tokenizer_config.json']['eos_token']
        eos = eos['content'] if isinstance(eos, dict) else eos
        for item in values['tokenizer.json']['added_tokens']:
            if item['content'] in (eos, '<|eot_id|>'):
                stops.add(item['id'])
        require(stops and all(type(value) is int and 0 <= value < 128256 for value in stops), 'configured_stop_ids_invalid')
        return stops

    def corpus(self, descriptor):
        plan_identity = descriptor['teacher_plan_identity_sha256']
        require(plan_identity in REGISTERED, 'unregistered_teacher_plan')
        registered = REGISTERED[plan_identity]
        self.phase = registered['name'] + '_bindings'
        original, generated = (self.root / descriptor[key] for key in ('original', 'generated'))
        source_path, target_path = original / 'completion.json', generated / 'completion.json'
        require(descriptor['original_completion_sha256'] == registered['original'] and descriptor['generated_completion_sha256'] == registered['generated'], 'descriptor_registration_mismatch')
        source = json.loads(self.pin(source_path, registered['original']).read_bytes())
        target = json.loads(self.pin(target_path, registered['generated']).read_bytes())
        require(source['status'] == target['status'] == 'complete' and source['scope'] == 'real_training_corpus_preparation' and target['scope'] == 'full_training_data_preparation', 'corpus_scope_or_completion_invalid')
        plan_path = self.pin(target['plan_path'])
        plan = json.loads(plan_path.read_bytes())
        copied = dict(plan)
        require(copied.pop('identity_sha256') == identity(copied) == plan_identity == target['plan_identity_sha256'], 'actual_teacher_plan_identity_mismatch')
        require(plan['scope'] == 'full_training_data_preparation' and plan['examples'] == registered['examples'] and plan['world_size'] == 8 and len(plan['shards']) == 8, 'teacher_population_contract_mismatch')
        require(Path(plan['corpus']).resolve() == original.resolve() and plan['corpus_completion_sha256'] == registered['original'] and plan['config_sha256'] == source['config_sha256'] and plan['target']['model_identity_sha256'] == target['target_identity_sha256'], 'teacher_original_target_binding_mismatch')
        require(row_hash(plan['target']['files']) == plan['target']['model_identity_sha256'] and hashlib.sha256(plan['system_prompt'].encode()).hexdigest() == plan['system_prompt_sha256'], 'teacher_model_or_system_prompt_self_binding_mismatch')
        self.pin(self.root / 'repos/EAGLE/eagle/traineagle3/main.py', plan['native_main_sha256'])
        settings = plan['config']['target_generation']
        require(settings['backend'] == 'stock_vllm_target_only' and settings['dtype'] == 'float16' and settings['max_context'] == 4096 and settings['max_new_tokens'] == 2048 and settings['temperature'] == 0 and settings['top_p'] == 1 and settings['batch_conversations_per_worker'] == 64, 'teacher_settings_contract_mismatch')
        require(settings['include_only_complete_eos_terminated_conversations'] is True and target['response_regime'] == 'target_generated', 'teacher_termination_policy_mismatch')
        for name, expected in plan['scripts'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe_teacher_source_filename')
            self.pin(plan_path.parent / 'inputs' / name, expected)
        for item in plan['target']['files'].values():
            require(re.fullmatch('[0-9a-f]{64}', item['sha256']), 'target_inventory_not_sha256_pinned')
        for name, item in plan['target']['files'].items():
            self.pin(Path(plan['target']['path']) / name, item['sha256'], item['size_bytes'])
        stops = self.stop_ids(plan)
        for directory, receipt in ((original, source), (generated, target)):
            for name, item in receipt['files'].items():
                require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe_corpus_file_inventory')
                self.pin(directory / name, item['sha256'], item['size_bytes'])
        originals, families = [], {}
        for split in ('train', 'dev', 'test'):
            for row in rows(original / (split + '.jsonl')):
                require(row['split'] == split and row['example_id'] not in self.global_ids, 'original_split_or_global_id_duplicate')
                validate_messages(row['messages'], row['original_response_complete'])
                require(row['original_response_complete'] == (len(row['messages']) % 2 == 0), 'original_complete_flag_mismatch')
                require(row['response_regime'] == ('original' if row['original_response_complete'] else 'original_prompt_only'), 'original_response_regime_mismatch')
                self.global_ids.add(row['example_id'])
                family = row['group_id']
                require(family not in families or families[family] == split, 'original_family_crosses_splits')
                families[family] = split
                originals.append((row['example_id'], row_hash(row), split))
        require(len(originals) == registered['examples'], 'original_full_population_count_mismatch')
        require(len(plan['input_proofs']) == 3 and {proof['split'] for proof in plan['input_proofs']} == {'train', 'dev', 'test'}, 'teacher_input_split_proofs_incomplete')
        for proof in plan['input_proofs']:
            path = original / (proof['split'] + '.jsonl')
            require(digest(path) == proof['sha256'] == source['files'][path.name]['sha256'], 'teacher_input_proof_mismatch')
        self.phase = registered['name'] + '_raw_worker_replay'
        counts, terminations, slurm_jobs = Counter(), Counter(), Counter()
        completed = {}
        seen = bytearray(len(originals))
        total_tokens = 0
        for rank, shard in enumerate(plan['shards']):
            shard_path = self.pin(shard['path'], shard['sha256'])
            inputs = iter(rows(shard_path))
            receipt_path = self.pin(plan_path.parent / ('worker-' + str(rank) + '-receipt.json'))
            receipt = json.loads(receipt_path.read_bytes())
            require(receipt['status'] == 'complete' and receipt['rank'] == rank and receipt['plan_identity_sha256'] == plan_identity and receipt['examples'] == shard['examples'], 'worker_receipt_coverage_mismatch')
            raw = self.pin(plan_path.parent / ('worker-' + str(rank) + '.jsonl'), receipt['raw_sha256'])
            observed = 0
            for batch in rows(raw):
                require(batch['plan_identity_sha256'] == plan_identity and batch['rank'] == rank and re.fullmatch('[0-9]+', batch['slurm_job_id']) and isinstance(batch['cuda_visible_devices'], str) and batch['cuda_visible_devices'].strip() and ',' not in batch['cuda_visible_devices'], 'worker_batch_plan_rank_or_gpu_metadata_invalid')
                require(0 < len(batch['outcomes']) <= 64 and math.isfinite(batch['wall_seconds']) and batch['wall_seconds'] >= 0, 'worker_batch_geometry_invalid')
                slurm_jobs[batch['slurm_job_id']] += 1
                batch_tokens = 0
                for outcome in batch['outcomes']:
                    item = next(inputs, None)
                    require(item is not None, 'raw_worker_has_extra_outcome')
                    ordinal = item['ordinal']
                    require(type(ordinal) is int and 0 <= ordinal < len(originals) and ordinal % 8 == rank and not seen[ordinal] and outcome['ordinal'] == ordinal, 'raw_ordinal_duplicate_gap_or_wrong_rank')
                    source_row = item['record']
                    expected_id, expected_hash, split = originals[ordinal]
                    require(source_row['example_id'] == outcome['example_id'] == expected_id and row_hash(source_row) == expected_hash and source_row['split'] == split, 'raw_shard_or_example_source_mismatch')
                    seen[ordinal] = 1
                    users = [turn for turn in source_row['messages'] if turn['role'] == 'user']
                    turns = outcome['generation_turns']
                    require(len(turns) <= len(users) and [turn['turn'] for turn in turns] == list(range(len(turns))), 'generation_turn_order_or_coverage_mismatch')
                    for turn in turns:
                        require(type(turn['prompt_tokens']) is int and 0 < turn['prompt_tokens'] < 4096 and type(turn['output_tokens']) is int and 0 <= turn['output_tokens'] <= min(2048, 4096 - turn['prompt_tokens']) and re.fullmatch('[0-9a-f]{64}', turn['prompt_sha256']), 'generation_prompt_or_output_budget_metadata_invalid')
                        batch_tokens += turn['output_tokens']
                        if turn['finish_reason'] == 'stop':
                            require(turn['stop_reason'] is None or type(turn['stop_reason']) is int and turn['stop_reason'] in stops, 'unregistered_generation_stop_reason')
                            terminations['default_eos_metadata' if turn['stop_reason'] is None else 'configured_token_' + str(turn['stop_reason'])] += 1
                        else:
                            terminations['nonstop:' + str(turn['finish_reason'])] += 1
                    counts['processed'] += 1
                    if outcome['status'] == 'complete':
                        record = outcome['record']
                        require(outcome['reason'] is None and len(turns) == len(users) and all(turn['finish_reason'] == 'stop' and turn['output_tokens'] > 0 for turn in turns), 'accepted_outcome_missing_complete_configured_stop_turns')
                        require(record['response_regime'] == 'target_generated' and record['original_response_complete'] is True, 'accepted_response_flags_invalid')
                        mutable = {'messages', 'response_regime', 'original_response_complete'}
                        require({key: value for key, value in record.items() if key not in mutable} == {key: value for key, value in source_row.items() if key not in mutable}, 'accepted_source_fields_changed')
                        validate_messages(record['messages'], True)
                        require([turn for turn in record['messages'] if turn['role'] == 'user'] == users and len(record['messages']) == 2 * len(users), 'accepted_user_turns_changed_or_extra_assistant')
                        completed[expected_id] = [split, row_hash(record), row_hash(native_row(record)), False, False, ordinal]
                        counts['accepted:' + split] += 1
                    else:
                        require(outcome['status'] == 'excluded' and outcome['record'] is None and outcome['reason'] in ('prompt_exceeds_generation_context', 'response_censored_at_generation_limit', 'empty_or_template_control_response'), 'excluded_outcome_reason_or_record_invalid')
                        if outcome['reason'] == 'prompt_exceeds_generation_context':
                            require(len(turns) < len(users), 'context_exclusion_has_all_turns')
                        elif outcome['reason'] == 'response_censored_at_generation_limit':
                            require(turns and turns[-1]['finish_reason'] != 'stop', 'censored_exclusion_has_no_nonstop_turn')
                        else:
                            require(turns and turns[-1]['finish_reason'] == 'stop', 'invalid_response_exclusion_has_no_stopped_turn')
                        counts['excluded:' + outcome['reason']] += 1
                    observed += 1
                require(batch_tokens == batch['generated_tokens'], 'worker_batch_token_accounting_mismatch')
                total_tokens += batch_tokens
            require(observed == shard['examples'] and next(inputs, None) is None, 'worker_shard_outcome_coverage_incomplete')
            print(json.dumps({'stage': 'teacher_worker_replayed', 'population': registered['name'], 'rank': rank, 'outcomes': observed}), flush=True)
        require(all(seen) and counts['processed'] == plan['examples'] and dict(counts) == target['counts'], 'full_teacher_population_or_receipt_counts_mismatch')
        self.phase = registered['name'] + '_canonical_native_merge_replay'
        for split in ('train', 'dev', 'test'):
            canonical_count = native_count = 0
            previous_ordinal = -1
            for row in rows(generated / (split + '.jsonl')):
                identifier = row['example_id']
                require(identifier in completed, 'canonical_row_not_in_accepted_raw_outcomes')
                item = completed[identifier]
                require(item[0] == row['split'] == split and item[1] == row_hash(row) and item[3] is False, 'canonical_raw_record_split_or_duplicate_mismatch')
                require(item[5] > previous_ordinal, 'canonical_merge_order_differs_from_raw_ordinals')
                previous_ordinal = item[5]
                item[3] = True
                canonical_count += 1
            previous_ordinal = -1
            for row in rows(generated / ('native-' + split + '.jsonl')):
                identifier = row['id']
                require(identifier in completed, 'native_row_not_in_accepted_raw_outcomes')
                item = completed[identifier]
                require(item[0] == split and item[2] == row_hash(row) and item[4] is False, 'native_raw_record_split_or_duplicate_mismatch')
                require(item[5] > previous_ordinal, 'native_merge_order_differs_from_raw_ordinals')
                previous_ordinal = item[5]
                item[4] = True
                native_count += 1
            require(canonical_count == native_count == counts['accepted:' + split], 'generated_split_canonical_native_count_mismatch')
        require(all(item[3] and item[4] for item in completed.values()), 'accepted_raw_record_missing_from_generated_corpora')
        return {'population': registered['name'], 'plan_identity_sha256': plan_identity, 'status': 'passed', 'original_completion_sha256': registered['original'], 'generated_completion_sha256': registered['generated'], 'counts': dict(counts), 'accepted_examples': len(completed), 'configured_stop_token_ids': sorted(stops), 'termination_metadata_counts': dict(terminations), 'retained_output_tokens_including_rejected_outcomes': total_tokens, 'worker_batch_slurm_job_counts': dict(slurm_jobs), 'all_source_user_fields_and_canonical_native_records_match': True, 'prompt_token_hashes_recomputed': False, 'output_last_token_ids_independently_verified': False, 'historical_gpu_ownership_requeried': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'teacher_replay_requires_cpu_slurm')
    root, output = args.root.resolve(), args.output.resolve()
    require(output.is_relative_to(root / 'outputs') and not output.exists(), 'choose_fresh_teacher_report_under_outputs')
    replay, start = Replay(root), time.monotonic()
    try:
        config = json.loads(replay.pin(args.config, args.config_sha256).read_bytes())
        require(len(config['corpora']) == 2 and {item['teacher_plan_identity_sha256'] for item in config['corpora']} == set(REGISTERED), 'teacher_replay_registration_coverage_mismatch')
        populations = [replay.corpus(item) for item in config['corpora']]
        replay.phase = 'final_all_input_bytes_rehash'
        require(all(Path(name).stat().st_size == item['size_bytes'] and digest(Path(name)) == item['sha256'] for name, item in replay.inventory.items()), 'teacher_replay_input_changed_before_completion')
        report = {'status': 'passed_independent_teacher_worker_canonical_native_replay', 'scope': 'both_registered_full_teacher_populations_and_configured_stop_metadata', 'populations': populations, 'unique_original_example_ids': len(replay.global_ids), 'all_bound_input_bytes_stable': True, 'limitations': ['Termination evidence is finish_reason/stop_reason under the hash-pinned stock-vLLM sampling source and target stop configuration, not an independent last-token-ID proof.', 'Raw journals retain output token counts rather than output token sequences; prompt-token hashes are checked for format and binding but not retokenized/recomputed.', 'Rejected-response text and failed next-turn prompts are not retained, so corresponding exclusion reasons are validated structurally and counted, not independently re-evaluated from missing text.', 'Retained batch token counts exclude uncommitted/lost work and model startup. Full physical cost needs separate scheduler/attempt accounting.', 'Model weights, plan/source/shard bytes and accepted output identity are verified; this does not regenerate responses, assert mathematical/code answer quality or establish new inference losslessness.']}
    except Exception as error:
        report = {'status': 'failed', 'phase': replay.phase, 'error_code': str(error) if isinstance(error, ReplayFailure) else type(error).__name__}
    report.update(audit_script_sha256=digest(Path(__file__)), artifact_inventory=replay.inventory, artifact_inventory_sha256=identity(replay.inventory), wall_seconds=time.monotonic() - start, slurm_job_id=os.environ['SLURM_JOB_ID'], cpu_only=True, project_helpers_imported=False, scientific_training_admitted=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': digest(output), 'output': str(output)}), flush=True)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
