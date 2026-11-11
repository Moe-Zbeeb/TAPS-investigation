import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def greedy(directory):
    counts, failures = Counter(), []
    for path in sorted(directory.glob('greedy-*.jsonl')):
        receipt = read(path.with_suffix('.receipt.json'))
        if receipt['status'] != 'complete' or receipt['raw_sha256'] != digest(path):
            raise ValueError('Greedy worker proof differs')
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                ar = row['outputs']['target_only']['comparison_token_ids']
                spec = row['outputs']['speculative']['comparison_token_ids']
                counts[row['method']] += 1
                if row['precision'] != 'float32' or row['equal'] != (ar == spec):
                    raise ValueError('Greedy comparison or precision differs')
                if ar == spec:
                    continue
                position = next((index for index, (a, b) in enumerate(zip(ar, spec)) if a != b), min(len(ar), len(spec)))
                case = row['case']
                metadata = {mode: row['outputs'][mode]['backend_metadata'] for mode in ('target_only', 'speculative')}
                ar_meta = metadata['target_only']
                guard = ar_meta['context_guard']
                length_only = position == len(ar) and len(ar) < len(spec)
                context_censored = length_only and ar_meta['first_stop_token_index'] is None and ar_meta['effective_context_overshoot_tokens'] == 1 and guard['prompt_tokens'] + len(ar) == guard['effective_context_limit'] + 1 and len(ar) < 512
                stop = {mode: {key: values.get(key) for key in ['first_stop_token_index', 'effective_context_limit', 'effective_context_overshoot_tokens', 'cap_overshoot_tokens', 'context_guard']} for mode, values in metadata.items()}
                failures.append({'method': row['method'], 'benchmark': case['benchmark'], 'question_id': case['question_id'], 'turn_id': case['turn_id'], 'input_sha256': case['input_sha256'], 'input_tokens': len(case['input_token_ids']), 'first_difference': position, 'ar_length': len(ar), 'speculative_length': len(spec), 'ar_tokens_at_difference': ar[position:position + 4], 'speculative_tokens_at_difference': spec[position:position + 4], 'length_only_extension': length_only, 'context_limit_censoring_explains_length': context_censored, 'stopping_metadata': stop})
    if dict(counts) != {'eagle3': 480, 'eagle2': 320, 'hass': 320}:
        raise ValueError('Incomplete greedy panel')
    return {'counts': dict(counts), 'mismatches': failures, 'summary_sha256': digest(directory / 'greedy-summary.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run the audit inside a CPU Slurm task')
    root = args.root.resolve()
    result = {'as_of_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'read_only_diagnostics_no_admission_change', 'source_sha256': digest(__file__), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'slurm_step_id': os.environ.get('SLURM_STEP_ID'), 'replications': {}, 'training_started': False}
    for label, run_id in [('eagle3', 'official-eagle3-replication-20261110-a10080-v1'), ('eagle2_hass', 'official-eagle2-hass-replication-20261110-a10080-v1')]:
        directory = root / 'outputs/runs' / run_id
        report, state = read(directory / 'report.json'), read(directory / 'pool.json')
        if state['status'] != 'complete' or state['report_sha256'] != digest(directory / 'report.json'):
            raise ValueError('Replication completion binding differs')
        cells = []
        for row in report['cells']:
            ref = row['reference']['published_tau']
            spec = row['speculative']
            mean, pooled = spec['question_mean_tokens_per_cycle'], spec['pooled_tokens_per_cycle']
            cells.append({'method': row['method'], 'benchmark': row['benchmark'], 'temperature': row['temperature'], 'published_tau': ref, 'question_mean_tau': mean, 'question_mean_ci95': spec['question_mean_tokens_per_cycle_ci95'], 'pooled_tau_diagnostic': pooled, 'relative_tau_error': (mean - ref) / ref, 'within_five_percent': abs(mean - ref) / ref <= 0.05, 'strict_gate': row['reproduction_gate'], 't0_mismatched_pairs': row['pairing']['t0_sequence_mismatch_count']})
        result['replications'][label] = {'run_id': run_id, 'report_sha256': digest(directory / 'report.json'), 'gate': report['reproduction_gate'], 'cells': cells}
    primary = greedy(root / 'outputs/validation-pipelines/primary-2212576-validation-queue-20261110-v2-full')
    repeat = greedy(root / 'outputs/validation-pipelines/greedy-node11-replication-20261110-v1')
    key = lambda row: (row['method'], row['benchmark'], row['question_id'], row['turn_id'], row['input_sha256'])
    result['greedy_controls'] = {'primary': primary, 'node11_repeat': repeat, 'same_mismatch_cases': {key(row) for row in primary['mismatches']} == {key(row) for row in repeat['mismatches']}, 'same_mismatch_details': primary['mismatches'] == repeat['mismatches']}
    corpus = root / 'outputs/teacher-generation/sharegpt-ultrachat-teacher-20261110-v3'
    workers = [read(corpus / ('worker-' + str(rank) + '-progress.json')) for rank in range(8)]
    result['data'] = {'source_population': 540107, 'processed': sum(row['completed_examples'] for row in workers), 'generated_tokens_this_attempt': sum(row['generated_tokens_this_attempt'] for row in workers), 'complete': (corpus / 'completion.json').exists(), 'workers': workers}
    result['tau_cells_within_tolerance'] = sum(row['within_five_percent'] for group in result['replications'].values() for row in group['cells'])
    result['strict_cells_passed'] = sum(group['gate']['passed_reference_cells'] for group in result['replications'].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps({key: result[key] for key in ['as_of_utc', 'tau_cells_within_tolerance', 'strict_cells_passed', 'greedy_controls']}))


if __name__ == '__main__':
    main()
