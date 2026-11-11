import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def key(benchmark, question, repeat):
    return benchmark, question, repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Replay requires CPU Slurm')
    original = args.root / 'outputs/runs/official-eagle3-replication-20261110-a10080-v1'
    current = args.root / 'outputs/seed-policy/first-turn-20261111-v1'
    require(digest(original / 'report.json') == '37cc63550c3f70fd0280e6348af415fd00b0c34ffd40e4da02b8e3a37df0a6d9', 'Original completed report changed')
    report = read(current / 'report.json')
    manifest = read(current / 'manifest.json')
    require(digest(current / 'manifest.json') == report['manifest_sha256'], 'Seed report manifest changed')
    driver = args.root / 'outputs/preparation/seed-policy-20261111-v1/driver.py'
    require(digest(driver) == manifest['source_sha256'] == '40672807438ed129f0de86aca6d7c154ce056d436144646b8cad1c4b6a8558be', 'Seed collector changed')
    originals, new_rows, proofs = {}, [], []
    for item in read(original / 'report.json')['raw_inputs']:
        require(digest(item['path']) == item['sha256'], 'Original raw bytes changed')
        proofs.append(item)
        with Path(item['path']).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('warmup') or row['temperature'] != 1 or row['generation_mode'] != 'speculative' or row['turn_id'] != 0 or row['benchmark'] not in {'alpaca', 'gsm8k'}:
                    continue
                case_key = key(row['benchmark'], row['question_id'], row['repeat'])
                require(case_key not in originals, 'Duplicate original measurement')
                originals[case_key] = row
    for item in report['workers']:
        path = Path(item['path'])
        require(digest(path) == item['sha256'] and digest(path.with_suffix('.receipt.json')) == item['receipt_sha256'], 'New raw worker proof changed')
        proofs.append(item)
        with path.open() as stream:
            new_rows.extend(json.loads(line) for line in stream)
    inputs = read(current / 'inputs.json')
    require(digest(current / 'inputs.json') == manifest['inputs_sha256'], 'Seed input bytes changed')
    spec = importlib.util.spec_from_file_location('completed_seed_collector', driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recomputed = module.summarize_rows(new_rows, inputs['cases'])
    require(recomputed['cells'] == report['cells'] and recomputed['measured_generations'] == report['measured_generations'], 'Recomputed seed statistics differ')
    seen, mismatches = set(), []
    for row in new_rows:
        if row['arm'] != 'registered_hash':
            continue
        case = row['case']
        case_key = key(case['benchmark'], case['question_id'], row['repeat'])
        require(case_key not in seen and case_key in originals, 'Legacy replay coverage differs')
        seen.add(case_key)
        old = originals[case_key]
        checks = {'input_tokens': old['input_token_ids'] == case['input_token_ids'], 'seed': old['seed'] == row['actual_seed'], 'raw_tokens': old['output_token_ids'] == row['output_token_ids'], 'comparison_tokens': old['comparison_token_ids'] == row['comparison_token_ids'], 'cycle_emissions': [cycle['emitted_tokens'] for cycle in old['cycles']] == [cycle['emitted_tokens'] for cycle in row['cycles']]}
        if not all(checks.values()):
            mismatches.append({'benchmark': case_key[0], 'question_id': case_key[1], 'repeat': case_key[2], 'checks': checks})
    require(len(originals) == len(seen) == 800 and seen == set(originals), 'Incomplete 800-generation legacy replay')
    value = {'status': 'complete', 'scope': 'cpu_raw_replay_and_legacy_token_comparison_not_independent_review', 'as_of_utc': datetime.now(timezone.utc).isoformat(), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'source_sha256': digest(__file__), 'seed_report_sha256': digest(current / 'report.json'), 'recomputed_generations': len(new_rows), 'statistics_recomputed_exactly': True, 'legacy_pairs': len(seen), 'legacy_mismatches': mismatches, 'all_legacy_tokens_and_cycle_emissions_identical': not mismatches, 'raw_proofs': proofs, 'training_admission': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({name: value[name] for name in ['status', 'statistics_recomputed_exactly', 'legacy_pairs', 'all_legacy_tokens_and_cycle_emissions_identical']}))


if __name__ == '__main__':
    main()
