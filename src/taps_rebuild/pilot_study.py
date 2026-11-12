import argparse
from collections import Counter
import json
import os
from pathlib import Path

from . import native_training as native, pool
from .pilot_cache import quantile_positions, read, records, require


DOMAINS = ('chat', 'math', 'code')


def cycle_order(rows, seed, rank, domain, cycle):
    return sorted(rows, key=lambda row: (pool.identity(['pilot-order-v1', seed, rank, domain, cycle, row['example_id']]), row['example_id']))


def schedule(pools, cohort, seed, rank, steps=1536):
    require(cohort in (*DOMAINS, 'generalist') and steps > 0 and (cohort != 'generalist' or steps % 3 == 0), 'Pilot schedule geometry differs')
    domains = DOMAINS if cohort == 'generalist' else (cohort,)
    require(all(pools[domain] for domain in domains), 'Pilot schedule pool is empty')
    cursors, orders, occurrences = Counter(), {}, Counter()
    for step in range(steps):
        domain = DOMAINS[(step + rank) % 3] if cohort == 'generalist' else cohort
        count, rows = cursors[domain], pools[domain]
        cycle, offset = divmod(count, len(rows))
        key = domain, cycle
        if key not in orders:
            orders[key] = cycle_order(rows, seed, rank, domain, cycle)
        row = orders[key][offset]
        identity = row['example_id']
        yield {'step': step, 'domain': domain, 'example_id': identity, 'index': row['index'], 'rank': rank, 'cycle': cycle, 'occurrence': occurrences[identity]}
        occurrences[identity] += 1
        cursors[domain] += 1


def mask_positions(row, quota, seed, occurrence, policy):
    return quantile_positions(row['eligible_positions'][policy], quota, seed, row['example_id'], occurrence)


def verify_identity(value):
    copied = dict(value)
    require(copied.pop('identity_sha256') == pool.identity(copied), 'Pilot artifact identity changed')
    return value


def prepare(root, cache, eligibility, config, output):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Pilot schedule preparation requires CPU Slurm')
    root, cache, eligibility, output = (Path(value).resolve() for value in (root, cache, eligibility, output))
    require(output.is_relative_to(root / 'outputs') and not output.exists(), 'Choose a fresh pilot study output')
    recipe = read(config)
    require(recipe['cohorts'] == [*DOMAINS, 'generalist'] and recipe['response_context_policies'] == ['original', 'target'] and recipe['seeds'] == [0, 1, 2] and recipe['world_size'] == 8 and recipe['examples_per_rank'] == 1536 and recipe['length'] == 2048, 'The preregistered24-run pilot grid differs')
    prerequisite = recipe['pipeline_prerequisite']
    require(pool.digest(root / prerequisite['path']) == prerequisite['sha256'], 'Prior independently reviewed pipeline evidence changed')
    cache_receipt = verify_identity(read(cache / 'completion.json'))
    gate = verify_identity(read(eligibility / 'completion.json'))
    require(gate['status'] == 'eligible_for_schedule_preparation' and gate['quota'] in (64, 32) and gate['cache_identity_sha256'] == cache_receipt['identity_sha256'] and gate['provenance_audit']['status'] == 'passed', 'Matched pilot data is not eligible')
    for directory, receipt in ((cache, cache_receipt), (eligibility, gate)):
        require(all(pool.digest(directory / name) == item['sha256'] for name, item in receipt['files'].items()), 'Pilot data artifact changed before schedule preparation')
    output.mkdir(parents=True)
    (output / 'plans').mkdir()
    (output / 'schedules').mkdir()
    quota, files, statistics = gate['quota'], {}, {}
    for rank in range(8):
        pools = {domain: [] for domain in DOMAINS}
        for row in records(eligibility / ('rank-' + str(rank) + '.eligible.jsonl')):
            if row['split'] == 'train' and min(map(len, row['eligible_positions'])) >= quota:
                require(row['rank'] == rank, 'A pilot pool crossed ranks')
                pools[row['domain']].append(row)
        require(all(len(rows) >= 128 for rows in pools.values()), 'Pilot rank lacks sufficient paired training groups')
        by_id = {row['example_id']: row for rows in pools.values() for row in rows}
        require(len(by_id) == sum(map(len, pools.values())), 'Pilot ID duplicated across domains')
        for seed in recipe['seeds']:
            for cohort in recipe['cohorts']:
                name = cohort + '-seed' + str(seed) + '-rank' + str(rank) + '.jsonl'
                path = output / 'schedules' / name
                counts, repetitions, context_coverage = Counter(), Counter(), [set(), set()]
                with path.open('xb') as stream:
                    for item in schedule(pools, cohort, seed, rank):
                        row = by_id[item['example_id']]
                        positions = [mask_positions(row, quota, seed, item['occurrence'], policy) for policy in (0, 1)]
                        item['mask_sha256'] = [pool.identity(value) for value in positions]
                        item['effective_contexts'] = quota
                        item['real_lengths'] = row['lengths']
                        stream.write(native.canonical(item) + b'\n')
                        counts[item['domain']] += 1
                        repetitions[item['example_id']] += 1
                        for policy in (0, 1):
                            context_coverage[policy].update((item['example_id'], value) for value in positions[policy])
                    stream.flush()
                    os.fsync(stream.fileno())
                expected = {domain: 512 for domain in DOMAINS} if cohort == 'generalist' else {cohort: 1536}
                require(dict(counts) == expected, 'Pilot domain budget differs')
                files['schedules/' + name] = {'sha256': pool.digest(path), 'size_bytes': path.stat().st_size}
                statistics[name] = {'domain_examples': dict(counts), 'pool_examples': {domain: len(rows) for domain, rows in pools.items()}, 'unique_examples': len(repetitions), 'repeated_exposures': 1536 - len(repetitions), 'repeat_multiplicity_histogram': {str(key): value for key, value in Counter(repetitions.values()).items()}, 'unique_base_contexts_original_target': list(map(len, context_coverage)), 'effective_contexts': 1536 * quota, 'per_head_effective_contexts': [1536 * quota] * 7, 'dense_slots': 1536 * 2048}
    source = Path(__file__).parent
    names = ['native_training.py', 'pool.py', 'pilot_cache.py', 'pilot_gate.py', 'pilot_study.py', 'pilot_train.py', 'pilot_queue.py', 'pilot_pipeline.py', 'pilot_smoke.py', 'native_gradients.py']
    code = {name: pool.digest(source / name) for name in names}
    order = []
    for seed in recipe['seeds']:
        conditions = [(cohort, policy) for cohort in recipe['cohorts'] for policy in recipe['response_context_policies']]
        order.extend((seed, cohort, policy) for cohort, policy in sorted(conditions, key=lambda item: pool.identity(['pilot-execution-order-v1', seed, *item])))
    common = {'schema_version': 1, 'study_id': recipe['study_id'], 'project_root': str(root), 'cache': str(cache), 'eligibility': str(eligibility), 'cache_identity_sha256': cache_receipt['identity_sha256'], 'eligibility_identity_sha256': gate['identity_sha256'], 'recipe': recipe, 'recipe_sha256': pool.digest(config), 'source_directory': str(source.resolve()), 'source_sha256': code, 'target': cache_receipt['target'], 'ancestor': cache_receipt['ancestor'], 'vocabulary_identity_sha256': gate['shared_binding']['vocabulary_identity_sha256'], 'hardware': gate['shared_binding']['hardware'], 'scan_runtime': gate['shared_binding']['runtime'], 'quota': quota, 'study_directory': str(output)}
    plans = []
    for ordinal, (seed, cohort, policy) in enumerate(order):
        run_id = recipe['study_id'] + '-' + cohort + '-' + policy + '-s' + str(seed)
        model_path = (root / 'models').resolve() / 'trained/pilot-24' / run_id
        require(not model_path.exists(), 'Pilot model output already exists')
        schedules = [{'path': str(output / 'schedules' / (cohort + '-seed' + str(seed) + '-rank' + str(rank) + '.jsonl')), 'sha256': files['schedules/' + cohort + '-seed' + str(seed) + '-rank' + str(rank) + '.jsonl']['sha256']} for rank in range(8)]
        plan = {**common, 'scope': 'scientific_pilot_requires_admission', 'run_id': run_id, 'ordinal': ordinal, 'cohort': cohort, 'policy': policy, 'policy_index': recipe['response_context_policies'].index(policy), 'seed': seed, 'steps': 1536, 'schedules': schedules, 'output': str(output / 'runs' / run_id), 'trained_models': str(model_path), 'initialization': 'released_common_ancestor', 'status': 'prepared_waiting_source_matched_qualification'}
        plan['identity_sha256'] = pool.identity(plan)
        path = output / 'plans' / (run_id + '.json')
        pool.atomic_json(path, plan)
        plans.append({'path': str(path), 'sha256': pool.digest(path), 'identity_sha256': plan['identity_sha256'], 'run_id': run_id})
        files[str(path.relative_to(output))] = {'sha256': pool.digest(path), 'size_bytes': path.stat().st_size}
    report = {'status': 'prepared_24_waiting_qualification', 'scope': recipe['scope'], 'common': common, 'plans': plans, 'files': files, 'schedule_statistics': statistics, 'logical_budget_per_run': {'examples': 12288, 'effective_base_contexts': 12288 * quota, 'per_head_effective_contexts': [12288 * quota] * 7, 'dense_slots': 25165824, 'attempted_and_successful_updates': 1536}, 'training_admitted': False}
    report['identity_sha256'] = pool.identity(report)
    pool.atomic_json(output / 'manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'cache', 'eligibility', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.cache, args.eligibility, args.config, args.output), indent=2))


if __name__ == '__main__':
    main()
