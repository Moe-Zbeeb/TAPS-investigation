import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from . import metrics, pool, runner


SCIENCE_KEYS = ('run_id', 'scope', 'effective_config', 'config_sha256', 'plan_sha256', 'asset_receipt_sha256', 'source_receipt_sha256', 'authored_source_hashes', 'code_commit', 'shard_count')
COMMON_KEYS = ('config_sha256', 'plan_sha256', 'asset_receipt_sha256', 'source_receipt_sha256', 'authored_source_hashes', 'code_commit', 'code_dirty', 'source_identity_kind', 'deployed_source_identity_sha256')
METHODS = {'eagle3', 'eagle2', 'hass'}
BENCHMARKS = {'eagle3': {'mt_bench', 'humaneval', 'gsm8k', 'alpaca', 'cnn_dm'}, 'eagle2': {'mt_bench', 'humaneval', 'gsm8k'}, 'hass': {'mt_bench', 'humaneval', 'gsm8k'}}
QUALIFICATIONS = ['Each method retains its own co-located target-only control. Hardware, drivers, allocation priority and preemption histories remain separate per input run.', 'Raw throughput and cross-method timing rankings across nodes can reflect hardware or system differences. AR-normalized speedups remain within-method comparisons; no same-hardware cross-method ranking is established.', 'This collection reproduces configured numerical cells and observed T=0 comparisons only. Temperature-one distributional correctness and losslessness remain unestablished; the scientific training gate remains closed.']


class ReferenceGroupError(ValueError):
    pass


def regular(path):
    path = Path(path)
    metadata = path.lstat()
    if not path.is_absolute() or path.resolve() != path or not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ReferenceGroupError('Expected an owned regular file without aliases: ' + str(path))
    return path


def read(path):
    return json.loads(regular(path).read_bytes(), parse_constant=runner.reject_nonfinite)


def check(condition, message):
    if not condition:
        raise ReferenceGroupError(message)


def cell_key(cell):
    return cell['method'], cell['benchmark'], float(cell['temperature'])


def reference_panel(config):
    references = config.get('reference_cells', [])
    check(config.get('scope') == 'reference' and set(config.get('method_order', [])) == METHODS and len(config['method_order']) == 3 and set(config.get('methods', {})) == METHODS, 'Original configuration must cover all three reference methods')
    check(all(len(config['methods'][method]['benchmarks']) == len(BENCHMARKS[method]) and set(config['methods'][method]['benchmarks']) == BENCHMARKS[method] for method in METHODS), 'Original method benchmark allocation differs from the registered reference panel')
    check(config.get('temperatures') == [0, 1] and config.get('timing_repeats') == 5 and config.get('max_new_tokens') == 512 and config.get('relative_acceptance_tolerance') == 0.05 and config.get('question_limit', 80) == 80, 'Original reference budgets or temperatures differ from the registered panel')
    expected = {(method, benchmark, float(temperature)) for method in config['method_order'] for benchmark in config['methods'][method]['benchmarks'] for temperature in config['temperatures']}
    keys = [cell_key(cell) for cell in references]
    check(len(keys) == 22 and len(set(keys)) == 22 and set(keys) == expected, 'Original configuration must define exactly the disjoint 22 reference cells')
    check(all(cell.get('expected_questions') == 80 and cell.get('expected_turns') == (2 if cell['benchmark'] == 'mt_bench' else 1) and cell.get('timing_repeats') == 5 for cell in references), 'Reference question, turn or timing-repeat coverage differs')
    metrics._validate_references(references)
    return expected


def hardware_identity(manifest):
    inventory, bindings = manifest['gpu_inventory'], manifest['device_bindings']
    count = manifest['shard_count']
    check(type(count) is int and count > 0 and len(inventory) == len(bindings) == count, 'Frozen shard and GPU inventory counts differ')
    check([item['shard_index'] for item in bindings] == list(range(count)), 'Frozen GPU bindings do not cover every shard once')
    for key in ('index', 'uuid'):
        check(len({item[key] for item in inventory}) == count and len({item[key] for item in bindings}) == count, 'Frozen GPU inventory contains duplicate physical devices')
    check(len({item['token'] for item in bindings}) == count, 'Frozen visible GPU tokens are duplicated')
    for binding in bindings:
        matching = [item for item in inventory if item['uuid'] == binding['uuid'] and item['index'] == binding['index']]
        check(len(matching) == 1, 'Frozen GPU binding is absent from its inventory')
    expected = pool.identity({'gpu_inventory': inventory, 'device_bindings': bindings})
    check(expected == manifest.get('hardware_identity_sha256'), 'Frozen hardware identity differs')


def run_documents(root, run_id):
    check(isinstance(run_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id), 'Invalid run ID')
    directory = root / 'outputs/runs' / run_id
    check(directory.resolve() == directory and directory.is_dir(), 'Run directory must exist without aliases')
    manifest_path, state_path, report_path = [directory / name for name in ('manifest.json', 'pool.json', 'report.json')]
    manifest, state, report = [read(path) for path in (manifest_path, state_path, report_path)]
    check(manifest.get('run_id') == run_id and manifest.get('root') == str(root) and manifest.get('run_dir') == str(directory), 'Frozen run ID or root differs')
    check(manifest.get('scope') == 'reference' and manifest.get('effective_config', {}).get('scope') == 'reference', 'Smoke or non-reference runs cannot be collected')
    check(all(key in manifest for key in SCIENCE_KEYS), 'Frozen scientific manifest is incomplete')
    check(pool.identity({key: manifest[key] for key in SCIENCE_KEYS}) == manifest.get('scientific_identity_sha256'), 'Frozen scientific identity differs')
    hardware_identity(manifest)
    expected_state = {'status': 'complete', 'run_id': run_id, 'scope': 'reference', 'active_methods': manifest['active_methods'], 'scientific_identity_sha256': manifest['scientific_identity_sha256'], 'hardware_identity_sha256': manifest['hardware_identity_sha256'], 'manifest_sha256': pool.digest(manifest_path), 'report_sha256': pool.digest(report_path)}
    check(all(state.get(key) == value for key, value in expected_state.items()), 'Pool completion, manifest or report binding differs')
    check(state.get('losslessness_established') is False and state.get('training_performed') is False and state.get('publication_performed') is False, 'Pool scope or correctness claim differs')
    paths = {'authored_source_root': directory / 'inputs/source', 'asset_receipt_path': directory / 'inputs/assets.json', 'source_receipt_path': directory / 'inputs/sources.json', 'asset_registry_path': directory / 'inputs/registries/assets.json', 'source_registry_path': directory / 'inputs/registries/sources.json'}
    check(all(manifest.get(key) == str(value) for key, value in paths.items()), 'Frozen input path differs from the run layout')
    for name, expected in manifest['authored_source_hashes'].items():
        relative = Path(name)
        check(not relative.is_absolute() and '..' not in relative.parts and relative.parts[:1] == ('src',), 'Invalid authored source inventory path')
        check(pool.digest(regular(paths['authored_source_root'] / relative)) == expected, 'Frozen authored source differs')
    for module in (pool, metrics, runner):
        name = 'src/taps_rebuild/' + Path(module.__file__).name
        check(manifest['authored_source_hashes'].get(name) == pool.digest(Path(module.__file__)), 'Current collection dependency differs from frozen source: ' + name)
    for name in ('config.json', 'TAPS_rebuild_plan.pdf'):
        regular(directory / 'inputs' / name)
    original = read(directory / 'inputs/config.json')
    check(pool.digest(directory / 'inputs/config.json') == manifest['config_sha256'] == pool.digest(regular(root / 'configs/reproduction.json')), 'Frozen original configuration differs from the registered reproduction configuration')
    panel = reference_panel(original)
    active = manifest['active_methods']
    check(isinstance(active, list) and active and len(set(active)) == len(active) and set(active) <= METHODS, 'Invalid active method filter')
    check(manifest['effective_config'] == pool.effective_config(original, active, False, manifest['shard_count']), 'Effective configuration differs from the original method filter')
    for name, receipt_key, registry_key, hash_key in (('assets', 'asset_receipt_path', 'asset_registry_path', 'asset_receipt_sha256'), ('sources', 'source_receipt_path', 'source_registry_path', 'source_receipt_sha256')):
        receipt = read(manifest[receipt_key])
        check(receipt.get('status') == 'complete' and receipt.get('root') == str(root) and pool.digest(manifest[receipt_key]) == manifest[hash_key], 'Frozen preparation receipt differs: ' + name)
        check(pool.digest(regular(manifest[registry_key])) == receipt.get('registry_sha256') and receipt.get('source_sha256') == manifest['authored_source_hashes'].get('src/taps_rebuild/' + name + '.py'), 'Frozen preparation registry or builder differs: ' + name)
    return {'run_id': run_id, 'directory': directory, 'manifest': manifest, 'state': state, 'report': report, 'original': original, 'panel': panel, 'manifest_sha256': expected_state['manifest_sha256'], 'pool_sha256': pool.digest(state_path), 'report_sha256': expected_state['report_sha256']}


def worker_proofs(root, run):
    manifest, state, directory = run['manifest'], run['state'], run['directory']
    attempts = state.get('attempts', [])
    check(attempts and attempts[-1].get('status') == 'complete', 'Latest pool attempt is not complete')
    attempt = attempts[-1]
    waves = attempt.get('waves', [])
    check([wave.get('method') for wave in waves] == manifest['active_methods'] and all(wave.get('status') == 'complete' for wave in waves), 'Latest pool attempt does not contain every complete method wave')
    proofs = []
    for wave in waves:
        method = wave['method']
        _, config, sources, _, _, frozen = runner.verify_provenance(root, directory / 'inputs/config.json', directory / 'manifest.json', method)
        bindings = manifest['device_bindings']
        records = wave.get('receipts', [])
        check(len(records) == len(bindings) and len({record['path'] for record in records}) == len(bindings), 'Method wave receipt count or uniqueness differs')
        for binding in bindings:
            shard, count = binding['shard_index'], len(bindings)
            path = directory / 'workers' / (method + '-' + str(shard) + '.json')
            matching = [record for record in records if record['path'] == str(path)]
            receipt = read(path)
            expected_count = pool._expected_cases(method, shard, count, config, sources)
            expected = {**frozen, 'status': 'complete', 'method': method, 'shard_index': shard, 'shard_count': count, 'cuda_visible_devices': binding['token'], 'slurm_job_id': attempt['slurm_job_id'], 'raw_path': str(directory / 'raw' / (method + '-shard' + str(shard) + '.jsonl')), 'planned_cases': expected_count, 'completed_cases': expected_count}
            check(len(matching) == 1 and matching[0].get('sha256') == pool.digest(path) and matching[0].get('completed_cases') == expected_count and expected_count > 0 and all(receipt.get(key) == value for key, value in expected.items()), 'Worker completion receipt or wave binding differs')
            raw = regular(receipt['raw_path'])
            questions = runner.selected_questions(config, sources, method, shard, count)
            planned = runner.planned_cases(config, method, questions)
            observed = set()
            with raw.open('rb') as stream:
                for line in stream:
                    check(line.endswith(b'\n') and line.strip(), 'Raw worker file contains an uncommitted or blank row')
                    row = json.loads(line, parse_constant=runner.reject_nonfinite)
                    check(row.get('warmup') is False, 'Worker raw files must contain only completed measured samples')
                    runner.validate_completed(row, frozen, planned, shard, count)
                    check(row['case_id'] not in observed, 'Raw worker case is duplicated')
                    observed.add(row['case_id'])
            check(observed == planned.keys() and len(observed) == expected_count, 'Raw worker file has incomplete case coverage')
            proofs.append({'method': method, 'shard_index': shard, 'worker_receipt': {'path': str(path), 'sha256': pool.digest(path)}, 'raw': {'path': str(raw), 'sha256': pool.digest(raw), 'completed_records': expected_count}})
    return proofs


def complete_coverage(report, expected):
    check({cell_key(cell) for cell in report['cells']} == expected and len(report['cells']) == len(expected), 'Regenerated report cells differ from the assigned panel')
    for cell in report['cells']:
        check(cell['pairing']['pairing_complete'] and all(cell.get('coverage', {}).get(mode, {}).get('complete') is True for mode in ('speculative', 'target_only')), 'Regenerated reference panel is incomplete or unpaired')


def raw_records(runs):
    for run in runs:
        for proof in run['worker_proofs']:
            raw = proof['raw']
            check(pool.digest(regular(raw['path'])) == raw['sha256'], 'Bound raw file changed during collection')
            with Path(raw['path']).open('rb') as stream:
                for line in stream:
                    yield json.loads(line, parse_constant=runner.reject_nonfinite)


def output_directory(root, output):
    path = Path(output)
    path = path if path.is_absolute() else root / path
    check(path.resolve() == path and path.is_relative_to(root / 'outputs') and path != root / 'outputs' and not path.is_relative_to(root / 'outputs/runs'), 'Output must be a fresh canonical directory below ROOT/outputs outside input runs')
    check(not path.exists() and not path.is_symlink(), 'Output directory already exists; choose a fresh collection ID')
    return path


def collect(root, run_ids, output):
    root = Path(root).resolve()
    check(run_ids and len(set(run_ids)) == len(run_ids), 'Input run IDs must be nonempty and unique')
    destination = output_directory(root, output)
    with ExitStack() as locks:
        runs = []
        for run_id in sorted(run_ids):
            check(isinstance(run_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id), 'Invalid run ID')
            lock_path = regular(root / 'outputs/runs' / run_id / 'pool.lock')
            lock = locks.enter_context(lock_path.open('r+b'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ReferenceGroupError('A controller or orphaned worker still holds the input run lock') from error
            runs.append(run_documents(root, run_id))
        first = runs[0]
        observed = set()
        methods = set()
        for run in runs:
            manifest = run['manifest']
            check(all(manifest.get(key) == first['manifest'].get(key) for key in COMMON_KEYS), 'Input runs have mismatched common configuration, code, asset or source identities')
            check(run['original'] == first['original'], 'Input runs have mismatched original reference configurations')
            selected = {cell_key(cell) for cell in manifest['effective_config']['reference_cells']}
            check(not observed.intersection(selected) and not methods.intersection(manifest['active_methods']), 'Input reference runs overlap methods or cells')
            observed.update(selected)
            methods.update(manifest['active_methods'])
        check(observed == first['panel'] and methods == METHODS, 'Input runs are missing methods or configured reference cells')
        for run in runs:
            manifest = run['manifest']
            selected = {cell_key(cell) for cell in manifest['effective_config']['reference_cells']}
            run['worker_proofs'] = worker_proofs(root, run)
            regenerated, smoke_valid = pool.summarize_run(root, run['directory'], manifest)
            check(not smoke_valid and regenerated == run['report'], 'Frozen report differs from its regenerated raw-log summary')
            complete_coverage(regenerated, selected)
            check(run['state'].get('reproduction_gate_passed') == regenerated['reproduction_gate']['passed'] and run['state'].get('reproduction_gate_status') == regenerated['reproduction_gate']['status'], 'Pool numerical-gate state differs from regenerated evidence')
        config = first['original']
        report = metrics.summarize(raw_records(runs), reference_cells=config['reference_cells'], relative_tolerance=config['relative_acceptance_tolerance'], bootstrap_samples=config.get('bootstrap_samples', 2000), seed=config.get('base_seed', 20261109))
        complete_coverage(report, first['panel'])
        proofs = []
        for run in runs:
            for proof in run['worker_proofs']:
                for item in (proof['raw'], proof['worker_receipt']):
                    check(pool.digest(regular(item['path'])) == item['sha256'], 'Bound raw or worker receipt changed during collection')
            for name, expected in (('manifest.json', run['manifest_sha256']), ('pool.json', run['pool_sha256']), ('report.json', run['report_sha256'])):
                check(pool.digest(regular(run['directory'] / name)) == expected, 'Input completion evidence changed during collection')
            proofs.append({'run_id': run['run_id'], 'manifest_path': str(run['directory'] / 'manifest.json'), 'manifest_sha256': run['manifest_sha256'], 'pool_path': str(run['directory'] / 'pool.json'), 'pool_sha256': run['pool_sha256'], 'report_path': str(run['directory'] / 'report.json'), 'report_sha256': run['report_sha256'], 'manifest': run['manifest'], 'attempt_history': run['state']['attempts'], 'worker_proofs': run['worker_proofs']})
        common = {key: first['manifest'].get(key) for key in COMMON_KEYS}
        manifest = {'schema_version': 1, 'scope': 'reference_group', 'root': str(root), 'output_dir': str(destination), 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'collector_sha256': pool.digest(Path(__file__)), 'common_identity': common, 'common_identity_sha256': pool.identity(common), 'original_config': config, 'reference_cells': config['reference_cells'], 'input_runs': proofs, 'cross_node_timing_qualifications': QUALIFICATIONS, 'losslessness_established': False, 'temperature_one_distributional_correctness': 'requires_separate_statistical_check', 'automatic_pipeline_gate_pass': False}
        report.update(scope='reference_group', active_methods=config['method_order'], reproduction_gate_scope='all_22_reference_cells', input_run_ids=[run['run_id'] for run in runs], raw_inputs=[proof['raw'] for run in runs for proof in run['worker_proofs']], cross_node_timing_qualifications=QUALIFICATIONS, automatic_pipeline_gate_pass=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()
        pool.atomic_json(destination / 'manifest.json', manifest)
        report['manifest_sha256'] = pool.digest(destination / 'manifest.json')
        pool.atomic_json(destination / 'report.json', report)
        text = metrics._markdown(report) + '\n\n' + '\n'.join('- ' + qualification for qualification in QUALIFICATIONS) + '\n'
        metrics._write(destination / 'report.md', text)
        state = {'status': 'complete', 'scope': 'reference_group', 'input_run_ids': report['input_run_ids'], 'reference_cells': 22, 'manifest_sha256': report['manifest_sha256'], 'report_sha256': pool.digest(destination / 'report.json'), 'reproduction_gate_passed': report['reproduction_gate']['passed'], 'reproduction_gate_status': report['reproduction_gate']['status'], 'losslessness_established': False, 'automatic_pipeline_gate_pass': False, 'training_performed': False, 'publication_performed': False}
        pool.atomic_json(destination / 'collection.json', state)
        return state


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = collect(args.root, args.run_id, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, pool.PoolError) as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
