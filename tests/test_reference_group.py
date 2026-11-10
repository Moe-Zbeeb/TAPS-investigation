import copy
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from taps_rebuild import pool, reference_group as group, runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pool.canonical(value) + b'\n')


class Fixture:
    def __init__(self, root, tau=1.0):
        self.root = Path(root).resolve()
        source = Path(pool.__file__).parent
        for name in ('pool.py', 'runner.py', 'metrics.py', 'assets.py', 'sources.py', '__init__.py', 'backends/official.py'):
            destination = self.root / 'src/taps_rebuild' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, destination)
        original = json.loads((source.parents[1] / 'configs/reproduction.json').read_bytes())
        original.update(cluster_root=str(self.root), legacy_runtime_root=str(self.root / 'legacy'), bootstrap_samples=1)
        for reference in original['reference_cells']:
            reference.update(published_tau=tau, source_url='https://example.test/unit-test-fixture')
        self.root.joinpath('TAPS_rebuild_plan.pdf').write_bytes(b'synthetic unit-test plan')
        original['plan_sha256'] = pool.digest(self.root / 'TAPS_rebuild_plan.pdf')
        self.config = original
        self.config_path = self.root / 'configs/reproduction.json'
        write(self.config_path, original)
        models = {}
        for key in {definition[role] for definition in original['methods'].values() for role in ('target', 'drafter')}:
            directory = self.root / 'models' / key
            write(directory / 'config.json', {'unit_test_fixture': True})
            files = {'config.json': {'size_bytes': (directory / 'config.json').stat().st_size, 'sha256': pool.digest(directory / 'config.json')}}
            models[key] = {'status': 'verified', 'path': str(directory), 'files': files, 'model_identity_sha256': pool.identity(files)}
        upstream = {}
        for name in ('EAGLE', 'HASS'):
            directory = self.root / 'repos' / name
            directory.mkdir(parents=True)
            file = directory / 'fixture.txt'
            file.write_bytes(name.encode())
            tracked = {'fixture.txt': {'kind': 'regular_file', 'size_bytes': file.stat().st_size, 'sha256': pool.digest(file)}}
            upstream[name] = {'path': str(directory), 'revision': 'a' * 40, 'tracked_files': tracked, 'source_identity_sha256': runner.identity(tracked), 'system_prompt': {'text': 'fixture prompt', 'sha256': runner.hashlib.sha256(b'fixture prompt').hexdigest()}}
        benchmarks = {}
        for benchmark in {benchmark for definition in original['methods'].values() for benchmark in definition['benchmarks']}:
            turns = 2 if benchmark == 'mt_bench' else 1
            file = self.root / 'data' / (benchmark + '.jsonl')
            file.parent.mkdir(exist_ok=True)
            file.write_bytes(b''.join(pool.canonical({'question_id': index, 'turns': ['fixture question'] * turns}) + b'\n' for index in range(80)))
            benchmarks[benchmark] = {'path': str(file), 'sha256': pool.digest(file), 'rows': 80, 'turns_per_row': turns, 'question_ids': list(range(80))}
        for name, contents in (('assets', {'models': models}), ('sources', {'sources': upstream, 'benchmarks': benchmarks})):
            registry = self.root / 'configs' / (name + '.json')
            write(registry, {'unit_test_fixture': name})
            write(self.root / 'outputs/preparation' / (name + '.json'), {'status': 'complete', 'root': str(self.root), 'registry': str(registry), 'registry_sha256': pool.digest(registry), 'source_sha256': pool.digest(self.root / 'src/taps_rebuild' / (name + '.py')), **contents})
        hashes = {str(path.relative_to(self.root)): pool.digest(path) for path in (self.root / 'src/taps_rebuild').rglob('*.py')}
        write(self.root / 'source-identity.json', {'git_commit': 'a' * 40, 'tracked_hashes': hashes, 'plan_sha256': original['plan_sha256']})

    def freeze(self, run_id, methods, count=2, smoke=False):
        inventory = [{'index': str(index), 'uuid': 'GPU-' + run_id + '-' + str(index), 'name': 'NVIDIA A100', 'memory_total_mib': 81920.0, 'driver_version': '580.95.05'} for index in range(count)]
        bindings = [{'shard_index': index, 'token': str(index), 'uuid': row['uuid'], 'index': row['index']} for index, row in enumerate(inventory)]
        hardware = {'gpu_inventory': inventory, 'device_bindings': bindings, 'hardware_identity_sha256': pool.identity({'gpu_inventory': inventory, 'device_bindings': bindings}), 'slurm': {'job': {'JobId': '123', 'Partition': 'high-priority', 'QOS': 'spec-priority'}, 'node': {'NodeName': run_id}}}
        effective = pool.effective_config(self.config, methods, smoke, count)
        prepared = pool.prerequisites(self.root, self.config_path, effective)
        directory = self.root / 'outputs/runs' / run_id
        directory.mkdir(parents=True)
        manifest = pool.freeze_run(self.root, directory, run_id, effective, prepared, hardware)
        (directory / 'pool.lock').touch()
        write(directory / 'report.json', {'unit_test_placeholder': True})
        state = {'status': 'complete', 'run_id': run_id, 'scope': manifest['scope'], 'active_methods': manifest['active_methods'], 'scientific_identity_sha256': manifest['scientific_identity_sha256'], 'hardware_identity_sha256': manifest['hardware_identity_sha256'], 'manifest_sha256': pool.digest(directory / 'manifest.json'), 'report_sha256': pool.digest(directory / 'report.json'), 'losslessness_established': False, 'training_performed': False, 'publication_performed': False, 'attempts': []}
        write(directory / 'pool.json', state)
        return directory, manifest, state

    def complete(self, run_id, methods, count):
        directory, manifest, state = self.freeze(run_id, methods, count)
        waves = []
        for method in manifest['active_methods']:
            _, config, sources, _, _, frozen = runner.verify_provenance(self.root, directory / 'inputs/config.json', directory / 'manifest.json', method)
            proofs = []
            for shard in range(count):
                questions = runner.selected_questions(config, sources, method, shard, count)
                planned = runner.planned_cases(config, method, questions)
                rows = []
                for fields in planned.values():
                    row = {**frozen, **fields, 'status': 'completed', 'warmup': False, 'shard_index': shard, 'shard_count': count, 'input_token_ids': [7, 8], 'input_sha256': runner.identity([7, 8]), 'output_text': 'fixture', 'output_token_ids': [42], 'comparison_token_ids': [42], 'new_token_count': 1, 'cycle_count': 1, 'wall_ms': 2 if fields['generation_mode'] == 'target_only' else 1, 'unused_final_drafting_ms': 0, 'cycles': [{'accepted_draft_tokens': 0, 'emitted_tokens': 1, 'drafting_ms': 0.1, 'verification_ms': 0.5, 'verifier_nodes': 1, 'proposed_nodes': 0}]}
                    rows.append(row)
                raw = directory / 'raw' / (method + '-shard' + str(shard) + '.jsonl')
                raw.parent.mkdir(exist_ok=True)
                raw.write_bytes(b''.join(pool.canonical(row) + b'\n' for row in rows))
                receipt = directory / 'workers' / (method + '-' + str(shard) + '.json')
                write(receipt, {**frozen, 'status': 'complete', 'method': method, 'shard_index': shard, 'shard_count': count, 'cuda_visible_devices': str(shard), 'slurm_job_id': '123', 'raw_path': str(raw), 'planned_cases': len(rows), 'completed_cases': len(rows)})
                proofs.append({'path': str(receipt), 'sha256': pool.digest(receipt), 'completed_cases': len(rows)})
            waves.append({'method': method, 'status': 'complete', 'receipts': proofs})
        report, _ = pool.summarize_run(self.root, directory, manifest)
        write(directory / 'report.json', report)
        state.update(report_sha256=pool.digest(directory / 'report.json'), reproduction_gate_passed=report['reproduction_gate']['passed'], reproduction_gate_status=report['reproduction_gate']['status'], attempts=[{'status': 'complete', 'slurm_job_id': '123', 'node': run_id, 'waves': waves}])
        write(directory / 'pool.json', state)
        return directory, manifest, state


class ReferenceGroupTests(unittest.TestCase):
    def test_complete_execution_with_failed_numerical_reproduction_is_reported_truthfully(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary, tau=2.0)
            fixture.complete('all', 'eagle3,eagle2,hass', 2)
            result = group.collect(fixture.root, ['all'], fixture.root / 'outputs/groups/numerical-failure')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['reproduction_gate_status'], 'failed')
            self.assertFalse(result['reproduction_gate_passed'])
            self.assertFalse(result['automatic_pipeline_gate_pass'])

    def test_complete_three_node_disjoint_panel_regenerates_all_22_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            for run_id, methods, count in [('e3', 'eagle3', 9), ('e2', 'eagle2', 8), ('hass', 'hass', 8)]:
                fixture.complete(run_id, methods, count)
            destination = fixture.root / 'outputs/reference-groups/full'
            result = group.collect(fixture.root, ['hass', 'e3', 'e2'], destination)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['reproduction_gate_passed'])
            self.assertFalse(result['automatic_pipeline_gate_pass'])
            report = json.loads((destination / 'report.json').read_bytes())
            manifest = json.loads((destination / 'manifest.json').read_bytes())
            self.assertEqual(report['reproduction_gate']['reference_cells'], 22)
            self.assertEqual({len(run['manifest']['device_bindings']) for run in manifest['input_runs']}, {8, 9})
            self.assertEqual(sum(proof['raw']['completed_records'] for run in manifest['input_runs'] for proof in run['worker_proofs']), report['analyzed_records'])
            self.assertFalse(report['reproduction_gate']['losslessness_established'])
            self.assertTrue((destination / 'report.md').is_file())
            with self.assertRaisesRegex(group.ReferenceGroupError, 'already exists'):
                group.collect(fixture.root, ['e3', 'e2', 'hass'], destination)
            raw = fixture.root / 'outputs/runs/e3/raw/eagle3-shard0.jsonl'
            rows = raw.read_bytes().splitlines(keepends=True)
            original = rows[0]
            tampered = json.loads(original)
            tampered['seed'] += 1
            rows[0] = pool.canonical(tampered) + b'\n'
            raw.write_bytes(b''.join(rows))
            with self.assertRaisesRegex(ValueError, 'planned cases'):
                group.collect(fixture.root, ['e3', 'e2', 'hass'], fixture.root / 'outputs/reference-groups/tampered')
            rows[0] = original
            raw.write_bytes(b''.join(rows))
            receipt = fixture.root / 'outputs/runs/e3/workers/eagle3-0.json'
            changed = json.loads(receipt.read_bytes())
            changed['status'] = 'failed'
            write(receipt, changed)
            with self.assertRaisesRegex(group.ReferenceGroupError, 'receipt or wave binding'):
                group.collect(fixture.root, ['e3', 'e2', 'hass'], fixture.root / 'outputs/reference-groups/failed-worker')

    def test_reference_panel_rejects_missing_cell_and_smoke_budgets(self):
        original = json.loads((Path(pool.__file__).parents[2] / 'configs/reproduction.json').read_bytes())
        self.assertEqual(len(group.reference_panel(original)), 22)
        for changes in ({'reference_cells': original['reference_cells'][:-1]}, {'max_new_tokens': 32}, {'timing_repeats': 1}):
            with self.assertRaises(group.ReferenceGroupError):
                group.reference_panel({**original, **changes})
        altered = copy.deepcopy(original)
        altered['methods']['eagle2']['benchmarks'], altered['methods']['eagle3']['benchmarks'] = altered['methods']['eagle3']['benchmarks'], altered['methods']['eagle2']['benchmarks']
        with self.assertRaisesRegex(group.ReferenceGroupError, 'benchmark allocation'):
            group.reference_panel(altered)

    def test_missing_overlap_and_common_code_mismatch_fail_before_raw_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            first, _, _ = fixture.freeze('first', 'eagle3')
            second, manifest, state = fixture.freeze('second', 'eagle3')
            with self.assertRaisesRegex(group.ReferenceGroupError, 'missing methods'):
                group.collect(fixture.root, ['first'], fixture.root / 'outputs/groups/missing')
            with self.assertRaisesRegex(group.ReferenceGroupError, 'overlap'):
                group.collect(fixture.root, ['first', 'second'], fixture.root / 'outputs/groups/overlap')
            manifest['code_dirty'] = True
            write(second / 'manifest.json', manifest)
            state['manifest_sha256'] = pool.digest(second / 'manifest.json')
            write(second / 'pool.json', state)
            with self.assertRaisesRegex(group.ReferenceGroupError, 'mismatched common'):
                group.collect(fixture.root, ['first', 'second'], fixture.root / 'outputs/groups/code')

    def test_run_document_identity_state_report_and_alias_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            directory, manifest, state = fixture.freeze('reference', 'eagle3')
            original = (directory / 'manifest.json').read_bytes()
            group.run_documents(fixture.root, 'reference')
            changed = copy.deepcopy(manifest)
            changed['effective_config']['max_new_tokens'] = 31
            write(directory / 'manifest.json', changed)
            with self.assertRaisesRegex(group.ReferenceGroupError, 'scientific identity'):
                group.run_documents(fixture.root, 'reference')
            (directory / 'manifest.json').write_bytes(original)
            changed = copy.deepcopy(manifest)
            changed['gpu_inventory'][0]['driver_version'] = 'changed'
            write(directory / 'manifest.json', changed)
            with self.assertRaisesRegex(group.ReferenceGroupError, 'hardware identity'):
                group.run_documents(fixture.root, 'reference')
            (directory / 'manifest.json').write_bytes(original)
            write(directory / 'report.json', {'tampered': True})
            with self.assertRaisesRegex(group.ReferenceGroupError, 'report binding'):
                group.run_documents(fixture.root, 'reference')
            actual = directory / 'pool.json'
            actual.rename(directory / 'state-copy.json')
            actual.symlink_to(directory / 'state-copy.json')
            with self.assertRaisesRegex(group.ReferenceGroupError, 'without aliases'):
                group.run_documents(fixture.root, 'reference')

    def test_smoke_live_lock_and_missing_resume_evidence_fail_before_reporting(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            directory, _, _ = fixture.freeze('smoke', 'eagle3', smoke=True)
            with self.assertRaisesRegex(group.ReferenceGroupError, 'Smoke'):
                group.run_documents(fixture.root, 'smoke')
            with (directory / 'pool.lock').open('r+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(group.ReferenceGroupError, 'still holds'):
                    group.collect(fixture.root, ['smoke'], fixture.root / 'outputs/reference-groups/live')
            (directory / 'pool.json').unlink()
            with self.assertRaises(FileNotFoundError):
                group.run_documents(fixture.root, 'smoke')

    def test_nfs_exclusive_lock_uses_write_access_without_changing_lock_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            directory, _, _ = fixture.freeze('smoke', 'eagle3', smoke=True)
            path = directory / 'pool.lock'
            contents = b'preserve existing lock contents\n'
            path.write_bytes(contents)
            original = fcntl.flock
            observed = []

            def nfs_lock(file, operation):
                descriptor = file.fileno() if hasattr(file, 'fileno') else file
                access = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
                if operation & fcntl.LOCK_EX and access == os.O_RDONLY:
                    raise OSError(errno.EBADF, 'NFS requires write access for an exclusive lock')
                observed.append(access)
                return original(file, operation)

            output = fixture.root / 'outputs/reference-groups/nfs'
            with patch.object(group.fcntl, 'flock', side_effect=nfs_lock):
                with self.assertRaisesRegex(group.ReferenceGroupError, 'Smoke'):
                    group.collect(fixture.root, ['smoke'], output)
            self.assertEqual(observed, [os.O_RDWR])
            self.assertEqual(path.read_bytes(), contents)
            self.assertFalse(output.exists())

    def test_output_and_coverage_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for output in (root.parent / 'outside', root / 'outputs', root / 'outputs/runs/overwrite'):
                with self.assertRaises(group.ReferenceGroupError):
                    group.output_directory(root, output)
            report = {'cells': [{'method': 'eagle3', 'benchmark': 'mt_bench', 'temperature': 0, 'pairing': {'pairing_complete': False}}]}
            with self.assertRaisesRegex(group.ReferenceGroupError, 'unpaired'):
                group.complete_coverage(report, {('eagle3', 'mt_bench', 0.0)})
            with self.assertRaisesRegex(group.ReferenceGroupError, 'assigned panel'):
                group.complete_coverage(report, set())

    def test_duplicate_run_ids_and_unsafe_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(group.ReferenceGroupError, 'unique'):
                group.collect(root, ['same', 'same'], root / 'outputs/group')
            with self.assertRaisesRegex(group.ReferenceGroupError, 'Invalid run ID'):
                group.run_documents(root, '../escape')


if __name__ == '__main__':
    unittest.main()
