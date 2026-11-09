import json
import os
from pathlib import Path
import signal
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from taps_rebuild import serving, serving_pool


class ServingPoolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = (Path(self.temporary.name) / 'project').resolve()
        self.root.mkdir()
        self.config = json.loads((Path(__file__).parents[1] / 'configs/serving.json').read_bytes())
        self.config['cluster_root'] = str(self.root)
        self.bindings = [{'rank': index, 'shard_index': index, 'index': str(index), 'token': str(index), 'uuid': 'GPU-' + str(index)} for index in range(8)]
        self.hardware = {'device_bindings': self.bindings, 'gpu_inventory': [{'name': 'NVIDIA A100', 'uuid': item['uuid']} for item in self.bindings], 'job_id': '1', 'node': 'node'}

    def test_assignment_uses_all_eight_gpus_once_and_four_servers_per_mode(self):
        config, assigned = serving_pool.effective_config(self.config, 'test', self.bindings, 18000)
        self.assertEqual([item['mode'] for item in assigned], ['target_only'] * 4 + ['eagle3'] * 4)
        self.assertEqual([item['port'] for item in assigned], list(range(18000, 18008)))
        self.assertEqual({item['binding']['uuid'] for item in assigned}, {item['uuid'] for item in self.bindings})
        self.assertTrue(all(method['tensor_parallel_size'] == 1 for method in config['methods'].values()))
        self.assertEqual(config['max_tokens'], 16)
        self.assertEqual(config['server']['max_running_requests'], 8)
        with self.assertRaises(serving_pool.SmokeError):
            serving_pool.effective_config(self.config, 'test', self.bindings[:7], 18000)
        with self.assertRaises(serving_pool.SmokeError):
            serving_pool.effective_config(self.config, 'test', self.bindings, 65529)

    def test_occupied_ports_are_not_reused(self):
        with socket.socket() as handle:
            handle.bind(('127.0.0.1', 0))
            with self.assertRaisesRegex(serving_pool.SmokeError, 'occupied'):
                serving_pool.available_ports([{'port': handle.getsockname()[1]}])

    def test_proc_group_guard_checks_session_uid_and_starttime(self):
        directory = self.root / 'proc/100'
        directory.mkdir(parents=True)
        parts = ['S', '1', '100', '100'] + ['0'] * 15 + ['500'] + ['0'] * 5
        (directory / 'stat').write_text('100 (name with spaces) ' + ' '.join(parts))
        group = {'pid': 100, 'uid': os.getuid(), 'starttime_ticks': '500'}
        self.assertEqual(serving_pool.owned_members(group, self.root / 'proc'), [100])
        with patch.object(os, 'getuid', return_value=os.getuid() + 1), self.assertRaises(serving_pool.SmokeError):
            serving_pool.owned_members(group, self.root / 'proc')
        parts[3] = '101'
        (directory / 'stat').write_text('100 (name) ' + ' '.join(parts))
        with self.assertRaises(serving_pool.SmokeError):
            serving_pool.owned_members(group, self.root / 'proc')

    def test_cleanup_signals_only_verified_groups_or_ungated_direct_children(self):
        child = Mock(pid=100)
        child.poll.return_value = None
        group = {'pid': 100, 'uid': os.getuid(), 'starttime_ticks': '500'}
        with patch.object(serving_pool, 'owned_members', side_effect=[[100], [], [], []]), patch.object(os, 'killpg') as kill:
            result = serving_pool.cleanup([(child, group)], grace=0)
        kill.assert_called_once_with(100, signal.SIGTERM)
        self.assertFalse(result['slurm_jobs_or_other_process_groups_signalled'])
        with patch.object(os, 'killpg') as kill:
            serving_pool.cleanup([(child, None)], grace=0)
        child.terminate.assert_called_once()
        kill.assert_not_called()
        group['uid'] = os.getuid() + 1
        with patch.object(os, 'killpg') as kill, self.assertRaises(serving_pool.SmokeError):
            serving_pool.cleanup([(child, group)])
        kill.assert_not_called()

    def test_health_failure_detects_exited_server(self):
        child = Mock()
        child.poll.return_value = 1
        with self.assertRaisesRegex(serving_pool.SmokeError, 'exited'):
            serving_pool.ready(self.config['methods']['vllm_ar'], child, 1)

    def test_smoke_requests_use_actual_usage_and_remain_ineligible_for_comparison(self):
        config, assigned = serving_pool.effective_config(self.config, 'test', self.bindings, 18000)
        launcher = self.root / 'launcher.json'
        launcher.write_text(json.dumps({'runtime': {'runtime_verified': True}}))
        counter = 0
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def request(method, prompt, settings, seed):
            nonlocal counter
            with lock:
                index = counter
                counter += 1
            started = time.monotonic()
            if index >= 3:
                barrier.wait(timeout=2)
                time.sleep(0.001)
            finished = time.monotonic()
            return {'eligible': True, 'status': 'completed', 'output_tokens': 16, 'usage': {'completion_tokens': 16}, 'latency_seconds': finished - started, 'request_start_monotonic': started, 'request_finish_monotonic': finished, 'ttft_seconds': 0.001, 'itl_seconds': None, 'stream_events': [{'offset_seconds': 0.001, 'data': '[DONE]'}], 'server_metrics': {}}

        child = Mock()
        child.poll.return_value = None
        with patch.object(serving_pool, 'ready'), patch.object(serving, 'checkpoint_records', return_value=({}, 'a' * 64)), patch.object(serving, 'server_provenance', return_value={'matched': True, 'receipt': str(launcher)}), patch.object(serving, 'perform_request', side_effect=request):
            result = serving_pool.exercise(self.root, assigned[0], config, child)
        self.assertEqual(counter, 43)
        self.assertEqual(result['actual_output_tokens'], 640)
        self.assertFalse(result['summary']['eligible'])
        self.assertEqual(result['summary']['observed_peak_inflight_requests'], 8)
        self.assertTrue(result['model_loading_verified_by_actual_responses'])
        records = [json.loads(line) for line in Path(result['requests_path']).read_text().splitlines()]
        self.assertEqual(sum(row.get('warmup', False) for row in records), 3)
        self.assertEqual(sum(row.get('record_type') == 'repeat_wall' for row in records), 5)

    def test_controller_launches_eight_isolated_groups_and_cleans_its_children(self):
        (self.root / 'configs').mkdir()
        (self.root / 'configs/serving.json').write_text(json.dumps(self.config))
        children = []

        def popen(command, **kwargs):
            child = Mock(pid=1000 + len(children))
            child.poll.return_value = None
            children.append((child, command, kwargs))
            return child

        def frozen(root, directory, config, assignments, hardware, cache):
            return {'assignments': assignments}

        def exercise(directory, assignment, config, child):
            return {'status': 'passed', 'assignment': assignment}

        cache = {'gpu_paths': {str(index): {name: str(self.root / 'local-cache' / str(index) / name) for name in ('root', 'xdg', 'inductor', 'triton', 'cuda', 'tmp')} for index in range(8)}}
        with patch.object(serving_pool.pool, 'allocation', return_value=self.hardware), patch.object(serving_pool.pool, '_command', return_value=''), patch.object(serving_pool, 'available_ports'), patch.object(serving_pool, 'compiled_cache', return_value=cache), patch.object(serving_pool.runtime, 'physical_python', side_effect=lambda value: value), patch.object(serving_pool, 'freeze', side_effect=frozen), patch.object(serving_pool.subprocess, 'Popen', side_effect=popen), patch.object(serving_pool.attach, 'process_identity', side_effect=lambda pid: {'pid': pid, 'uid': os.getuid(), 'starttime_ticks': '500'}), patch.object(serving_pool.os, 'getpgid', side_effect=lambda pid: pid), patch.object(serving_pool, 'exercise', side_effect=exercise), patch.object(serving_pool, 'cleanup', return_value={'status': 'complete'}) as cleaned:
            directory, result = serving_pool.controller(self.root, 'configs/serving.json', 'all-eight')
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(children), 8)
        self.assertEqual({value[2]['env']['CUDA_VISIBLE_DEVICES'] for value in children}, set(map(str, range(8))))
        for child, command, keywords in children:
            self.assertTrue(keywords['start_new_session'])
            self.assertEqual(keywords['cwd'], self.root)
            self.assertEqual(keywords['env']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
            self.assertEqual(keywords['env']['PATH'].split(os.pathsep)[0], str(self.root / 'envs/serving-vllm/bin'))
            self.assertEqual(keywords['env'].get('HOME'), os.environ.get('HOME'))
            for name in ('XDG_CACHE_HOME', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'TMPDIR'):
                self.assertTrue(keywords['env'][name].startswith(str(self.root / 'local-cache')))
            self.assertEqual(keywords['env']['MAX_JOBS'], '4')
            self.assertNotIn('sbatch', command)
            self.assertNotIn('scancel', command)
        self.assertEqual(len(cleaned.call_args.args[0]), 8)
        self.assertEqual(len(list(directory.glob('start-gpu-*.json'))), 8)

    def test_compiled_cache_preserves_cuda_binding_and_is_guarded_once_per_run(self):
        local = tempfile.TemporaryDirectory(prefix='taps-test-', dir='/tmp')
        self.addCleanup(local.cleanup)
        path = Path(local.name) / 'c'

        def guard(root, requested):
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(requested.parent, Path('/tmp'))
            self.assertTrue(requested.name.endswith('-s' + serving_pool.pool.identity('job.with.dots')[:8]))
            return path, {'kind': 'node_local_tmp', 'path': str(path), 'node': 'node', 'job_id': '1'}

        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}, clear=True), patch.object(serving_pool.runtime, 'cache_directory', side_effect=guard) as checked:
            result = serving_pool.compiled_cache(self.root, 'job.with.dots', self.hardware)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
        checked.assert_called_once()
        self.assertEqual(len(result['gpu_paths']), 8)
        self.assertTrue(all(Path(value['tmp']).is_dir() for value in result['gpu_paths'].values()))
        self.assertIn('runtime.py', serving_pool.MODULES)
        for binding in (None, '', '2,5'):
            environment = {} if binding is None else {'CUDA_VISIBLE_DEVICES': binding}
            with self.subTest(cuda_visible_devices=binding), patch.dict(os.environ, environment, clear=True):
                with patch.object(serving_pool.runtime, 'cache_directory', side_effect=ValueError('bad proof')), self.assertRaises(ValueError):
                    serving_pool.compiled_cache(self.root, 'different', self.hardware)
                self.assertEqual(dict(os.environ), environment)

    def test_real_cluster_ipc_path_budget_fits_uuid_socket_names(self):
        prefix = Path('/tmp/taps-rebuild-vllm-29562-2212576-deep-chungus-10-s12345678')
        for rank in range(8):
            path = prefix / ('g' + str(rank)) / '12345678-1234-1234-1234-123456789abc'
            self.assertLessEqual(len(os.fsencode(str(path))), 106)


if __name__ == '__main__':
    unittest.main()
