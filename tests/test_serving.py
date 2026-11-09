import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from taps_rebuild import serving


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == '/health' else 404)
        self.end_headers()

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        server = self.server
        with server.lock:
            server.active += 1
            server.maximum_active = max(server.maximum_active, server.active)
            server.requests.append(request)
        try:
            if server.behavior == 'timeout':
                time.sleep(0.08)
            else:
                time.sleep(0.004)
            if server.behavior == 'http_error':
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"error":"busy"}')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream' if request['stream'] else 'application/json')
            self.end_headers()
            native = self.path == '/generate'
            if not request['stream']:
                if native:
                    body = {'text': 'α' * 100, 'meta_info': {'completion_tokens': 2, 'prompt_tokens': 3, 'finish_reason': {'type': 'length'}}}
                else:
                    body = {'model': request['model'], 'choices': [{'index': 0, 'text': 'α' * 100, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}
                self.wfile.write(json.dumps(body).encode())
                return
            if server.behavior == 'malformed':
                self.wfile.write(b'data: invalid-json\n\n')
                return
            if native:
                events = [{'text': 'alpha', 'meta_info': {'completion_tokens': 1, 'prompt_tokens': 3, 'finish_reason': None}}, {'text': 'alpha beta', 'meta_info': {'completion_tokens': 2, 'prompt_tokens': 3, 'finish_reason': {'type': 'length'}}}]
            else:
                first = {'model': request['model'], 'choices': [{'index': 0, 'text': 'alpha', 'finish_reason': None, 'logprobs': {'tokens': ['alpha']}}]}
                second = {'model': request['model'], 'choices': [{'index': 0, 'text': ' beta', 'finish_reason': 'length', 'logprobs': {'tokens': [' beta']}}]}
                if server.behavior == 'without_logprobs':
                    first['choices'][0].pop('logprobs')
                    second['choices'][0].pop('logprobs')
                events = [first, second]
                if server.behavior == 'coalesced':
                    events = [{'model': request['model'], 'choices': [{'index': 0, 'text': 'alpha beta', 'finish_reason': 'length', 'logprobs': {'tokens': ['alpha', ' beta']}}]}]
                if server.behavior != 'no_usage':
                    events.append({'model': request['model'], 'choices': [], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}})
            for event in events:
                self.wfile.write(b'data: ' + json.dumps(event).encode() + b'\n\n')
                self.wfile.flush()
                time.sleep(0.004)
            if server.behavior != 'no_done':
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with server.lock:
                server.active -= 1


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / 'project'
        self.root.mkdir()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.lock = threading.Lock()
        self.server.active = 0
        self.server.maximum_active = 0
        self.server.requests = []
        self.server.behavior = 'stream'
        thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base_url = 'http://127.0.0.1:' + str(self.server.server_port)
        self.config = json.loads((PROJECT / 'configs/serving.json').read_bytes())
        self.config.update(cluster_root=str(self.root), legacy_runtime_root=str(self.root / 'legacy'), concurrencies=[1, 3], requests_per_repeat=4, max_tokens=2, request_timeout_seconds=2)
        self.config['server']['readiness_timeout_seconds'] = 0.5
        self.config['methods'] = {name: self.config['methods'][name] for name in ('vllm_ar', 'vllm_eagle3')}
        for method in self.config['methods'].values():
            method['endpoint'] = self.base_url + '/v1/completions'
        self.method = self.config['methods']['vllm_ar']
        environment = patch.dict(os.environ, {'SLURM_JOB_ID': 'test-cpu', 'CUDA_VISIBLE_DEVICES': ''})
        environment.start()
        self.addCleanup(environment.stop)

    def request(self, protocol='openai_completions'):
        method = dict(self.method, protocol=protocol)
        if protocol == 'sglang_generate':
            method.update(engine='sglang', endpoint=self.base_url + '/generate')
        return serving.perform_request(method, 'literal prompt', self.config, 17)

    def project_fixture(self, provenance=True):
        configuration = self.root / 'configs/serving.json'
        configuration.parent.mkdir()
        configuration.write_text(json.dumps(self.config))
        (self.root / 'configs/assets.json').write_text('{}')
        source = self.root / 'src/taps_rebuild/assets.py'
        source.parent.mkdir(parents=True)
        source.write_text('fixture-source')
        models = {}
        for name in ('llama31_instruct_8b', 'eagle3_llama31_8b'):
            models[name] = {'path': str(self.root / 'models' / name), 'model_identity_sha256': hashlib.sha256(name.encode()).hexdigest(), 'source': {'repo_id': 'public/' + name, 'revision': 'a' * 40}, 'files': {}}
        receipt = self.root / 'outputs/preparation/assets.json'
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({'status': 'complete', 'root': str(self.root), 'registry_sha256': serving.digest(self.root / 'configs/assets.json'), 'source_sha256': serving.digest(source), 'models': models}))
        prompts = self.root / self.config['prompt_stream']['path']
        prompts.parent.mkdir(parents=True)
        rows = [{'id': index, 'domain': domain, 'prompt': 'prompt ' + str(index)} for index, domain in enumerate(('chat', 'math', 'code', 'chat'))]
        prompts.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        if provenance:
            for name, method in self.config['methods'].items():
                checkpoints, _ = serving.checkpoint_records(self.root, method)
                _, _, identity = serving.server_command(self.root, self.config, name, checkpoints)
                directory = self.root / 'outputs/serving/servers'
                directory.mkdir(parents=True, exist_ok=True)
                (directory / (name + '-mock.json')).write_text(json.dumps({'status': 'launched', 'server_identity_sha256': identity, 'runtime': {'runtime_verified': True}}))
        return configuration

    def test_openai_stream_uses_final_usage_and_real_event_timestamps(self):
        result = self.request()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['output_tokens'], 2)
        self.assertEqual(result['output_text'], 'alpha beta')
        self.assertIsNotNone(result['ttft_seconds'])
        self.assertEqual(len(result['itl_seconds']), 1)
        self.assertGreaterEqual(result['itl_seconds'][0], 0)
        self.assertEqual(result['request']['seed'], 17)
        self.assertEqual(result['stream_events'][-1]['data'], '[DONE]')
        self.assertFalse(result['server_metrics']['memory_bytes']['available'])

    def test_coalesced_or_unlabelled_events_do_not_invent_itl(self):
        for behavior in ('coalesced', 'without_logprobs'):
            self.server.behavior = behavior
            with self.subTest(behavior=behavior):
                result = self.request()
                self.assertTrue(result['eligible'])
                self.assertIsNotNone(result['ttft_seconds'])
                self.assertIsNone(result['itl_seconds'])
                self.assertEqual(result['output_tokens'], 2)

    def test_native_sglang_uses_cumulative_counts_without_unsupported_seed(self):
        result = self.request('sglang_generate')
        self.assertTrue(result['eligible'])
        self.assertEqual(result['output_tokens'], 2)
        self.assertEqual(result['output_text'], 'alpha beta')
        self.assertIsNotNone(result['itl_seconds'])
        self.assertFalse(result['per_request_seed_applied'])
        self.assertNotIn('seed', result['request']['sampling_params'])

    def test_nonstream_counts_are_never_estimated_from_unicode_bytes(self):
        self.config['stream'] = False
        result = self.request()
        self.assertTrue(result['eligible'])
        self.assertEqual(result['output_tokens'], 2)
        self.assertEqual(len(result['output_text'].encode()), 200)
        self.assertIsNone(result['ttft_seconds'])
        self.assertIsNone(result['itl_seconds'])

    def test_errors_timeouts_missing_usage_and_incomplete_stream_are_ineligible(self):
        for behavior in ('no_usage', 'no_done', 'http_error', 'malformed', 'timeout'):
            self.server.behavior = behavior
            self.config['request_timeout_seconds'] = 0.01 if behavior == 'timeout' else 2
            with self.subTest(behavior=behavior):
                result = self.request()
                self.assertFalse(result['eligible'])
                self.assertNotEqual(result['status'], 'completed')
        self.assertTrue(result['ineligibility_reasons'])

    def test_fixed_output_budget_must_be_observed_in_usage(self):
        self.config['max_tokens'] = 3
        result = self.request()
        self.assertFalse(result['eligible'])
        self.assertIn('fixed_output_budget_not_met', result['ineligibility_reasons'])

    def test_sse_chunk_boundaries_multiline_and_incomplete_events(self):
        chunks = [b': keepalive\nda', b'ta: {"a":\n', b'data: 1}\r\n\r\ndata: [DONE]\n\n']
        self.assertEqual(list(serving.sse_messages(chunks)), ['{"a":\n1}', '[DONE]'])
        with self.assertRaises(serving.ServingError):
            list(serving.sse_messages([b'data: {"a":1}']))

    def test_only_actually_provided_server_metric_fields_are_aggregated(self):
        result = serving.metric_capabilities([{'meta_info': {'draft_time': 0.125, 'memory': 1234}}], {'drafting_seconds': 'meta_info.draft_time', 'memory_bytes': 'meta_info.memory'})
        self.assertEqual(result['drafting_seconds']['value'], 0.125)
        self.assertEqual(result['memory_bytes']['value'], 1234)
        self.assertFalse(result['verification_seconds']['available'])

    def test_controlled_shift_and_natural_prompt_order_are_deterministic(self):
        rows = [{'id': 'c', 'domain': 'chat'}, {'id': 'm', 'domain': 'math'}, {'id': 'x', 'domain': 'code'}]
        natural = serving.prompt_schedule(rows, {'mode': 'natural'}, 5)
        self.assertEqual([row['id'] for row in natural], ['c', 'm', 'x', 'c', 'm'])
        shifted = serving.prompt_schedule(rows, {'mode': 'controlled_shift', 'domain_schedule': [{'domain': 'chat', 'requests': 2}, {'domain': 'math', 'requests': 1}, {'domain': 'code', 'requests': 1}]}, 6)
        self.assertEqual([row['id'] for row in shifted], ['c', 'c', 'm', 'x', 'c', 'c'])

    def test_functional_load_runner_pairs_requests_excludes_warmups_and_uses_concurrency(self):
        path = self.project_fixture()
        result = serving.run_load(self.root, path, 'load-test')
        self.assertEqual(len(result['cells']), 4)
        self.assertTrue(all(cell['eligible'] for cell in result['cells']))
        self.assertTrue(all(cell['completed_requests'] == 20 and cell['output_tokens_from_usage'] == 40 for cell in result['cells']))
        self.assertTrue(all(pair['eligible'] and pair['actual_prompt_and_seed_pairs_match'] for pair in result['paired_comparisons']))
        self.assertGreaterEqual(self.server.maximum_active, 2)
        self.assertFalse(result['automatic_pipeline_gate_pass'])
        run = self.root / 'outputs/serving/load-test'
        records = [json.loads(line) for line in (run / 'requests.jsonl').read_text().splitlines()]
        self.assertEqual(sum(row.get('warmup') is True for row in records), 12)
        self.assertTrue(all(row['raw_log_writing_excluded'] for row in records if row.get('record_type') == 'repeat_wall'))
        before = serving.digest(run / 'requests.jsonl')
        with self.assertRaises(FileExistsError):
            serving.run_load(self.root, path, 'load-test')
        self.assertEqual(serving.digest(run / 'requests.jsonl'), before)
        row = next(row for row in records if row.get('method') == 'vllm_eagle3' and row.get('warmup') is False)
        row['request']['prompt'] = 'changed actual prompt'
        (run / 'requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
        changed = serving.summarize_run(run)
        pair = next(pair for pair in changed['paired_comparisons'] if pair['offered_concurrency'] == row['offered_concurrency'])
        self.assertFalse(pair['eligible'])
        self.assertIsNone(pair['observed_output_token_throughput_ratio'])

    def test_unverified_server_and_smoke_cannot_produce_eligible_speedups(self):
        path = self.project_fixture(provenance=False)
        result = serving.run_load(self.root, path, 'smoke-test', smoke=True)
        self.assertTrue(all(not cell['eligible'] for cell in result['cells']))
        self.assertTrue(all(pair['observed_output_token_throughput_ratio'] is None for pair in result['paired_comparisons']))
        self.assertEqual(result['scope'], 'smoke')

    def test_stock_server_commands_use_isolated_python_and_known_flags(self):
        self.project_fixture()
        checkpoints, _ = serving.checkpoint_records(self.root, self.config['methods']['vllm_eagle3'])
        command, environment, _ = serving.server_command(self.root, self.config, 'vllm_eagle3', checkpoints)
        self.assertIn('--speculative-config', command)
        self.assertEqual(environment['VLLM_USE_V1'], '1')
        speculative = json.loads(command[command.index('--speculative-config') + 1])
        self.assertEqual(speculative['method'], 'eagle3')
        self.assertEqual(speculative['num_speculative_tokens'], 3)
        self.config['methods']['vllm_eagle3']['tensor_parallel_size'] = 2
        with self.assertRaisesRegex(serving.ServingError, 'distributed inference'):
            serving.server_command(self.root, self.config, 'vllm_eagle3', checkpoints)
        self.config['engines']['vllm']['serving_python'] = str(self.root / 'legacy/envs/taps-eagle/bin/python')
        with self.assertRaisesRegex(serving.ServingError, 'separate environment'):
            serving.server_command(self.root, self.config, 'vllm_ar', checkpoints)

    def test_separate_virtualenvs_can_share_the_same_base_python_binary(self):
        self.project_fixture()
        base = self.root / 'python/bin/python'
        base.parent.mkdir(parents=True)
        base.write_text('base-interpreter-fixture')
        legacy = self.root / 'legacy/envs/taps-eagle/bin/python'
        legacy.parent.mkdir(parents=True)
        legacy.symlink_to(base)
        separate = self.root / self.config['engines']['vllm']['serving_python']
        separate.parent.mkdir(parents=True)
        separate.symlink_to(base)
        self.assertEqual(legacy.resolve(), separate.resolve())
        checkpoints, _ = serving.checkpoint_records(self.root, self.method)
        command, _, _ = serving.server_command(self.root, self.config, 'vllm_ar', checkpoints)
        self.assertEqual(command[0], str(separate))

    def test_live_allocation_graph_is_required_before_server_work(self):
        query = Mock()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(serving.ServingError):
            serving.allocation_resources(self.method, query)
        query.assert_not_called()
        job = f'JobId=42 UserId=user({os.getuid()}) JobState=RUNNING NodeList=deep-chungus-10 NumNodes=1 AllocTRES=cpu=64,gres/gpu=8,gres/gpu:a100=8'
        inventory = '0, GPU-a, NVIDIA A100-SXM4-80GB, 565.57.01\n1, GPU-b, NVIDIA A100-SXM4-80GB, 565.57.01\n'
        with patch.dict(os.environ, {'SLURM_JOB_ID': '42', 'SLURMD_NODENAME': 'deep-chungus-10', 'CUDA_VISIBLE_DEVICES': '0', 'SLURM_JOB_GPUS': '0-7', 'SLURM_STEP_GPUS': ''}), patch.object(serving.attach, 'command_output', side_effect=[job, inventory]):
            resource = serving.allocation_resources(self.method)
        self.assertEqual(resource['hardware'][0]['driver_version'], '565.57.01')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '42', 'SLURMD_NODENAME': 'deep-chungus-10', 'CUDA_VISIBLE_DEVICES': '1', 'SLURM_JOB_GPUS': '0', 'SLURM_STEP_GPUS': ''}), patch.object(serving.attach, 'command_output', side_effect=[job, inventory]), self.assertRaises(serving.ServingError):
            serving.allocation_resources(self.method)

    def test_runtime_version_and_cli_capabilities_are_checked_without_installation(self):
        python = self.root / 'isolated/bin/python'
        python.parent.mkdir(parents=True)
        python.write_text('fixture')
        python.chmod(0o755)
        command = [str(python), '-m', 'vllm.entrypoints.openai.api_server', '--model', 'prepared', '--speculative-config', '{}']
        metadata = {'packages': {'vllm': '0.8.5.post1', 'torch': '2.6.0+cu124', 'transformers': '4.51.1'}, 'torch_cuda': '12.4', 'prefix': str(python.parent.parent), 'base_prefix': str(self.root / 'base-python'), 'source_file_sha256': dict(serving.PROFILES['vllm']['source_files'])}
        responses = [subprocess.CompletedProcess([], 0, stdout=json.dumps(metadata), stderr=''), subprocess.CompletedProcess([], 0, stdout='--model --speculative-config', stderr='')]
        with patch.object(serving.subprocess, 'run', side_effect=responses) as run:
            result = serving.runtime_check(command, 'vllm', {})
        self.assertTrue(result['runtime_verified'])
        self.assertEqual(run.call_count, 2)
        metadata['torch_cuda'] = '12.8'
        with patch.object(serving.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(metadata), stderr='')), self.assertRaises(serving.ServingError):
            serving.runtime_check(command, 'vllm', {})

    def test_cpu_config_and_endpoint_guards_prevent_unintended_load(self):
        for environment in ({}, {'SLURM_JOB_ID': 'job'}, {'SLURM_JOB_ID': 'job', 'CUDA_VISIBLE_DEVICES': '0'}):
            with patch.dict(os.environ, environment, clear=True), self.assertRaises(serving.ServingError):
                serving.run_load(self.root, self.root / 'absent.json', 'run')
        for endpoint in ('http://user:password@host/v1/completions', 'http://host/v1/completions?token=x', 'file:///tmp/data'):
            with self.subTest(endpoint=endpoint), self.assertRaises(serving.ServingError):
                serving.endpoint(endpoint, 'openai_completions')
        path = self.project_fixture()
        self.config['measured_repeats'] = 4
        path.write_text(json.dumps(self.config))
        with self.assertRaises(serving.ServingError):
            serving.read_config(self.root, path)


if __name__ == '__main__':
    unittest.main()
