import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

from . import attach, pool, runtime, serving


MODULES = ['__init__.py', 'serving_pool.py', 'serving.py', 'pool.py', 'attach.py', 'assets.py', 'sources.py', 'runtime.py']


class SmokeError(ValueError):
    pass


def effective_config(config, run_id, bindings, base_port):
    if len(bindings) != 8 or len({item['uuid'] for item in bindings}) != 8 or len({item['token'] for item in bindings}) != 8:
        raise SmokeError('Serving smoke requires eight distinct physical GPUs')
    if type(base_port) is not int or not 1024 <= base_port <= 65528:
        raise SmokeError('Reserve eight consecutive unprivileged loopback ports')
    value = copy.deepcopy(config)
    value.update(max_tokens=16, request_timeout_seconds=120, concurrencies=[8], requests_per_repeat=8, measured_repeats=5, warmup_requests=3, temperature=0.0, ignore_eos=True, stream=True)
    value['server'].update(max_model_len=1024, max_running_requests=8, memory_fraction=0.8)
    methods = {}
    assignments = []
    for rank, binding in enumerate(bindings):
        mode = 'target_only' if rank < 4 else 'eagle3'
        original = 'vllm_ar' if mode == 'target_only' else 'vllm_eagle3'
        name = 'smoke-' + run_id + '-gpu' + str(rank)
        method = copy.deepcopy(config['methods'][original])
        method.update(engine='vllm', mode=mode, protocol='openai_completions', endpoint='http://127.0.0.1:' + str(base_port + rank) + '/v1/completions', bind_host='127.0.0.1', tensor_parallel_size=1)
        methods[name] = method
        assignments.append({'rank': rank, 'method': name, 'mode': mode, 'binding': binding, 'port': base_port + rank})
    value.update(methods=methods, method_order=list(methods))
    return value, assignments


def available_ports(assignments):
    held = []
    try:
        for item in assignments:
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            held.append(handle)
            handle.bind(('127.0.0.1', item['port']))
    except OSError as error:
        raise SmokeError('A requested loopback port is occupied') from error
    finally:
        for handle in held:
            handle.close()


def compiled_cache(root, run_id, hardware):
    label = 's' + pool.identity(run_id)[:8]
    requested = Path('/tmp') / ('taps-rebuild-vllm-' + str(os.getuid()) + '-' + hardware['job_id'] + '-' + hardware['node'] + '-' + label)
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        path, proof = runtime.cache_directory(root, requested)
    finally:
        if previous is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    record = {'proof': proof, 'path': str(path), 'run_id': run_id, 'role': 'Disposable per-run compiled GPU kernels and temporary files; weights, environment and evidence remain persistent', 'gpu_paths': {}}
    for binding in hardware['device_bindings']:
        rank = binding['shard_index']
        directory = path / ('g' + str(rank))
        if len(os.fsencode(str(directory / '12345678-1234-1234-1234-123456789abc'))) > 106:
            raise SmokeError('Run-local temporary path exceeds the Unix socket filename budget')
        for name in ('xdg', 'inductor', 'triton', 'cuda'):
            (directory / name).mkdir(parents=True)
        record['gpu_paths'][str(rank)] = {'root': str(directory), 'xdg': str(directory / 'xdg'), 'inductor': str(directory / 'inductor'), 'triton': str(directory / 'triton'), 'cuda': str(directory / 'cuda'), 'tmp': str(directory)}
    return record


def freeze(root, directory, config, assignments, hardware, cache):
    snapshot = directory / 'source/src/taps_rebuild'
    snapshot.mkdir(parents=True)
    hashes = {}
    for name in MODULES:
        original = root / 'src/taps_rebuild' / name
        if original.is_symlink() or not original.is_file():
            raise SmokeError('Serving smoke source must be regular files')
        destination = snapshot / name
        destination.write_bytes(original.read_bytes())
        hashes[name] = serving.digest(destination)
    config_path = directory / 'config.json'
    pool.atomic_json(config_path, config)
    receipt_path = root / 'outputs/preparation/runtime-vllm/runtime.json'
    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get('status') != 'complete' or receipt.get('identity', {}).get('serving_source_sha256') != serving.digest(root / 'src/taps_rebuild/serving.py') or receipt['identity']['requirements_sha256'] != serving.digest(root / 'configs/runtime-vllm.txt') or receipt['identity']['serving_config_sha256'] != serving.digest(root / 'configs/serving.json'):
        raise SmokeError('A matching complete CPU runtime preparation receipt is required')
    pool.atomic_json(directory / 'runtime-preparation.json', receipt)
    checkpoints, asset_hash = serving.checkpoint_records(root, config['methods'][assignments[4]['method']], verify=False)
    manifest = {'scope': 'stock_serving_engineering_smoke', 'scientific_result': False, 'performance_comparison_eligible': False, 'full_model_distributional_correctness_established': False, 'automatic_pipeline_gate_pass': False, 'source_sha256': hashes, 'config_sha256': serving.digest(config_path), 'runtime_receipt_sha256': serving.digest(receipt_path), 'asset_receipt_sha256': asset_hash, 'checkpoints': checkpoints, 'assignments': assignments, 'hardware': hardware, 'operational_compiled_cache': cache, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'http_workload': {'warmups_per_server': 3, 'measured_repeats': 5, 'requests_per_repeat': 8, 'offered_concurrency': 8, 'fixed_output_tokens': 16, 'prompts': 'Authored mechanical smoke prompts; not held-out traffic'}, 'batching_scope': 'Eight concurrent HTTP requests exercise each stock server; internal engine batch size is not observed', 'server_component_metrics': 'Unavailable unless actually provided by the server response'}
    pool.atomic_json(directory / 'manifest.json', manifest)
    return manifest


def owned_members(group, proc_root=Path('/proc')):
    result = []
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            parts = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(parts[2]) != group['pid']:
                continue
            if path.stat().st_uid != os.getuid() or int(parts[3]) != group['pid'] or int(parts[19]) < int(group['starttime_ticks']):
                raise SmokeError('Controller-owned process group identity no longer matches')
            if int(path.name) == group['pid'] and parts[19] != group['starttime_ticks']:
                raise SmokeError('Controller child PID identity changed')
            result.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def cleanup(children, grace=15):
    groups = []
    for child, group in children:
        if group is None:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            continue
        if group.get('uid') != os.getuid() or not group.get('starttime_ticks') or group['pid'] != child.pid:
            raise SmokeError('Refusing to signal an unverified process group')
        if owned_members(group):
            try:
                os.killpg(group['pid'], signal.SIGTERM)
            except ProcessLookupError:
                continue
            groups.append(group)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(owned_members(group) for group in groups):
        for child, _ in children:
            child.poll()
        time.sleep(0.1)
    for group in groups:
        if owned_members(group):
            try:
                os.killpg(group['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
    for child, _ in children:
        child.wait(timeout=10)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(owned_members(group) for group in groups):
        time.sleep(0.1)
    if any(owned_members(group) for group in groups):
        raise SmokeError('Some controller-owned server descendants did not exit')
    return {'status': 'complete', 'process_groups': [group['pid'] for group in groups], 'slurm_jobs_or_other_process_groups_signalled': False}


def ready(method, child, timeout):
    parsed = serving.endpoint(method['endpoint'], method['protocol'])
    url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, '/health', '', ''))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise SmokeError('A controller-owned stock server exited before readiness')
        try:
            with urllib.request.build_opener(serving.NoRedirect()).open(url, timeout=min(2, max(0.01, deadline - time.monotonic()))) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError) as error:
            if isinstance(error, urllib.error.HTTPError):
                error.close()
        time.sleep(0.25)
    raise SmokeError('Stock server readiness deadline exceeded')


def exercise(directory, assignment, config, child):
    name = assignment['method']
    method = config['methods'][name]
    output = directory / ('gpu-' + str(assignment['rank']))
    output.mkdir()
    ready(method, child, config['server']['readiness_timeout_seconds'])
    root = Path(config['cluster_root']).resolve()
    checkpoints, _ = serving.checkpoint_records(root, method, verify=False)
    provenance = serving.server_provenance(root, config, name, checkpoints)
    if not provenance['matched']:
        raise SmokeError('Actual stock server lacks matching verified launcher provenance')
    launcher = json.loads(Path(provenance['receipt']).read_bytes())
    pool.atomic_json(output / 'server-launch.json', launcher)
    prompts = config['smoke_prompts']
    if len(prompts) < 3:
        raise SmokeError('Three explicit smoke prompts are required')
    measured = []
    walls = []
    with (output / 'requests.jsonl').open('x') as raw, (output / 'stream-events.jsonl').open('x') as events:
        def record(result, warmup, repeat, index):
            result.update(method=name, warmup=warmup, repeat=repeat, request_index=index, prompt_id=prompts[index % len(prompts)]['id'], offered_concurrency=1 if warmup else 8)
            for event in result.pop('stream_events', []):
                serving.append_json(events, {'warmup': warmup, 'repeat': repeat, 'request_index': index, **event})
            serving.append_json(raw, result)
            if not result.get('eligible') or result.get('output_tokens') != 16:
                raise SmokeError('Stock model response failed the streaming, usage or fixed-output smoke check')
        for index in range(3):
            result = serving.perform_request(method, prompts[index]['prompt'], config, config['seed'] + index)
            record(result, True, -1, index)
        with ThreadPoolExecutor(max_workers=8) as executor:
            for repeat in range(5):
                if child.poll() is not None:
                    raise SmokeError('A controller-owned stock server exited during smoke requests')
                started = time.monotonic()
                futures = {executor.submit(serving.perform_request, method, prompts[index % len(prompts)]['prompt'], config, config['seed'] + repeat * 8 + index): index for index in range(8)}
                results = [(futures[future], future.result()) for future in as_completed(futures)]
                wall = max(result['request_finish_monotonic'] for _, result in results) - started
                walls.append(wall)
                for index, result in results:
                    record(result, False, repeat, index)
                    measured.append(result)
                serving.append_json(raw, {'record_type': 'repeat_wall', 'repeat': repeat, 'wall_seconds': wall, 'raw_log_writing_excluded': True})
    summary = serving.summarize_cell(measured, sum(walls), 40, True, True, True)
    if summary['observed_peak_inflight_requests'] < 2:
        raise SmokeError('Concurrent HTTP requests were not observed')
    result = {'status': 'passed', 'scope': 'engineering_smoke_not_performance_benchmark', 'assignment': assignment, 'summary': summary, 'stream_events_path': str(output / 'stream-events.jsonl'), 'requests_path': str(output / 'requests.jsonl'), 'actual_output_tokens': summary['output_tokens_from_usage'], 'server_provenance': provenance, 'launcher_receipt_sha256': serving.digest(output / 'server-launch.json'), 'model_loading_verified_by_actual_responses': True, 'engine_internal_batch_size': None, 'server_draft_verify_memory_counters': 'unavailable unless returned in actual response'}
    pool.atomic_json(output / 'result.json', result)
    return result


def worker(root, directory, rank):
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    for name, expected in manifest['source_sha256'].items():
        if serving.digest(directory / 'source/src/taps_rebuild' / name) != expected:
            raise SmokeError('Frozen serving smoke source differs')
    if serving.digest(directory / 'config.json') != manifest['config_sha256']:
        raise SmokeError('Frozen serving smoke configuration differs')
    assignment = manifest['assignments'][rank]
    if assignment['rank'] != rank or os.environ.get('CUDA_VISIBLE_DEVICES') != assignment['binding']['token']:
        raise SmokeError('Worker GPU binding differs from the frozen manifest')
    gate = directory / ('start-gpu-' + str(rank) + '.json')
    deadline = time.monotonic() + 30
    while not gate.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not gate.exists() or json.loads(gate.read_bytes()) != attach.process_identity(os.getpid()):
        raise SmokeError('Controller did not verify this worker before its GPU launch')
    config, _, _ = serving.read_config(root, directory / 'config.json')
    result = serving.launch_server(root, config, assignment['method'])
    if result.get('returncode', 0) != 0:
        raise SmokeError('Stock server failed; inspect its launcher receipt and log')


def controller(root, config_path, run_id, expected_gpus=8, base_port=18000):
    if expected_gpus != 8 or type(expected_gpus) is not int or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', run_id):
        raise SmokeError('Use all eight GPUs and a new bounded run ID')
    root = Path(root).resolve()
    hardware = pool.allocation(8)
    if any('A100' not in item['name'] for item in hardware['gpu_inventory']):
        raise SmokeError('This serving smoke graph is configured for eight A100 GPUs')
    active = pool._command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']).strip()
    if active:
        raise SmokeError('Existing GPU compute processes must finish before this eight-GPU smoke starts')
    config, _, _ = serving.read_config(root, config_path)
    effective, assignments = effective_config(config, run_id, hardware['device_bindings'], base_port)
    configured_python = Path(config['engines']['vllm']['serving_python'])
    configured_python = configured_python if configured_python.is_absolute() else root / configured_python
    actual_python = runtime.physical_python(configured_python)
    if configured_python.parent.parent.resolve() != actual_python.parent.parent:
        raise SmokeError('Configured and executed server environments no longer identify the same prefix')
    effective['engines']['vllm']['serving_python'] = str(actual_python)
    effective['runtime_execution'] = {'configured_python': str(configured_python), 'executed_python': str(actual_python), 'same_resolved_venv_prefix': configured_python.parent.parent.resolve() == actual_python.parent.parent}
    available_ports(assignments)
    directory = root / 'outputs/serving-pools' / run_id
    directory.mkdir(parents=True, exist_ok=False)
    children = []
    logs = []
    status = {'status': 'starting', 'scope': 'engineering_smoke_not_scientific_results', 'run_id': run_id, 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    previous_handlers = {}
    with (directory / '.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = compiled_cache(root, run_id, hardware)
        status['operational_compiled_cache'] = cache
        manifest = freeze(root, directory, effective, assignments, hardware, cache)
        pool.atomic_json(directory / 'state.json', status)
        try:
            def interrupted(signum, frame):
                raise InterruptedError('Serving smoke controller interrupted')
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, interrupted)
            for item in assignments:
                paths = cache['gpu_paths'][str(item['rank'])]
                environment = pool.worker_environment(item['binding']['token'], directory / 'source', root, Path('/usr/bin/python3'))
                environment['PATH'] = str(actual_python.parent) + os.pathsep + environment.get('PATH', '/usr/bin:/bin')
                environment.update(SLURM_STEP_GPUS=item['binding']['index'], SLURM_JOB_GPUS=','.join(binding['index'] for binding in hardware['device_bindings']), CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONNOUSERSITE='1', XDG_CACHE_HOME=paths['xdg'], TORCHINDUCTOR_CACHE_DIR=paths['inductor'], TRITON_CACHE_DIR=paths['triton'], CUDA_CACHE_PATH=paths['cuda'], TMPDIR=paths['tmp'], MAX_JOBS='4', VLLM_WORKER_MULTIPROC_METHOD='spawn')
                stream = (directory / ('launcher-gpu-' + str(item['rank']) + '.log')).open('xb')
                logs.append(stream)
                command = ['/usr/bin/python3', '-B', '-m', 'taps_rebuild.serving_pool', 'worker', '--root', str(root), '--directory', str(directory), '--rank', str(item['rank'])]
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                children.append((process, None))
                group = attach.process_identity(process.pid)
                if os.getpgid(process.pid) != process.pid or group.get('uid') != os.getuid() or not group.get('starttime_ticks'):
                    raise SmokeError('New stock server wrapper did not establish its own process group')
                children[-1] = (process, group)
                pool.atomic_json(directory / ('start-gpu-' + str(item['rank']) + '.json'), group)
            status.update(status='running', children=[identity for _, identity in children])
            pool.atomic_json(directory / 'state.json', status)
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(exercise, directory, assignment, effective, children[assignment['rank']][0]) for assignment in assignments]
                results = [future.result() for future in as_completed(futures)]
            status.update(status='passed', workers=sorted(results, key=lambda result: result['assignment']['rank']), model_loading_verified_by_actual_responses=True, performance_comparison_eligible=False)
        except BaseException as error:
            status.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            try:
                status['cleanup'] = cleanup(children)
            except BaseException as error:
                status.update(status='failed', cleanup={'status': 'failed', 'error_type': type(error).__name__})
                raise
            finally:
                for stream in logs:
                    stream.close()
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
                status['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
                pool.atomic_json(directory / 'state.json', status)
    return directory, status


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', nargs='?', choices=('controller', 'worker'), default='controller')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/serving.json'))
    parser.add_argument('--run-id')
    parser.add_argument('--expected-gpus', type=int, default=8)
    parser.add_argument('--base-port', type=int, default=18000)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--rank', type=int)
    arguments = parser.parse_args(argv)
    if arguments.action == 'worker':
        if not arguments.directory or arguments.rank is None or not 0 <= arguments.rank < 8:
            raise SmokeError('Worker mode requires the controller directory and rank')
        worker(arguments.root.resolve(), arguments.directory.resolve(), arguments.rank)
    else:
        directory, result = controller(arguments.root, arguments.config, arguments.run_id or '', arguments.expected_gpus, arguments.base_port)
        print(json.dumps({'status': result['status'], 'directory': str(directory), 'scope': result['scope']}), flush=True)


if __name__ == '__main__':
    main()
