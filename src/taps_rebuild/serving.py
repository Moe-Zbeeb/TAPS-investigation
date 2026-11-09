import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

from . import assets, attach, sources


PROFILES = {
    'vllm': {'version': '0.8.5.post1', 'torch': '2.6.0', 'transformers': '4.51.1', 'cuda': '12.4', 'module': 'vllm.entrypoints.openai.api_server', 'per_request_seed': True},
    'sglang': {'version': '0.4.6.post5', 'torch': '2.6.0', 'transformers': '4.51.1', 'cuda': '12.4', 'module': 'sglang.launch_server', 'per_request_seed': False},
}
PROFILES['vllm']['source_files'] = {
    'vllm/v1/spec_decode/eagle.py': '6018c7ee359dcdde90d8228c647a6944534a599be24d7beebafc88c4dd7db5dd',
    'vllm/model_executor/models/llama_eagle3.py': 'a533d7eebb793bd60a92ddcc3de3a02986dbf3b05c2d638c3ee323427d6a8a72',
    'vllm/entrypoints/openai/protocol.py': '41035c9a6fb42af1e78049511008f29ff0009bf97ca94284fe774d78a9929937',
}
PROFILES['sglang']['source_files'] = {
    'sglang/srt/speculative/eagle_worker.py': 'da4eb281b909d452b5237bc9369bc84909fc01cde55b78b7c4546bd59ba02326',
    'sglang/srt/models/llama_eagle3.py': '282bd4ad69f5eeded3dbc1bad07c0ccb2f7d881f406dd6cf9d2d9e97d9f7afaa',
    'sglang/srt/sampling/sampling_params.py': 'b5f5305a0c6ee89597b0e0a55ff3051937deb60444eb709066c2293a19d67281',
}
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class ServingError(ValueError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def cpu_allocation():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ServingError('Load generation requires a CPU Slurm step and empty CUDA_VISIBLE_DEVICES')


def endpoint(value, protocol):
    parsed = urllib.parse.urlsplit(value)
    expected = '/v1/completions' if protocol == 'openai_completions' else '/generate'
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path.rstrip('/') != expected:
        raise ServingError('Use an explicit public HTTP completion endpoint without credentials or query parameters')
    return parsed


def read_config(root, path):
    path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve()
    contents = path.read_bytes()
    config = json.loads(contents)
    if config.get('schema_version') != 1 or Path(config['cluster_root']).resolve() != root:
        raise ServingError('Serving configuration does not identify this project root')
    if any(type(config[name]) is not int or config[name] < minimum for name, minimum in (('warmup_requests', 3), ('measured_repeats', 5), ('requests_per_repeat', 1), ('max_tokens', 1))):
        raise ServingError('Require at least three warmups, five measured repeats and positive request/output budgets')
    if not 0 <= config['temperature'] or not math.isfinite(config['temperature']) or not math.isfinite(config['request_timeout_seconds']) or config['request_timeout_seconds'] <= 0:
        raise ServingError('Invalid serving sampling or timeout settings')
    if not config['concurrencies'] or len(set(config['concurrencies'])) != len(config['concurrencies']) or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in config['concurrencies']):
        raise ServingError('Offered concurrency values must be unique positive integers')
    if config['requests_per_repeat'] < max(config['concurrencies']) or type(config['stream']) is not bool or type(config.get('ignore_eos', False)) is not bool or type(config['seed']) is not int or config['seed'] < 0:
        raise ServingError('Provide enough requests for the offered concurrency and explicit valid sampling settings')
    if config['server']['readiness_timeout_seconds'] <= 0 or not 0 < config['server']['memory_fraction'] < 1 or any(type(config['server'][name]) is not int or config['server'][name] < 1 for name in ('max_model_len', 'max_running_requests', 'vllm_speculative_tokens', 'sglang_num_steps', 'sglang_topk', 'sglang_draft_tokens')):
        raise ServingError('Invalid bounded server resource configuration')
    for name, method in config['methods'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or method['protocol'] not in ('openai_completions', 'sglang_generate'):
            raise ServingError('Invalid serving method name or explicit HTTP protocol')
        endpoint(method['endpoint'], method['protocol'])
        if not method.get('model') or method['engine'] not in PROFILES or method.get('mode') not in ('target_only', 'eagle3'):
            raise ServingError('Stock serving supports only configured autoregressive and EAGLE3 methods')
        if method['engine'] == 'vllm' and method['protocol'] != 'openai_completions':
            raise ServingError('The stock vLLM profile requires its OpenAI completions protocol')
        profile = config['engines'][method['engine']]
        if profile['version'] != PROFILES[method['engine']]['version'] or not profile.get('serving_python'):
            raise ServingError('Choose the pinned engine and an explicit separate serving_python')
        if not isinstance(method['tensor_parallel_size'], int) or method['tensor_parallel_size'] < 1:
            raise ServingError('Tensor parallel size must be a positive integer')
    return config, path, contents


def read_prompts(root, specification):
    path = Path(specification['path'])
    path = path if path.is_absolute() else root / path
    contents = path.read_bytes()
    rows = [json.loads(line) for line in contents.decode('utf-8').splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) or not isinstance(row.get('id'), (str, int)) or isinstance(row.get('id'), bool) or not isinstance(row.get('prompt'), str) or not row['prompt'].strip() or not isinstance(row.get('domain'), str) or not row['domain'] for row in rows):
        raise ServingError('Prompt JSONL requires unique IDs, literal nonempty prompts and domain labels')
    if len({json.dumps(row['id']) for row in rows}) != len(rows):
        raise ServingError('Prompt stream IDs must be unique')
    if specification['mode'] not in ('natural', 'controlled_shift'):
        raise ServingError('Prompt stream mode must be natural or controlled_shift')
    return rows, {'path': str(path.resolve()), 'sha256': hashlib.sha256(contents).hexdigest(), 'rows': len(rows), 'mode': specification['mode'], 'declared_provenance': specification.get('provenance'), 'declared_split': specification.get('split'), 'prompt_format': specification.get('prompt_format', 'literal_text')}


def prompt_schedule(rows, specification, count):
    if specification['mode'] == 'natural':
        return [rows[index % len(rows)] for index in range(count)]
    domains = {}
    for row in rows:
        domains.setdefault(row['domain'], []).append(row)
    blocks = specification.get('domain_schedule', [])
    if not blocks or any(block.get('domain') not in domains or not isinstance(block.get('requests'), int) or block['requests'] < 1 for block in blocks):
        raise ServingError('Controlled traffic requires available domains and positive block lengths')
    pattern = [block['domain'] for block in blocks for _ in range(block['requests'])]
    offsets = {domain: 0 for domain in domains}
    result = []
    for index in range(count):
        domain = pattern[index % len(pattern)]
        result.append(domains[domain][offsets[domain] % len(domains[domain])])
        offsets[domain] += 1
    return result


def request_payload(method, prompt, config, seed):
    settings = {'temperature': config['temperature'], 'top_p': 1.0, 'top_k': -1, 'repetition_penalty': 1.0, 'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'ignore_eos': config.get('ignore_eos', False)}
    if method['protocol'] == 'openai_completions':
        payload = {'model': method['model'], 'prompt': prompt, 'max_tokens': config['max_tokens'], 'n': 1, 'stream': config['stream'], **settings}
        if PROFILES[method['engine']]['per_request_seed']:
            payload['seed'] = seed
        if config['stream']:
            payload['stream_options'] = {'include_usage': True}
        return payload
    return {'text': prompt, 'sampling_params': dict(settings, max_new_tokens=config['max_tokens']), 'stream': config['stream']}


def sse_messages(chunks):
    buffer = b''
    data = []
    for chunk in chunks:
        buffer += chunk
        if len(buffer) > MAX_RESPONSE_BYTES:
            raise ServingError('Stream line exceeds the response bound')
        while b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)
            line = line.rstrip(b'\r')
            if not line:
                if data:
                    yield b'\n'.join(data).decode('utf-8')
                    data = []
            elif line.startswith(b'data:'):
                data.append(line[5:].lstrip(b' '))
    if buffer.strip() or data:
        raise ServingError('Stream ended inside an incomplete event')


def response_state():
    return {'text': '', 'usage': None, 'finish_reason': None, 'events': [], 'output_events': [], 'responses': [], 'done': False, 'last_completion_count': 0, 'final_stream_usage': False}


def consume_event(state, event, offset, protocol, model):
    if not isinstance(event, dict) or event.get('error'):
        raise ServingError('Server returned a malformed event or explicit error')
    state['events'].append({'offset_seconds': offset, 'data': event})
    state['responses'].append(event)
    state['protocol'] = protocol
    delta_tokens = None
    visible = False
    if protocol == 'openai_completions':
        if event.get('model') and event['model'] != model:
            raise ServingError('Response model differs from the requested served model')
        choices = event.get('choices', [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ServingError('Only one completion per request is supported')
        if choices:
            choice = choices[0]
            if choice.get('index', 0) != 0 or not isinstance(choice.get('text', ''), str):
                raise ServingError('Unexpected completion choice')
            text = choice.get('text', '')
            state['text'] += text
            visible = bool(text)
            logprobs = choice.get('logprobs') or {}
            if isinstance(logprobs.get('tokens'), list) and all(isinstance(token, str) for token in logprobs['tokens']):
                delta_tokens = len(logprobs['tokens'])
            if choice.get('finish_reason') is not None:
                state['finish_reason'] = choice['finish_reason']
        if event.get('usage') is not None:
            state['usage'] = event['usage']
            if not choices and state['finish_reason'] is not None:
                state['final_stream_usage'] = True
    else:
        text = event.get('text', '')
        if not isinstance(text, str) or not text.startswith(state['text']):
            raise ServingError('Native SGLang stream must contain cumulative text')
        visible = len(text) > len(state['text'])
        state['text'] = text
        metadata = event.get('meta_info', {})
        count = metadata.get('completion_tokens')
        if not isinstance(count, int) or isinstance(count, bool) or count < state['last_completion_count']:
            raise ServingError('Native SGLang events require nondecreasing completion-token counts')
        delta_tokens = count - state['last_completion_count']
        state['last_completion_count'] = count
        state['usage'] = {'prompt_tokens': metadata.get('prompt_tokens'), 'completion_tokens': count}
        if metadata.get('finish_reason') is not None:
            reason = metadata['finish_reason']
            state['finish_reason'] = reason.get('type') if isinstance(reason, dict) else reason
    if visible or (delta_tokens is not None and delta_tokens > 0):
        state['output_events'].append({'offset_seconds': offset, 'token_count': delta_tokens})


def metric_capabilities(responses, paths):
    result = {}
    for name in ('drafting_seconds', 'verification_seconds', 'memory_bytes'):
        path = paths.get(name)
        found = None
        if path:
            for response in responses:
                value = response
                for key in path.split('.'):
                    value = value.get(key) if isinstance(value, dict) else None
                if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                    found = value
        result[name] = {'available': found is not None, 'value': found, 'response_path': path, 'reason': None if found is not None else 'not_provided_by_server'}
    return result


def finish_response(state, streaming, metric_paths):
    usage = state['usage']
    count = usage.get('completion_tokens') if isinstance(usage, dict) else None
    reasons = []
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        reasons.append('missing_or_zero_usage_completion_tokens')
    if state['finish_reason'] not in ('stop', 'length', 'eos_token'):
        reasons.append('missing_or_unsuccessful_finish_reason')
    if streaming and not state['done']:
        reasons.append('missing_stream_terminator')
    if streaming and state.get('protocol') == 'openai_completions' and not state['final_stream_usage']:
        reasons.append('missing_final_stream_usage')
    arrivals = state['output_events']
    intervals = [right['offset_seconds'] - left['offset_seconds'] for left, right in zip(arrivals, arrivals[1:])]
    token_events = bool(arrivals) and all(item['token_count'] == 1 for item in arrivals) and len(arrivals) == count
    return {'eligible': not reasons, 'ineligibility_reasons': reasons, 'output_text': state['text'], 'usage': usage, 'output_tokens': count if isinstance(count, int) and not isinstance(count, bool) else None, 'finish_reason': state['finish_reason'], 'ttft_seconds': arrivals[0]['offset_seconds'] if streaming and arrivals else None, 'ttft_definition': 'Request start to first output stream event', 'inter_output_event_seconds': intervals if streaming else [], 'itl_seconds': intervals if streaming and token_events and len(arrivals) > 1 else None, 'itl_unavailable_reason': None if streaming and token_events and len(arrivals) > 1 else 'no_complete_one_token_per_event_evidence', 'server_metrics': metric_capabilities(state['responses'], metric_paths)}


def perform_request(method, prompt, config, seed):
    payload = request_payload(method, prompt, config, seed)
    state = response_state()
    started = time.monotonic()
    result = {'request': payload, 'seed_requested': seed, 'per_request_seed_applied': PROFILES[method['engine']]['per_request_seed'], 'stream_events': state['events'], 'request_start_monotonic': started}
    try:
        request = urllib.request.Request(method['endpoint'], data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream' if config['stream'] else 'application/json'}, method='POST')
        opener = urllib.request.build_opener(NoRedirect())
        with opener.open(request, timeout=config['request_timeout_seconds']) as response:
            total = 0

            def chunks():
                nonlocal total
                while True:
                    if time.monotonic() - started > config['request_timeout_seconds']:
                        raise TimeoutError('Request stream exceeded the configured deadline')
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_RESPONSE_BYTES:
                        raise ServingError('Response exceeds the bounded body size')
                    yield chunk

            if config['stream']:
                if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                    raise ServingError('Streaming request did not return an event stream')
                for data in sse_messages(chunks()):
                    offset = time.monotonic() - started
                    if data == '[DONE]':
                        state['events'].append({'offset_seconds': offset, 'data': '[DONE]'})
                        state['done'] = True
                        break
                    consume_event(state, json.loads(data), offset, method['protocol'], method['model'])
            else:
                body = b''.join(chunks())
                consume_event(state, json.loads(body), time.monotonic() - started, method['protocol'], method['model'])
        result.update(finish_response(state, config['stream'], method.get('provided_metric_paths', {})))
        if config.get('ignore_eos') and result['output_tokens'] != config['max_tokens']:
            result['eligible'] = False
            result['ineligibility_reasons'].append('fixed_output_budget_not_met')
        if result['output_tokens'] is not None and result['output_tokens'] > config['max_tokens']:
            result['eligible'] = False
            result['ineligibility_reasons'].append('output_budget_exceeded')
        result['status'] = 'completed' if result['eligible'] else 'ineligible'
    except Exception as error:
        result.update(status='error', eligible=False, ineligibility_reasons=['request_error'], error_type=type(error).__name__, output_tokens=None, ttft_seconds=None, itl_seconds=None, inter_output_event_seconds=[], server_metrics=metric_capabilities([], {}))
        if isinstance(error, urllib.error.HTTPError):
            result['http_status'] = error.code
            error.close()
        if isinstance(error, ServingError):
            result['error'] = str(error)
    result['latency_seconds'] = time.monotonic() - started
    result['request_finish_monotonic'] = started + result['latency_seconds']
    if result['latency_seconds'] > config['request_timeout_seconds']:
        result.update(status='error', eligible=False, ineligibility_reasons=list(set(result['ineligibility_reasons'] + ['request_deadline_exceeded'])))
    return result


def quantile(values, probability):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize_cell(records, wall_seconds, expected_requests, warmup_ok, provenance_ok, smoke):
    completed = [row for row in records if row.get('eligible') and row.get('status') == 'completed']
    tokens = sum(row['output_tokens'] for row in completed)
    errors = len(records) - len(completed)
    ttft = [row['ttft_seconds'] for row in completed if row.get('ttft_seconds') is not None]
    itl = [value for row in completed for value in (row.get('itl_seconds') or [])]
    latency = [row['latency_seconds'] for row in completed]
    edges = sorted((value, delta) for row in records for value, delta in ((row['request_start_monotonic'], 1), (row['request_finish_monotonic'], -1)))
    active = 0
    peak = 0
    for _, delta in edges:
        active += delta
        peak = max(peak, active)
    eligible = len(records) == expected_requests and len(completed) == expected_requests and expected_requests > 0 and wall_seconds > 0 and warmup_ok and provenance_ok and not smoke
    server_metrics = {}
    for name in ('drafting_seconds', 'verification_seconds', 'memory_bytes'):
        values = [row['server_metrics'][name]['value'] for row in completed if row.get('server_metrics', {}).get(name, {}).get('available')]
        server_metrics[name] = {'availability': 'provided' if values and len(values) == len(completed) else 'partial' if values else 'unavailable', 'requests_with_measurement': len(values), 'mean_reported_value': sum(values) / len(values) if values else None, 'maximum_reported_value': max(values) if values else None}
    return {'eligible': eligible, 'scope': 'smoke' if smoke else 'serving_measurement', 'expected_requests': expected_requests, 'observed_requests': len(records), 'observed_peak_inflight_requests': peak, 'completed_requests': len(completed), 'errors_or_ineligible_requests': errors, 'warmup_complete': warmup_ok, 'server_provenance_matched': provenance_ok, 'output_tokens_from_usage': tokens, 'observed_wall_seconds': wall_seconds, 'observed_output_tokens_per_second': tokens / wall_seconds if wall_seconds > 0 and completed else None, 'request_latency_seconds': {'p50': quantile(latency, 0.5), 'p90': quantile(latency, 0.9), 'p99': quantile(latency, 0.99)}, 'ttft_seconds': {'requests_with_evidence': len(ttft), 'p50': quantile(ttft, 0.5), 'p99': quantile(ttft, 0.99)}, 'itl_seconds': {'intervals_with_evidence': len(itl), 'p50': quantile(itl, 0.5), 'p99': quantile(itl, 0.99)}, 'server_metrics': server_metrics, 'server_component_and_memory_metrics': 'Only explicitly configured response fields; no client-side estimates'}


def checkpoint_records(root, method, verify=False):
    receipt_path = root / 'outputs/preparation/assets.json'
    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get('status') != 'complete' or Path(receipt['root']).resolve() != root or receipt['registry_sha256'] != digest(root / 'configs/assets.json') or receipt['source_sha256'] != digest(root / 'src/taps_rebuild/assets.py'):
        raise ServingError('A matching completed asset preparation receipt is required')
    selected = {}
    names = [method['target_asset']] + ([method['draft_asset']] if method['mode'] == 'eagle3' else [])
    for name in names:
        record = receipt['models'][name]
        if verify:
            for relative, value in record['files'].items():
                assets.verify_file(Path(record['path']) / relative, {'size_bytes': value['size_bytes'], 'lfs_sha256': value['sha256']})
        selected[name] = {'path': record['path'], 'model_identity_sha256': record['model_identity_sha256'], 'source': record['source']}
    return selected, digest(receipt_path)


def server_command(root, config, name, checkpoints):
    method = config['methods'][name]
    engine = method['engine']
    profile = PROFILES[engine]
    specification = config['engines'][engine]
    python = Path(specification['serving_python'])
    python = python if python.is_absolute() else root / python
    legacy = Path(config['legacy_runtime_root']) / 'envs'
    if python.parent.parent.resolve() in {(legacy / 'taps-eagle').resolve(), (legacy / 'taps-hass').resolve()}:
        raise ServingError('Serving must use a separate environment')
    target = checkpoints[method['target_asset']]['path']
    parsed = endpoint(method['endpoint'], method['protocol'])
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    if parsed.scheme != 'http':
        raise ServingError('Stock launcher serves HTTP; terminate TLS externally with separate provenance')
    server = config['server']
    command = [str(python), '-m', profile['module']]
    if engine == 'vllm':
        command += ['--model', target, '--served-model-name', method['model'], '--host', method['bind_host'], '--port', str(port), '--tensor-parallel-size', str(method['tensor_parallel_size']), '--dtype', 'half', '--max-model-len', str(server['max_model_len']), '--max-num-seqs', str(server['max_running_requests']), '--gpu-memory-utilization', str(server['memory_fraction']), '--seed', str(config['seed']), '--generation-config', 'vllm']
        if method['mode'] == 'eagle3':
            if method['tensor_parallel_size'] != 1:
                raise ServingError('Pinned vLLM EAGLE3 distributed inference requires separate implementation and validation')
            command += ['--speculative-config', json.dumps({'method': 'eagle3', 'model': checkpoints[method['draft_asset']]['path'], 'num_speculative_tokens': server['vllm_speculative_tokens']}, separators=(',', ':'))]
        overrides = {'VLLM_USE_V1': '1'}
    else:
        command += ['--model-path', target, '--served-model-name', method['model'], '--host', method['bind_host'], '--port', str(port), '--tp-size', str(method['tensor_parallel_size']), '--dtype', 'float16', '--context-length', str(server['max_model_len']), '--max-running-requests', str(server['max_running_requests']), '--mem-fraction-static', str(server['memory_fraction']), '--random-seed', str(config['seed'])]
        if method['mode'] == 'eagle3':
            command += ['--speculative-algorithm', 'EAGLE3', '--speculative-draft-model-path', checkpoints[method['draft_asset']]['path'], '--speculative-num-steps', str(server['sglang_num_steps']), '--speculative-eagle-topk', str(server['sglang_topk']), '--speculative-num-draft-tokens', str(server['sglang_draft_tokens'])]
        overrides = {}
    identity = hashlib.sha256(canonical({'command': command, 'environment_overrides': overrides, 'checkpoints': checkpoints})).hexdigest()
    return command, overrides, identity


def allocation_resources(method, query=None):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise ServingError('Server launch requires a GPU Slurm allocation')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != method['tensor_parallel_size'] or any(not value for value in visible) or len(set(visible)) != len(visible):
        raise ServingError('Expose exactly the configured tensor-parallel GPU graph')
    query = attach.command_output if query is None else query
    job = attach.fields(query(['scontrol', 'show', 'job', '-o', os.environ['SLURM_JOB_ID']]))
    owner = re.fullmatch(r'[^()]+\(([0-9]+)\)', job.get('UserId', ''))
    count = attach.gpu_count(job.get('AllocTRES') or job.get('TRES', ''))
    if not owner or int(owner.group(1)) != os.getuid() or job.get('JobState') != 'RUNNING' or job.get('NodeList') != os.environ['SLURMD_NODENAME'] or job.get('NumNodes') != '1' or count is None or count < len(visible):
        raise ServingError('The live allocation does not own the requested single-node GPU graph')
    owned = set()
    for value in (os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS', '')).split(','):
        if re.fullmatch(r'[0-9]+-[0-9]+', value):
            first, last = map(int, value.split('-'))
            owned.update(str(index) for index in range(first, last + 1))
        elif value:
            owned.add(value)
    if not owned:
        raise ServingError('Allocated GPU identifiers are required for the explicit resource graph')
    inventory = query(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader,nounits'])
    records = []
    for line in inventory.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 4:
            raise ServingError('Unexpected NVIDIA hardware inventory')
        if parts[0] in visible or parts[1] in visible:
            if parts[0] not in owned and parts[1] not in owned:
                raise ServingError('A selected GPU does not belong to the live allocation')
            driver = tuple(int(value) for value in parts[3].split('.'))
            if driver < (550, 54, 15):
                raise ServingError('CUDA12.4.1 serving profile requires driver550.54.15 or newer')
            records.append({'index': parts[0], 'uuid': parts[1], 'name': parts[2], 'driver_version': parts[3]})
    if len(records) != len(visible):
        raise ServingError('Visible GPU graph does not match the hardware inventory')
    return {'job_id': os.environ['SLURM_JOB_ID'], 'node': os.environ['SLURMD_NODENAME'], 'visible_devices': visible, 'allocated_gpus': count, 'hardware': records}


def runtime_check(command, engine, environment):
    python = command[0]
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ServingError('Prepare the explicitly configured separate serving environment first')
    script = '\n'.join(['import hashlib,importlib.metadata as m,json,sys,torch', 'distribution=m.distribution(' + repr(engine) + ')', 'files=' + repr(list(PROFILES[engine]['source_files'])), 'hashes={name:hashlib.sha256(distribution.locate_file(name).read_bytes()).hexdigest() for name in files}', 'print(json.dumps({"packages":{p:m.version(p) for p in [' + repr(engine) + ',"torch","transformers"]},"torch_cuda":torch.version.cuda,"prefix":sys.prefix,"base_prefix":sys.base_prefix,"source_file_sha256":hashes}))'])
    result = subprocess.run([python, '-c', script], env=environment, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise ServingError('Separate serving runtime import preflight failed')
    value = json.loads(result.stdout.strip().splitlines()[-1])
    profile = PROFILES[engine]
    if Path(value['prefix']).resolve() != Path(python).parent.parent.resolve() or value['prefix'] == value['base_prefix']:
        raise ServingError('The configured executable must identify a separate virtual environment')
    if any(value['packages'][package].split('+')[0] != expected for package, expected in ((engine, profile['version']), ('torch', profile['torch']), ('transformers', profile['transformers']))) or value['torch_cuda'] != profile['cuda']:
        raise ServingError('Installed serving runtime differs from the pinned CUDA12.4 profile')
    if value['source_file_sha256'] != profile['source_files']:
        raise ServingError('Installed proposer, model or protocol code differs from the pinned stock source')
    help_result = subprocess.run(command[:3] + ['--help'], env=environment, capture_output=True, text=True, timeout=120)
    if help_result.returncode != 0 or any(argument not in help_result.stdout for argument in command[3:] if argument.startswith('--')):
        raise ServingError('Installed server CLI does not advertise all configured stock capabilities')
    value['cli_help_sha256'] = hashlib.sha256(help_result.stdout.encode()).hexdigest()
    value['runtime_verified'] = True
    value['verification_scope'] = 'Package versions, CUDA runtime version and CLI capability advertisement; model load still requires a real response'
    return value


def launch_server(root, config, name, detach=False):
    method = config['methods'][name]
    resources = allocation_resources(method)
    checkpoints, asset_hash = checkpoint_records(root, method, verify=True)
    command, overrides, identity = server_command(root, config, name, checkpoints)
    environment = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD'):
        environment.pop(key, None)
    environment.update(overrides)
    environment.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
    runtime = runtime_check(command, method['engine'], environment)
    output = root / 'outputs/serving/servers'
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = output / (name + '-' + stamp + '.json')
    log = path.with_suffix('.log')
    receipt = {'status': 'launching', 'method': name, 'endpoint': method['endpoint'], 'model': method['model'], 'command': command, 'environment_overrides': overrides, 'server_identity_sha256': identity, 'source_sha256': digest(Path(__file__)), 'checkpoints': checkpoints, 'asset_receipt_sha256': asset_hash, 'resources': resources, 'runtime': runtime, 'log': str(log), 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'stock_custom_proposer_supported': False}
    sources.atomic_json(path, receipt)
    with log.open('xb') as stream:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=detach)
    receipt.update(status='launched', pid=process.pid, process_identity=attach.process_identity(process.pid))
    sources.atomic_json(path, receipt)
    print(json.dumps({'status': 'launched', 'pid': process.pid, 'receipt': str(path), 'log': str(log)}), flush=True)
    returncode = process.poll() if detach else process.wait()
    if returncode is not None:
        receipt.update(status='completed' if returncode == 0 else 'failed', returncode=returncode)
        sources.atomic_json(path, receipt)
    return receipt


def server_provenance(root, config, name, checkpoints):
    _, _, identity = server_command(root, config, name, checkpoints)
    for path in sorted((root / 'outputs/serving/servers').glob(name + '-*.json'), reverse=True):
        receipt = json.loads(path.read_bytes())
        if receipt.get('status') == 'launched' and receipt.get('server_identity_sha256') == identity and receipt.get('runtime', {}).get('runtime_verified') is True:
            return {'matched': True, 'receipt': str(path), 'receipt_sha256': digest(path), 'server_identity_sha256': identity, 'basis': 'Matching launcher receipt; HTTP is not a cryptographic process attestation'}
    return {'matched': False, 'server_identity_sha256': identity, 'basis': 'No matching verified server launch receipt'}


def append_json(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + '\n')
    stream.flush()


def wait_endpoint(method, timeout):
    parsed = endpoint(method['endpoint'], method['protocol'])
    url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, '/health', '', ''))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.build_opener(NoRedirect()).open(url, timeout=min(2, max(0.01, deadline - time.monotonic()))) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError) as error:
            if isinstance(error, urllib.error.HTTPError):
                error.close()
        time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    raise ServingError('Configured server did not become healthy within the readiness deadline')


def run_load(root, config_path, run_id, methods=None, smoke=False):
    cpu_allocation()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id):
        raise ServingError('Invalid serving run ID')
    root = Path(root).resolve()
    config, path, config_contents = read_config(root, config_path)
    selected = methods.split(',') if methods else config['method_order']
    if not selected or len(set(selected)) != len(selected) or any(name not in config['methods'] for name in selected):
        raise ServingError('Select unique configured serving methods')
    if smoke:
        rows = config.get('smoke_prompts', [])
        if len(rows) < 3 or any(not isinstance(row.get('prompt'), str) or not row['prompt'].strip() or not row.get('id') or not row.get('domain') for row in rows):
            raise ServingError('Smoke mode requires three explicit authored mechanical prompts')
        stream_info = {'sha256': hashlib.sha256(canonical(rows)).hexdigest(), 'rows': len(rows), 'mode': 'authored_engineering_smoke', 'natural_heldout_traffic': False}
    else:
        rows, stream_info = read_prompts(root, config['prompt_stream'])
    count = min(2, config['requests_per_repeat']) if smoke else config['requests_per_repeat']
    schedule = prompt_schedule(rows, {'mode': 'natural'} if smoke else config['prompt_stream'], count)
    concurrencies = [1] if smoke else config['concurrencies']
    run = root / 'outputs/serving' / run_id
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir()
    manifest = {'run_id': run_id, 'scope': 'smoke' if smoke else 'serving_measurement', 'root': str(root), 'config': config, 'config_path': str(path), 'config_sha256': hashlib.sha256(config_contents).hexdigest(), 'source_sha256': digest(Path(__file__)), 'prompt_stream': stream_info, 'selected_methods': selected, 'concurrencies': concurrencies, 'requested_requests_per_repeat': count, 'warmup_requests_by_concurrency': {str(value): max(config['warmup_requests'], value) for value in concurrencies}, 'schedule_ids': [row['id'] for row in schedule], 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'checkpoint_identity': {}, 'server_provenance': {}, 'concurrency_definition': 'Closed-loop offered maximum in-flight HTTP requests, not a fixed server batch size', 'connection_policy': 'Fresh urllib HTTP connection per request; observed latency includes client and network overhead', 'full_model_distributional_correctness_established': False, 'automatic_pipeline_gate_pass': False}
    for name in selected:
        checkpoints, asset_hash = checkpoint_records(root, config['methods'][name])
        manifest['checkpoint_identity'][name] = {'models': checkpoints, 'asset_receipt_sha256': asset_hash}
        manifest['server_provenance'][name] = server_provenance(root, config, name, checkpoints)
    sources.atomic_json(run / 'manifest.json', manifest)
    sources.atomic_json(run / 'state.json', {'status': 'running'})
    with (run / 'requests.jsonl').open('x') as raw, (run / 'stream-events.jsonl').open('x') as events:
        try:
            for name in selected:
                method = config['methods'][name]
                wait_endpoint(method, config['server']['readiness_timeout_seconds'])
                for concurrency in concurrencies:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        warmup = []
                        warmup_count = max(config['warmup_requests'], concurrency)
                        tasks = {pool.submit(perform_request, method, rows[index % len(rows)]['prompt'], config, config['seed'] + index): index for index in range(warmup_count)}
                        for future in as_completed(tasks):
                            result = future.result()
                            index = tasks[future]
                            result.update(method=name, offered_concurrency=concurrency, warmup=True, repeat=-1, request_index=index, prompt_id=rows[index % len(rows)]['id'])
                            for event in result.pop('stream_events', []):
                                append_json(events, {'method': name, 'concurrency': concurrency, 'warmup': True, 'repeat': -1, 'request_index': index, **event})
                            append_json(raw, result)
                            warmup.append(result)
                        warmup_ok = len(warmup) == warmup_count and all(row['eligible'] for row in warmup)
                        for repeat in range(config['measured_repeats']):
                            started = time.monotonic()
                            tasks = {pool.submit(perform_request, method, row['prompt'], config, config['seed'] + repeat * count + index): (index, row) for index, row in enumerate(schedule)}
                            completed = []
                            for future in as_completed(tasks):
                                result = future.result()
                                index, prompt = tasks[future]
                                result.update(method=name, offered_concurrency=concurrency, warmup=False, repeat=repeat, request_index=index, prompt_id=prompt['id'], domain=prompt['domain'], warmup_complete=warmup_ok)
                                completed.append(result)
                            finished = max(result['request_finish_monotonic'] for result in completed)
                            for result in completed:
                                for event in result.pop('stream_events', []):
                                    append_json(events, {'method': name, 'concurrency': concurrency, 'warmup': False, 'repeat': repeat, 'request_index': result['request_index'], **event})
                                append_json(raw, result)
                            append_json(raw, {'record_type': 'repeat_wall', 'method': name, 'offered_concurrency': concurrency, 'repeat': repeat, 'wall_seconds': finished - started, 'warmup_complete': warmup_ok, 'raw_log_writing_excluded': True})
            sources.atomic_json(run / 'state.json', {'status': 'complete'})
        except Exception as error:
            sources.atomic_json(run / 'state.json', {'status': 'failed', 'error_type': type(error).__name__})
            raise
    return summarize_run(run)


def summarize_run(run):
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_bytes())
    state = json.loads((run / 'state.json').read_bytes())
    records = [json.loads(line) for line in (run / 'requests.jsonl').read_text().splitlines() if line.strip()]
    cells = []
    for name in manifest['selected_methods']:
        for concurrency in manifest['concurrencies']:
            selected = [row for row in records if row.get('method') == name and row.get('offered_concurrency') == concurrency]
            measured = [row for row in selected if row.get('warmup') is False]
            walls = [row for row in selected if row.get('record_type') == 'repeat_wall']
            warmups = [row for row in selected if row.get('warmup') is True]
            expected = manifest['requested_requests_per_repeat'] * manifest['config']['measured_repeats']
            coverage = {(row['repeat'], row['request_index']) for row in measured}
            expected_keys = {(repeat, index) for repeat in range(manifest['config']['measured_repeats']) for index in range(manifest['requested_requests_per_repeat'])}
            warmup_ok = len(warmups) == manifest['warmup_requests_by_concurrency'][str(concurrency)] and all(row['eligible'] for row in warmups) and len(walls) == manifest['config']['measured_repeats'] and all(row['warmup_complete'] for row in walls)
            summary = summarize_cell(measured, sum(row['wall_seconds'] for row in walls), expected, warmup_ok, manifest['server_provenance'][name]['matched'], manifest['scope'] == 'smoke')
            summary.update(method=name, offered_concurrency=concurrency, exact_coverage=coverage == expected_keys and len(measured) == expected)
            summary['eligible'] = summary['eligible'] and summary['exact_coverage'] and state['status'] == 'complete'
            cells.append(summary)
    comparisons = []
    definitions = manifest['config']['methods']
    for candidate in cells:
        method = definitions[candidate['method']]
        if method['mode'] != 'eagle3':
            continue
        for baseline in cells:
            other = definitions[baseline['method']]
            if other['mode'] != 'target_only' or other['engine'] != method['engine'] or other['protocol'] != method['protocol'] or other['target_asset'] != method['target_asset'] or candidate['offered_concurrency'] != baseline['offered_concurrency']:
                continue
            valid = candidate['eligible'] and baseline['eligible']
            paired = {}
            for name in (candidate['method'], baseline['method']):
                paired[name] = {(row['repeat'], row['request_index']): (row['prompt_id'], row['seed_requested'], row['request'].get('prompt', row['request'].get('text'))) for row in records if row.get('warmup') is False and row.get('method') == name and row.get('offered_concurrency') == candidate['offered_concurrency']}
            prompts_match = paired[candidate['method']] == paired[baseline['method']] and bool(paired[candidate['method']])
            valid = valid and prompts_match
            comparisons.append({'method': candidate['method'], 'baseline': baseline['method'], 'offered_concurrency': candidate['offered_concurrency'], 'eligible': valid, 'actual_prompt_and_seed_pairs_match': prompts_match, 'observed_output_token_throughput_ratio': candidate['observed_output_tokens_per_second'] / baseline['observed_output_tokens_per_second'] if valid else None, 'basis': 'Same actual prompt and seed pairs and output cap; tokens from server usage and active load-window wall time, excluding raw log writes', 'equal_output_or_distributional_correctness_established': False})
    result = {'scope': manifest['scope'], 'run_id': manifest['run_id'], 'manifest_sha256': digest(run / 'manifest.json'), 'raw_requests_sha256': digest(run / 'requests.jsonl'), 'stream_events_sha256': digest(run / 'stream-events.jsonl'), 'state': state, 'cells': cells, 'paired_comparisons': comparisons, 'automatic_pipeline_gate_pass': False, 'stock_custom_proposer_supported': False, 'qualifications': ['Offered concurrency is not a fixed server batch size.', 'Missing usage, errors, incomplete streams, incomplete coverage, unmatched server provenance and smoke scope are ineligible for performance comparisons.', 'TTFT uses observed output stream arrival; ITL requires complete one-token-per-event evidence.', 'Server drafting, verification and memory values are unavailable unless explicitly present in configured raw response fields.', 'Full-model distributional correctness remains a separate requirement.']}
    sources.atomic_json(run / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('command', 'launch', 'load', 'summarize'))
    parser.add_argument('--root', type=Path, default=Path('/mnt/nfs/home/mohamadzbib/projects/taps-rebuild'))
    parser.add_argument('--config', type=Path, default=Path('configs/serving.json'))
    parser.add_argument('--method')
    parser.add_argument('--methods')
    parser.add_argument('--run-id')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.action == 'load':
            if not args.run_id:
                raise ServingError('Load generation requires a fresh run ID')
            result = run_load(root, args.config, args.run_id, args.methods, args.smoke)
            print(json.dumps({'run_id': result['run_id'], 'scope': result['scope'], 'summary': str(root / 'outputs/serving' / args.run_id / 'summary.json')}), flush=True)
        elif args.action == 'summarize':
            cpu_allocation()
            if not args.run_id or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', args.run_id):
                raise ServingError('Summarization requires a valid existing run ID')
            result = summarize_run(root / 'outputs/serving' / args.run_id)
            print(json.dumps({'run_id': result['run_id'], 'cells': len(result['cells'])}), flush=True)
        else:
            config, _, _ = read_config(root, args.config)
            if args.method not in config['methods']:
                raise ServingError('Choose one configured server method')
            if args.action == 'launch':
                result = launch_server(root, config, args.method, args.detach)
                if result.get('returncode', 0) != 0:
                    raise ServingError('The launched server exited with a failure')
            else:
                checkpoints, _ = checkpoint_records(root, config['methods'][args.method])
                command, overrides, identity = server_command(root, config, args.method, checkpoints)
                print(json.dumps({'command': command, 'environment_overrides': overrides, 'server_identity_sha256': identity, 'runtime_verified': False, 'stock_custom_proposer_supported': False}), flush=True)
    except Exception as error:
        result = {'status': 'failed', 'error_type': type(error).__name__}
        if isinstance(error, (ServingError, assets.PreparationError)):
            result['error'] = str(error)
        print(json.dumps(result), flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
