import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


class PoolError(RuntimeError):
    pass


class PoolInterrupted(PoolError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('wb') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise PoolError('Inspection command failed: ' + ' '.join(arguments[:3]))
    return result.stdout


def _fields(text):
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/]*)=(\S*)', text))


def _gpu_count(tres):
    generic = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', tres)
    if generic:
        return int(generic.group(1))
    typed = re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)', tres)
    return sum(map(int, typed)) if typed else None


def allocation(expected_gpus, environment=None, command=None):
    environment = os.environ if environment is None else environment
    command = _command if command is None else command
    if isinstance(expected_gpus, bool) or not isinstance(expected_gpus, int) or expected_gpus < 1:
        raise PoolError('expected_gpus must be positive')
    job_id = environment.get('SLURM_JOB_ID', '')
    if not re.fullmatch(r'\d+', job_id):
        raise PoolError('A live Slurm allocation is required')
    if 'CUDA_VISIBLE_DEVICES' in environment and not environment['CUDA_VISIBLE_DEVICES'].strip():
        raise PoolError('Explicitly empty CUDA_VISIBLE_DEVICES is not a GPU allocation')
    job_text = command(['scontrol', 'show', 'job', '-o', job_id])
    job = _fields(job_text)
    owner = re.search(r'\((\d+)\)$', job.get('UserId', ''))
    owner_id = int(owner.group(1)) if owner else int(job['UserId']) if job.get('UserId', '').isdigit() else None
    if job.get('JobState') != 'RUNNING' or owner_id != os.getuid() or job.get('NumNodes') != '1':
        raise PoolError('Slurm job must be running, owned by this UID, and use exactly one node')
    node_name = environment.get('SLURMD_NODENAME') or socket.gethostname().split('.')[0]
    if job.get('NodeList') != node_name:
        raise PoolError('Controller is not on the allocated node')
    if _gpu_count(job.get('AllocTRES') or job.get('TRES', '')) != expected_gpus:
        raise PoolError('Allocated GPU count differs from expected_gpus')
    node_text = command(['scontrol', 'show', 'node', '-o', node_name])
    node = _fields(node_text)
    if node.get('NodeName') != node_name:
        raise PoolError('Slurm node inventory differs from the allocated node')
    configured = _gpu_count(node.get('CfgTRES', ''))
    if configured is None:
        counts = re.findall(r'(?:^|,)gpu(?::[A-Za-z0-9_.-]+)?:(\d+)(?:\(|,|$)', node.get('Gres', ''))
        configured = sum(map(int, counts)) if counts else None
    if configured != expected_gpus:
        raise PoolError('Allocation does not own all configured GPUs on this node')
    inventory_text = command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv,noheader,nounits'])
    inventory = []
    for fields in csv.reader(io.StringIO(inventory_text), skipinitialspace=True):
        if not fields:
            continue
        if len(fields) != 5:
            raise PoolError('Unexpected nvidia-smi inventory format')
        index, uuid, name, memory, driver = [field.strip() for field in fields]
        try:
            memory = float(memory)
        except ValueError as error:
            raise PoolError('Invalid GPU memory inventory') from error
        if not index.isdigit() or not uuid or not name or not driver or not math.isfinite(memory) or memory <= 0:
            raise PoolError('Incomplete GPU inventory')
        inventory.append({'index': index, 'uuid': uuid, 'name': name, 'memory_total_mib': memory, 'driver_version': driver})
    inventory.sort(key=lambda item: int(item['index']))
    if len(inventory) != expected_gpus or len({item['index'] for item in inventory}) != expected_gpus or len({item['uuid'] for item in inventory}) != expected_gpus:
        raise PoolError('Physical GPU inventory differs from the full-node allocation')
    tokens = environment['CUDA_VISIBLE_DEVICES'].split(',') if 'CUDA_VISIBLE_DEVICES' in environment else [item['index'] for item in inventory]
    tokens = [token.strip() for token in tokens]
    if len(tokens) != expected_gpus or len(set(tokens)) != expected_gpus or any(not token for token in tokens):
        raise PoolError('CUDA_VISIBLE_DEVICES must contain exactly one unique token per GPU')
    bindings = []
    for shard, token in enumerate(tokens):
        matches = [item for item in inventory if item['index'] == token or item['uuid'] == token or (token.startswith('GPU-') and item['uuid'].startswith(token))]
        if len(matches) != 1:
            raise PoolError('CUDA_VISIBLE_DEVICES token does not identify one physical GPU')
        bindings.append({'shard_index': shard, 'token': token, 'uuid': matches[0]['uuid'], 'index': matches[0]['index']})
    if len({item['uuid'] for item in bindings}) != expected_gpus:
        raise PoolError('CUDA_VISIBLE_DEVICES aliases the same physical GPU more than once')
    hardware = {'gpu_inventory': inventory, 'device_bindings': bindings}
    return {**hardware, 'hardware_identity_sha256': identity(hardware), 'job_id': job_id, 'node': node_name, 'slurm': {'job': {key: value for key, value in job.items() if key in {'JobId', 'JobName', 'UserId', 'Account', 'Partition', 'QOS', 'NodeList', 'NumNodes', 'NumCPUs', 'TRES', 'AllocTRES', 'StartTime', 'EndTime'}}, 'node': {key: value for key, value in node.items() if key in {'NodeName', 'Gres', 'CfgTRES', 'AllocTRES', 'RealMemory', 'CPUTot', 'State', 'Version'}}, 'job_record_sha256': hashlib.sha256(job_text.encode()).hexdigest(), 'node_record_sha256': hashlib.sha256(node_text.encode()).hexdigest()}}


def effective_config(config, methods, smoke, gpu_count):
    value = copy.deepcopy(config)
    order = value['method_order']
    if not order or len(order) != len(set(order)) or any(method not in value['methods'] for method in order):
        raise PoolError('Invalid configured method order')
    if methods is not None:
        selected = methods.split(',') if isinstance(methods, str) else list(methods)
        selected = [method.strip() for method in selected]
        if not selected or any(not method for method in selected) or len(selected) != len(set(selected)) or not set(selected) <= set(order):
            raise PoolError('Invalid methods filter')
        order = [method for method in order if method in selected]
    value['method_order'] = order
    value['methods'] = {method: value['methods'][method] for method in order}
    value['reference_cells'] = [cell for cell in value.get('reference_cells', []) if cell['method'] in order]
    if smoke:
        if gpu_count > 80:
            raise PoolError('The 80-question smoke source cannot cover this many GPUs')
        value.update(scope='smoke', temperatures=[0], timing_repeats=1, warmup_generations_per_mode=1, max_new_tokens=32, question_limit=gpu_count)
        for method in order:
            value['methods'][method]['benchmarks'] = ['mt_bench']
    elif value.get('scope') != 'reference':
        raise PoolError('Use --smoke for smoke-scoped execution')
    return value


def _source_files(root):
    directory = root / 'src/taps_rebuild'
    files = {}
    for parent, directories, names in os.walk(directory, followlinks=False):
        for name in directories:
            if (Path(parent) / name).is_symlink():
                raise PoolError('Authored source directories cannot be symlinks')
        for name in sorted(names):
            path = Path(parent) / name
            if path.suffix == '.py':
                if path.is_symlink() or path.resolve() != path or not path.is_file():
                    raise PoolError('Authored source must be regular files')
                files[str(path.relative_to(root))] = path.read_bytes()
    required = {'src/taps_rebuild/runner.py', 'src/taps_rebuild/backends/official.py', 'src/taps_rebuild/metrics.py', 'src/taps_rebuild/assets.py', 'src/taps_rebuild/sources.py'}
    if not required <= files.keys():
        raise PoolError('Authored source inventory is incomplete')
    return dict(sorted(files.items()))


def _code_identity(root, source_hashes, plan_hash):
    if (root / '.git').exists():
        commit = _command(['git', '-C', str(root), 'rev-parse', 'HEAD']).strip()
        dirty = bool(_command(['git', '-C', str(root), 'status', '--porcelain=v1', '--untracked-files=normal']).strip())
        if not re.fullmatch('[0-9a-f]{40}', commit):
            raise PoolError('Invalid project Git commit')
        return {'code_commit': commit, 'code_dirty': dirty, 'source_identity_kind': 'git'}
    candidates = [path for path in (root / 'source-identity.json', root / 'deployed-source-identity.json') if path.is_file()]
    if len(candidates) != 1:
        raise PoolError('A Git checkout or one deployed source-identity.json is required')
    path = candidates[0]
    value = json.loads(path.read_text())
    if not re.fullmatch('[0-9a-f]{40}', value.get('git_commit', '')) or not isinstance(value.get('tracked_hashes'), dict):
        raise PoolError('Malformed deployed source identity')
    for name, expected in value['tracked_hashes'].items():
        relative = Path(name)
        file = root / relative
        if relative.is_absolute() or '..' in relative.parts or file.is_symlink() or file.resolve() != file or digest(file) != expected:
            raise PoolError('Deployed tracked file identity differs: ' + name)
    if any(value['tracked_hashes'].get(name) != expected for name, expected in source_hashes.items()) or value.get('plan_sha256', plan_hash) != plan_hash:
        raise PoolError('Deployed authored source or plan identity differs')
    return {'code_commit': value['git_commit'], 'code_dirty': False, 'source_identity_kind': 'deployed_source_identity', 'deployed_source_identity_sha256': digest(path)}


def prerequisites(root, config_path, selected):
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    plan = root / 'TAPS_rebuild_plan.pdf'
    plan_hash = digest(plan)
    if config.get('plan_sha256') != plan_hash:
        raise PoolError('Rebuild plan hash differs from the configuration')
    if config.get('cluster_root') and Path(config['cluster_root']).resolve() != root:
        raise PoolError('Configuration belongs to another project root')
    source_files = _source_files(root)
    hashes = {name: hashlib.sha256(contents).hexdigest() for name, contents in source_files.items()}
    code = _code_identity(root, hashes, plan_hash)
    receipts = {}
    registries = {}
    for name in ('assets', 'sources'):
        path = root / 'outputs/preparation' / (name + '.json')
        contents = path.read_bytes()
        receipt = json.loads(contents)
        registry = Path(receipt['registry']).resolve()
        if receipt.get('status') != 'complete' or Path(receipt['root']).resolve() != root or registry != root / 'configs' / (name + '.json'):
            raise PoolError('A complete preparation receipt for this root is required: ' + name)
        registry_bytes = registry.read_bytes()
        if hashlib.sha256(registry_bytes).hexdigest() != receipt['registry_sha256'] or hashes['src/taps_rebuild/' + name + '.py'] != receipt['source_sha256']:
            raise PoolError('Preparation registry or builder identity differs: ' + name)
        receipts[name] = {'bytes': contents, 'value': receipt, 'sha256': hashlib.sha256(contents).hexdigest()}
        registries[name] = registry_bytes
    for method in selected['method_order']:
        definition = selected['methods'][method]
        if definition['profile'] not in ('eagle', 'hass') or definition['source'] not in receipts['sources']['value']['sources']:
            raise PoolError('Selected method source or runtime profile is unavailable')
        for role in ('target', 'drafter'):
            model = receipts['assets']['value']['models'].get(definition[role])
            if not model or model.get('status') != 'verified':
                raise PoolError('Selected model is not prepared: ' + definition[role])
        for benchmark in definition['benchmarks']:
            entry = receipts['sources']['value']['benchmarks'].get(benchmark)
            if not entry or entry['rows'] != 80 or len(entry.get('question_ids', [])) != 80 or entry.get('turns_per_row') not in (1, 2):
                raise PoolError('Pinned 80-question benchmark is not prepared: ' + benchmark)
    return {'config_bytes': config_bytes, 'config': config, 'plan_bytes': plan.read_bytes(), 'plan_sha256': plan_hash, 'source_files': source_files, 'authored_source_hashes': hashes, 'code': code, 'receipts': receipts, 'registries': registries}


def _immutable_bytes(path, contents):
    path = Path(path)
    for parent in [path.parent, *path.parent.parents]:
        if parent.is_symlink():
            raise PoolError('Frozen output parent cannot be a symlink')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != contents:
            raise PoolError('Refusing to overwrite incompatible frozen input: ' + str(path))
        return
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('wb') as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def freeze_run(root, run_dir, run_id, effective, prepared, hardware):
    science = {'run_id': run_id, 'scope': effective['scope'], 'effective_config': effective, 'config_sha256': hashlib.sha256(prepared['config_bytes']).hexdigest(), 'plan_sha256': prepared['plan_sha256'], 'asset_receipt_sha256': prepared['receipts']['assets']['sha256'], 'source_receipt_sha256': prepared['receipts']['sources']['sha256'], 'authored_source_hashes': prepared['authored_source_hashes'], 'code_commit': prepared['code']['code_commit'], 'shard_count': len(hardware['device_bindings'])}
    manifest = {**science, **prepared['code'], 'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root), 'run_dir': str(run_dir), 'active_methods': effective['method_order'], 'scientific_identity_sha256': identity(science), 'hardware_identity_sha256': hardware['hardware_identity_sha256'], 'gpu_inventory': hardware['gpu_inventory'], 'device_bindings': hardware['device_bindings'], 'initial_slurm': hardware['slurm'], 'authored_source_root': str(run_dir / 'inputs/source'), 'asset_receipt_path': str(run_dir / 'inputs/assets.json'), 'source_receipt_path': str(run_dir / 'inputs/sources.json'), 'asset_registry_path': str(run_dir / 'inputs/registries/assets.json'), 'source_registry_path': str(run_dir / 'inputs/registries/sources.json')}
    path = run_dir / 'manifest.json'
    if path.exists():
        if path.is_symlink():
            raise PoolError('Frozen manifest cannot be a symlink')
        old = json.loads(path.read_text())
        if any(key not in old for key in science) or identity({key: old[key] for key in science}) != old.get('scientific_identity_sha256'):
            raise PoolError('Existing manifest scientific content was changed')
        if identity({'gpu_inventory': old.get('gpu_inventory'), 'device_bindings': old.get('device_bindings')}) != old.get('hardware_identity_sha256'):
            raise PoolError('Existing manifest hardware content was changed')
        if old.get('scientific_identity_sha256') != manifest['scientific_identity_sha256']:
            raise PoolError('Scientific identity differs from the existing run; use a new run ID')
        if old.get('hardware_identity_sha256') != manifest['hardware_identity_sha256']:
            raise PoolError('Hardware inventory, driver or device mapping differs; refusing to merge runs')
        for name in ('authored_source_root', 'asset_receipt_path', 'source_receipt_path', 'asset_registry_path', 'source_registry_path'):
            if old.get(name) != manifest[name]:
                raise PoolError('Frozen provenance path differs: ' + name)
        manifest = old
    elif any(path.exists() and any(path.iterdir()) for path in (run_dir / 'raw', run_dir / 'workers')):
        raise PoolError('Existing worker outputs have no compatible frozen manifest')
    _immutable_bytes(run_dir / 'inputs/config.json', prepared['config_bytes'])
    _immutable_bytes(run_dir / 'inputs/TAPS_rebuild_plan.pdf', prepared['plan_bytes'])
    for name in ('assets', 'sources'):
        _immutable_bytes(run_dir / 'inputs' / (name + '.json'), prepared['receipts'][name]['bytes'])
        _immutable_bytes(run_dir / 'inputs/registries' / (name + '.json'), prepared['registries'][name])
    for name, contents in prepared['source_files'].items():
        _immutable_bytes(run_dir / 'inputs/source' / name, contents)
    if not path.exists():
        atomic_json(path, manifest)
    return manifest


def worker_environment(device, frozen_source, root, python, environment=None):
    result = dict(os.environ if environment is None else environment)
    for name in list(result):
        upper = name.upper()
        if upper in {'SSH_AUTH_SOCK', 'KRB5CCNAME', 'GPG_AGENT_INFO', 'SLURM_JWT'} or upper.startswith(('AWS_', 'AZURE_')) or upper.endswith(('_TOKEN', '_PASSWORD', '_SECRET', '_API_KEY', '_ACCESS_KEY_ID', '_ACCESS_KEY', '_SESSION_TOKEN')):
            result.pop(name, None)
    result.update(CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(frozen_source / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', HF_HOME=str(root / 'cache/hf'), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false', UV_NO_CONFIG='1', UV_PROJECT_ENVIRONMENT=str(python.parent.parent))
    result.pop('PYTHONHOME', None)
    result.pop('LD_PRELOAD', None)
    return result


def terminate_children(children, grace=10):
    for child in children:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    while any(child.poll() is None for child in children) and time.monotonic() < deadline:
        time.sleep(0.02)
    for child in children:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in children:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise PoolError('A controller-owned child did not exit') from error


def _expected_cases(method, shard, count, config, sources):
    total = 0
    for benchmark in config['methods'][method]['benchmarks']:
        entry = sources['benchmarks'][benchmark]
        limit = min(entry['rows'], config.get('question_limit', entry['rows']))
        total += len(range(shard, limit, count)) * entry['turns_per_row'] * len(config['temperatures']) * config['timing_repeats'] * 2
    return total


def run_wave(root, run_dir, manifest, hardware, method, sources, lock_fd, stopped=lambda: False):
    config = manifest['effective_config']
    python = Path(config['legacy_runtime_root']) / 'envs' / ('taps-' + config['methods'][method]['profile']) / 'bin/python'
    if not python.is_file() or not os.access(python, os.X_OK):
        raise PoolError('Prepared runtime interpreter is unavailable: ' + str(python))
    children = []
    logs = []
    commands = []
    count = len(hardware['device_bindings'])
    (run_dir / 'logs').mkdir(exist_ok=True)
    try:
        for binding in hardware['device_bindings']:
            shard = binding['shard_index']
            command = [str(python), '-B', '-m', 'taps_rebuild.runner', '--root', str(root), '--config', str(run_dir / 'inputs/config.json'), '--run-dir', str(run_dir), '--manifest', str(run_dir / 'manifest.json'), '--method', method, '--shard-index', str(shard), '--shard-count', str(count)]
            log = (run_dir / 'logs' / (method + '-' + str(shard) + '.log')).open('ab', buffering=0)
            logs.append(log)
            environment = worker_environment(binding['token'], Path(manifest['authored_source_root']), root, python)
            child = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock_fd,))
            children.append(child)
            commands.append({'shard_index': shard, 'device': binding['token'], 'pid': child.pid, 'command': command, 'log': str(log.name)})
        while True:
            if stopped():
                raise PoolInterrupted('Controller received a stop signal')
            codes = [child.poll() for child in children]
            if any(code is not None and code != 0 for code in codes):
                raise PoolError('Worker failed in method ' + method + ': ' + str(codes))
            if all(code is not None for code in codes):
                break
            time.sleep(0.05)
        receipts = []
        manifest_hash = digest(run_dir / 'manifest.json')
        for binding in hardware['device_bindings']:
            shard = binding['shard_index']
            path = run_dir / 'workers' / (method + '-' + str(shard) + '.json')
            deadline = time.monotonic() + 10
            while not path.is_file() and time.monotonic() < deadline:
                time.sleep(0.1)
            receipt = json.loads(path.read_text())
            required = {'status': 'complete', 'method': method, 'shard_index': shard, 'shard_count': count, 'cuda_visible_devices': binding['token'], 'scientific_identity_sha256': manifest['scientific_identity_sha256'], 'manifest_sha256': manifest_hash, 'raw_path': str(run_dir / 'raw' / (method + '-shard' + str(shard) + '.jsonl'))}
            if any(receipt.get(key) != value for key, value in required.items()):
                raise PoolError('Worker completion receipt differs: ' + str(path))
            expected = _expected_cases(method, shard, count, config, sources)
            if expected < 1 or receipt.get('planned_cases') != expected or receipt.get('completed_cases') != expected:
                raise PoolError('Worker did not complete its entire assigned panel: ' + str(path))
            receipts.append({'path': str(path), 'sha256': digest(path), 'completed_cases': expected})
        return {'method': method, 'status': 'complete', 'workers': commands, 'receipts': receipts}
    except BaseException:
        terminate_children(children)
        raise
    finally:
        for log in logs:
            log.close()


def summarize_run(root, run_dir, manifest):
    for name, expected in manifest['authored_source_hashes'].items():
        if digest(Path(manifest['authored_source_root']) / name) != expected:
            raise PoolError('Frozen authored source changed before reporting')
    if digest(run_dir / 'inputs/config.json') != manifest['config_sha256'] or digest(run_dir / 'inputs/TAPS_rebuild_plan.pdf') != manifest['plan_sha256']:
        raise PoolError('Frozen configuration or plan changed before reporting')
    for name, receipt_key, registry_key, hash_key in (('assets', 'asset_receipt_path', 'asset_registry_path', 'asset_receipt_sha256'), ('sources', 'source_receipt_path', 'source_registry_path', 'source_receipt_sha256')):
        if digest(manifest[receipt_key]) != manifest[hash_key]:
            raise PoolError('Frozen preparation receipt changed before reporting: ' + name)
        receipt = json.loads(Path(manifest[receipt_key]).read_text())
        if digest(manifest[registry_key]) != receipt['registry_sha256']:
            raise PoolError('Frozen registry changed before reporting: ' + name)
    sources = json.loads(Path(manifest['source_receipt_path']).read_text())
    config = manifest['effective_config']
    manifest_hash = digest(run_dir / 'manifest.json')
    records = []
    raw_inputs = []
    for method in manifest['active_methods']:
        for binding in manifest['device_bindings']:
            shard = binding['shard_index']
            path = run_dir / 'raw' / (method + '-shard' + str(shard) + '.jsonl')
            content = path.read_bytes()
            if not content.endswith(b'\n'):
                raise PoolError('Worker raw file has an uncommitted tail: ' + str(path))
            expected = set()
            count = len(manifest['device_bindings'])
            for benchmark in config['methods'][method]['benchmarks']:
                entry = sources['benchmarks'][benchmark]
                identifiers = entry['question_ids'][:config.get('question_limit', entry['rows'])][shard::count]
                for identifier in identifiers:
                    for turn in range(entry['turns_per_row']):
                        for repeat in range(config['timing_repeats']):
                            for temperature in config['temperatures']:
                                for mode in ('speculative', 'target_only'):
                                    expected.add(canonical([benchmark, float(temperature), identifier, turn, repeat, mode]))
            observed = set()
            analyzed = 0
            for number, line in enumerate(content.decode().splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('warmup') is True:
                    records.append(row)
                    continue
                if any(row.get(key) != value for key, value in {'run_id': manifest['run_id'], 'scope': manifest['scope'], 'method': method, 'scientific_identity_sha256': manifest['scientific_identity_sha256'], 'manifest_sha256': manifest_hash, 'shard_index': shard, 'shard_count': count}.items()):
                    raise PoolError('Raw worker provenance differs: ' + str(path) + ':' + str(number))
                key = canonical([row['benchmark'], float(row['temperature']), row['question_id'], row['turn_id'], row['repeat'], row['generation_mode']])
                if key in observed or key not in expected:
                    raise PoolError('Raw worker case is duplicated or outside its assigned panel: ' + str(path))
                observed.add(key)
                analyzed += 1
                records.append(row)
            if observed != expected or analyzed != _expected_cases(method, shard, count, config, sources):
                raise PoolError('Raw worker file does not contain its entire assigned panel: ' + str(path))
            raw_inputs.append({'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'completed_records': analyzed})
    path = Path(manifest['authored_source_root']) / 'src/taps_rebuild/metrics.py'
    spec = importlib.util.spec_from_file_location('taps_frozen_metrics_' + manifest['scientific_identity_sha256'][:16], path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    report = metrics.summarize(records, reference_cells=config.get('reference_cells'), relative_tolerance=config.get('relative_acceptance_tolerance', 0.05), bootstrap_samples=config.get('bootstrap_samples', 2000), seed=config.get('base_seed', 20261109))
    report.update(run_id=manifest['run_id'], scope=manifest['scope'], active_methods=manifest['active_methods'], reproduction_gate_scope='active_methods', manifest_sha256=manifest_hash, raw_inputs=raw_inputs)
    observed = [cell for cell in report['cells'] if cell['speculative']['cases'] or cell['target_only']['cases']]
    smoke_valid = manifest['scope'] == 'smoke' and bool(observed) and all(cell['temperature'] == 0 and cell['pairing']['pairing_complete'] and cell['pairing']['t0_checked_pairs'] > 0 and not cell['pairing']['t0_sequence_mismatch_count'] and cell['speculative']['totals']['comparison_tokens'] > 0 and cell['target_only']['totals']['comparison_tokens'] > 0 and not any(reason == 'empty_comparison_tokens' for reason in cell['reproduction_gate']['reasons']) for cell in observed)
    return report, smoke_valid


def run_pool(root, config_path, run_id, expected_gpus, *, smoke=False, methods=None):
    root = Path(root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id):
        raise PoolError('Invalid run ID')
    hardware = allocation(expected_gpus)
    original = json.loads(config_path.read_text())
    config = effective_config(original, methods, smoke, expected_gpus)
    prepared = prerequisites(root, config_path, config)
    run_dir = root / 'outputs/runs' / run_id
    if run_dir.is_symlink() or run_dir.resolve() != run_dir:
        raise PoolError('Run directory cannot use symlink aliases')
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'pool.lock').is_symlink():
        raise PoolError('Run lock cannot be a symlink')
    with (run_dir / 'pool.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PoolError('Another controller or orphaned worker holds this run lock') from error
        manifest = freeze_run(root, run_dir, run_id, config, prepared, hardware)
        state_path = run_dir / 'pool.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'run_id': run_id, 'attempts': []}
        attempt = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'slurm_job_id': hardware['job_id'], 'node': hardware['node'], 'slurm': hardware['slurm'], 'waves': [], 'status': 'running'}
        state['attempts'].append(attempt)
        state.update(status='running', scope=manifest['scope'], active_methods=manifest['active_methods'], scientific_identity_sha256=manifest['scientific_identity_sha256'], manifest_sha256=digest(run_dir / 'manifest.json'), hardware_identity_sha256=hardware['hardware_identity_sha256'], smoke_validation_passed=False, reproduction_gate_passed=False, losslessness_established=False, training_performed=False, publication_performed=False)
        atomic_json(state_path, state)
        stopped = {'signal': None}
        previous = {}

        def stop(signum, frame):
            stopped['signal'] = signum

        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            previous[signum] = signal.signal(signum, stop)
        try:
            for method in manifest['active_methods']:
                if stopped['signal'] is not None:
                    raise PoolInterrupted('Controller interrupted before the next method wave')
                wave = run_wave(root, run_dir, manifest, hardware, method, prepared['receipts']['sources']['value'], lock.fileno(), lambda: stopped['signal'] is not None)
                attempt['waves'].append(wave)
                atomic_json(state_path, state)
            if stopped['signal'] is not None:
                raise PoolInterrupted('Controller interrupted before reporting')
            report, smoke_valid = summarize_run(root, run_dir, manifest)
            if stopped['signal'] is not None:
                raise PoolInterrupted('Controller interrupted during reporting')
            atomic_json(run_dir / 'report.json', report)
            attempt.update(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat())
            state.update(status='complete', smoke_validation_passed=smoke_valid if manifest['scope'] == 'smoke' else None, reproduction_gate_passed=report['reproduction_gate']['passed'], reproduction_gate_status=report['reproduction_gate']['status'], reproduction_gate_scope='active_methods', report_sha256=digest(run_dir / 'report.json'), completed_at_utc=attempt['completed_at_utc'])
            atomic_json(state_path, state)
            return state
        except BaseException as error:
            attempt.update(status='interrupted' if isinstance(error, PoolInterrupted) else 'failed', error_type=type(error).__name__, error=str(error), finished_at_utc=datetime.now(timezone.utc).isoformat())
            state.update(status=attempt['status'], error_type=type(error).__name__, error=str(error), stop_signal=stopped['signal'])
            atomic_json(state_path, state)
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/reproduction.json'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--expected-gpus', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--methods')
    args = parser.parse_args(argv)
    try:
        state = run_pool(args.root, args.config, args.run_id, args.expected_gpus, smoke=args.smoke, methods=args.methods)
    except (OSError, ValueError, PoolError) as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({'status': state['status'], 'scope': state['scope'], 'active_methods': state['active_methods'], 'smoke_validation_passed': state['smoke_validation_passed'], 'reproduction_gate_passed': state['reproduction_gate_passed']}), flush=True)


if __name__ == '__main__':
    main()
