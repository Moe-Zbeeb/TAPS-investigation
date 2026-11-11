import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time

from . import native_training as native, pool


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic(path, value, replace=True):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.' + str(time.time_ns()) + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(native.canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        temporary.replace(path)
    else:
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink()
    sync_directory(path.parent)


def sealed(value):
    return {**value, 'identity_sha256': pool.identity(value)}


def checked(path):
    value = read(path)
    require(pool.identity({k: v for k, v in value.items() if k != 'identity_sha256'}) == value['identity_sha256'], 'Frozen parallel identity differs')
    return value


def cpu():
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'This action requires CPU Slurm')


def records(path):
    with Path(path).open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n'), 'Frozen JSONL has an incomplete row')
            yield json.loads(line)


def original(bootstrap):
    folder = Path(bootstrap['directory']) / 'teacher'
    for name, expected in bootstrap['legacy']['scripts'].items():
        require(pool.digest(folder / name) == expected, 'Original teacher source changed')
    name = 'frozen_teacher_' + bootstrap['legacy']['identity_sha256']
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, folder / 'corpus_generate.py', submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def bootstrap(path):
    value = checked(Path(path) / 'bootstrap.json')
    require(Path(value['directory']).resolve() == Path(path).resolve() and value['adapter_sha256'] == pool.digest(__file__), 'Parallel deployment changed')
    for item in value['proof_inputs']:
        require(pool.digest(item['path']) == item['sha256'], 'Frozen proof input changed')
    require(pool.digest(value['secondary_schedule']['path']) == value['secondary_schedule']['sha256'], 'Frozen second-node recovery schedule changed')
    original(value)
    return value


def initialize(root, legacy_path, destination):
    cpu()
    root, destination = Path(root).resolve(), Path(destination).resolve()
    require(destination.is_relative_to(root / 'outputs/teacher-generation') and not destination.exists(), 'Choose a fresh parallel generation directory')
    legacy = checked(legacy_path)
    require(legacy['scope'] == 'full_training_data_preparation' and legacy['world_size'] == 8 and legacy['examples'] == 540107, 'Unexpected original teacher population')
    target = native.model_record(read(root / 'outputs/preparation/assets.json'), legacy['target_id'], verify_content=True)
    require(target == legacy['target'], 'Teacher model/tokenizer content differs from original plan')
    destination.mkdir()
    (destination / 'teacher').mkdir()
    (destination / 'proof-inputs').mkdir()
    for name, expected in legacy['scripts'].items():
        source = Path(legacy['directory']) / 'inputs' / name
        require(pool.digest(source) == expected, 'Original source snapshot differs')
        shutil.copyfile(source, destination / 'teacher' / name)
    inputs = []
    batch = legacy['config']['target_generation']['batch_conversations_per_worker']
    require(batch == 64, 'Frozen original generation batch size differs')
    for rank, shard in enumerate(legacy['shards']):
        require(pool.digest(shard['path']) == shard['sha256'], 'Original source shard changed')
        start = (shard['examples'] // batch - 1) * batch
        chosen = [row for index, row in enumerate(records(shard['path'])) if start <= index < start + batch]
        require(len(chosen) == batch, 'Missing complete original tail batch')
        path = destination / 'proof-inputs' / (str(rank) + '.jsonl')
        with path.open('xb') as stream:
            for row in chosen:
                stream.write(native.canonical(row) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        inputs.append({'rank': rank, 'path': str(path), 'sha256': pool.digest(path), 'examples': len(chosen)})
    for name in ('results', 'locks', 'workers', 'chunks', 'base', 'attempts'):
        (destination / name).mkdir()
    python = root / legacy['config']['target_generation']['runtime_python']
    package_query = 'import importlib.metadata as m,json;print(json.dumps({name:m.version(name) for name in ("vllm","torch","transformers")}))'
    packages = json.loads(subprocess.check_output([str(python), '-B', '-c', package_query], text=True))
    require(packages['vllm'] == '0.8.5.post1', 'Original serving runtime version differs')
    schedule = Path(__file__).parents[2] / 'secondary-schedule.json'
    require(read(schedule)['directory'] == str(destination), 'Second-node recovery schedule directory differs')
    value = sealed({'format': 'parallel-teacher-bootstrap-v1', 'directory': str(destination), 'root': str(root), 'legacy_plan_path': str(legacy_path), 'legacy_plan_sha256': pool.digest(legacy_path), 'legacy': legacy, 'proof_inputs': inputs, 'runtime_packages': packages, 'adapter_sha256': pool.digest(__file__), 'secondary_schedule': {'path': str(schedule), 'sha256': pool.digest(schedule)}, 'world_size': 16, 'scope': 'New source-bound parallel teacher plan; preserve committed responses; no byte-identical FP16 continuation claim'})
    atomic(destination / 'bootstrap.json', value, replace=False)
    return value


def validate_outcomes(inputs, outcomes, generator):
    require(len(inputs) == len(outcomes), 'Chunk output coverage differs')
    for source, item in zip(inputs, outcomes):
        require(type(item['ordinal']) is int and item['ordinal'] == source['ordinal'] and item['example_id'] == source['record']['example_id'], 'Teacher output ordinal/example identity differs')
        require(item['status'] in ('complete', 'excluded'), 'Unknown teacher outcome')
        if item['status'] == 'excluded':
            require(item['record'] is None and item['reason'] in {'prompt_exceeds_generation_context', 'response_censored_at_generation_limit', 'empty_or_template_control_response'}, 'Invalid excluded teacher outcome')
            continue
        row, expected = item['record'], source['record']
        changed = {'messages', 'response_regime', 'original_response_complete'}
        require({k: v for k, v in row.items() if k not in changed} == {k: v for k, v in expected.items() if k not in changed}, 'An immutable source field changed')
        require([x for x in row['messages'] if x['role'] == 'user'] == [x for x in expected['messages'] if x['role'] == 'user'], 'Teacher user turns changed')
        require(item['reason'] is None and row['response_regime'] == 'target_generated' and row['original_response_complete'] is True, 'Invalid completed teacher response')
        generator.native_training.native_row(row, canonical_messages=True, split=expected['split'])


def validate_metadata(value, row):
    directory, legacy = Path(value['directory']), value['legacy']
    require(row['bootstrap_identity_sha256'] == value['identity_sha256'] and row['target_identity_sha256'] == legacy['target']['model_identity_sha256'] and row['generation_settings_sha256'] == pool.identity(legacy['config']['target_generation']) and row['original_source_hashes_sha256'] == pool.identity(legacy['scripts']), 'Teacher result target/source/settings provenance differs')
    require(type(row['worker']) is int and 0 <= row['worker'] < 16, 'Unknown parallel worker identity')
    path = Path(row['attempt_path']).resolve()
    require(path.parent == (directory / 'attempts').resolve() and pool.digest(path) == row['attempt_sha256'], 'Teacher hardware attempt changed')
    attempt = read(path)
    hardware = attempt['hardware']
    require(attempt['bootstrap_identity_sha256'] == value['identity_sha256'] and attempt['adapter_sha256'] == value['adapter_sha256'] and hardware['job_id'] == row['slurm_job_id'] and hardware['node'] == row['node'] == ('deep-chungus-10' if row['worker'] < 8 else 'deep-chungus-11'), 'Teacher node/attempt binding differs')
    bindings = hardware['device_bindings']
    require(len(bindings) == 8 and {x['shard_index'] for x in bindings} == set(range(8)) and len({x['uuid'] for x in bindings}) == 8, 'Teacher allocation is not a full unique GPU node')
    binding = next(x for x in bindings if x['shard_index'] == row['worker'] % 8)
    gpu = next(x for x in hardware['gpu_inventory'] if x['uuid'] == binding['uuid'])
    require(row['gpu'] == gpu and row['cuda_visible_devices'] == binding['token'] and gpu['memory_total_mib'] >= 75000, 'Teacher result GPU binding differs')
    require(row['runtime']['packages'] == value['runtime_packages'] and row['runtime']['dtype'] == 'float16' and row['runtime']['python_executable'] == str(Path(value['root']) / legacy['config']['target_generation']['runtime_python']), 'Teacher result runtime differs')
    return {'path': str(path), 'sha256': row['attempt_sha256']}


def journal(path, plan_identity, rank):
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                return
            row = json.loads(line)
            require(row['plan_identity_sha256'] == plan_identity and row['rank'] == rank, 'Original journal provenance differs')
            yield from row['outcomes']


def journal_boundary(path):
    boundary = 0
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                return {'committed_bytes': boundary, 'uncommitted_tail_bytes': len(line), 'uncommitted_tail_sha256': hashlib.sha256(line).hexdigest()}
            boundary += len(line)
    return {'committed_bytes': boundary, 'uncommitted_tail_bytes': 0, 'uncommitted_tail_sha256': None}


def claim(directory, tasks, cursor=0):
    directory = Path(directory)
    for offset in range(len(tasks)):
        index = (cursor + offset) % len(tasks)
        if (directory / 'results' / (str(index) + '.json')).exists():
            continue
        lock = (directory / 'locks' / (str(index) + '.lock')).open('a+b')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        if (directory / 'results' / (str(index) + '.json')).exists():
            lock.close()
            continue
        return index, lock
    return (-1, None) if any(not (directory / 'results' / (str(index) + '.json')).exists() for index in range(len(tasks))) else None


def commit_chunk(directory, plan, index, result, metadata):
    task = plan['tasks'][index]
    require(not (Path(directory) / 'results' / (str(index) + '.json')).exists(), 'Completed chunks must never be overwritten')
    value = {**metadata, **result, 'plan_identity_sha256': plan['identity_sha256'], 'task_index': index, 'input_sha256': task['sha256']}
    atomic(Path(directory) / 'results' / (str(index) + '.json'), value, replace=False)


def job_fields(job):
    result = subprocess.run(['scontrol', 'show', 'job', '-o', str(job)], capture_output=True, text=True)
    if result.returncode == 0:
        return pool._fields(result.stdout)
    result = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job), '-o', 'JobIDRaw,State,User,JobName%40'], text=True)
    rows = [line.split('|') for line in result.splitlines() if line.split('|')[0] == str(job)]
    require(len(rows) == 1, 'Cannot resolve exact job identity')
    row = rows[0]
    require(row[2] == 'mohamadzbib', 'Job accounting owner differs')
    return {'JobState': row[1].split()[0].rstrip('+'), 'JobName': row[3], 'UserId': row[2] + '(' + str(os.getuid()) + ')'}


def prepare(path, secondary_job, old_coordinator, old_step):
    with (Path(path) / 'migration.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return prepare_locked(path, secondary_job, old_coordinator, old_step)


def prepare_locked(path, secondary_job, old_coordinator, old_step):
    cpu()
    value = bootstrap(path)
    directory, legacy = Path(path), value['legacy']
    if (directory / 'plan.json').exists():
        return checked(directory / 'plan.json')
    deadline = time.monotonic() + 3600
    while not all((directory / ('proof-' + str(worker) + '.json')).exists() for worker in range(8, 16)):
        state = subprocess.check_output(['squeue', '-h', '-j', secondary_job, '-o', '%T'], text=True).strip()
        require(state in {'RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'}, 'Second-node qualification failed; old generation was preserved')
        require(time.monotonic() < deadline, 'Second-node qualification deadline reached; old generation was preserved')
        time.sleep(3)
    generator = original(value)
    proofs = []
    proof_descriptors = []
    secondary = job_fields(secondary_job)
    require(secondary['JobState'] == 'RUNNING' and secondary['JobName'] == 'taps-parallel-gen' and secondary['UserId'].startswith('mohamadzbib('), 'Second-node compute is not live and owned')
    for worker in range(8, 16):
        proof = read(directory / ('proof-' + str(worker) + '.json'))
        task = value['proof_inputs'][worker - 8]
        require(proof['bootstrap_identity_sha256'] == value['identity_sha256'] and proof['slurm_job_id'] == secondary_job and proof['worker'] == worker and proof['input_sha256'] == task['sha256'] and proof['generated_tokens'] > 0 and any(row['status'] == 'complete' for row in proof['outcomes']), 'A second-node GPU did not pass actual teacher qualification')
        attempt_descriptor = validate_metadata(value, proof)
        proof_descriptors.append({'path': str(directory / ('proof-' + str(worker) + '.json')), 'sha256': pool.digest(directory / ('proof-' + str(worker) + '.json')), 'attempt': attempt_descriptor})
        validate_outcomes(list(records(task['path'])), proof['outcomes'], generator)
        proofs.extend(proof['outcomes'])
    require(len({row['ordinal'] for row in proofs}) == len(proofs) == 512, 'Second-node qualification population differs')
    holder = read(Path(value['root']).parent / 'rl-infra/cluster/deep10/current.json')
    job = job_fields(old_coordinator)
    require(job['JobState'] in {'RUNNING', 'CANCELLED'} and job['JobName'] == 'taps-data-finish' and job['UserId'].startswith('mohamadzbib('), 'Original coordinator identity changed')
    if job['JobState'] == 'RUNNING':
        steps = subprocess.check_output(['scontrol', 'show', 'step', '-o', holder['job_id']], text=True)
        step_rows = [pool._fields(line) for line in steps.splitlines()]
        attempt_paths = sorted(Path(legacy['directory']).glob('attempt-*.json'), key=lambda p: p.stat().st_mtime_ns)
        attempt = read(attempt_paths[-1])
        require(attempt['plan_identity_sha256'] == legacy['identity_sha256'] and attempt['hardware']['job_id'] == holder['job_id'], 'Current teacher attempt differs from the live holder')
        running = [row for row in step_rows if row.get('Name') in {'taps-teacher-resume', 'taps-teacher-data'} and row.get('State') == 'RUNNING' and row.get('UserId') == str(os.getuid())]
        require(len(running) == 1 and running[0]['StepId'].startswith(holder['job_id'] + '.') and running[0]['StepId'].split('.')[1].isdigit(), 'Original teacher workload step identity changed')
        old_step = running[0]['StepId']
        intent_path = directory / 'handoff-start.json'
        if intent_path.exists():
            intent = read(intent_path)
            require(intent['old_coordinator'] == old_coordinator and intent['old_step'] == old_step and intent['secondary_job'] == secondary_job and intent['original_plan_sha256'] == value['legacy_plan_sha256'], 'Durable migration intent differs from live work')
        else:
            atomic(intent_path, {'old_coordinator': old_coordinator, 'old_step': old_step, 'old_holder': holder['job_id'], 'secondary_job': secondary_job, 'proofs_complete': True, 'original_plan_sha256': value['legacy_plan_sha256'], 'protected_holder_preserved': True, 'created_at_utc': datetime.now(timezone.utc).isoformat()}, replace=False)
        subprocess.run(['scancel', old_coordinator], check=True)
    else:
        require((directory / 'handoff-start.json').exists(), 'An unrelated coordinator cancellation cannot authorize a migration')
        intent = read(directory / 'handoff-start.json')
        require(intent['old_coordinator'] == old_coordinator and intent['original_plan_sha256'] == value['legacy_plan_sha256'], 'Durable migration intent differs')
        old_step = intent['old_step']
    old_holder = old_step.split('.')[0]
    current_steps = subprocess.run(['scontrol', 'show', 'step', '-o', old_holder], capture_output=True, text=True).stdout
    if any(pool._fields(line).get('StepId') == old_step for line in current_steps.splitlines()):
        result = subprocess.run(['scancel', old_step], capture_output=True, text=True)
        if result.returncode:
            current_steps = subprocess.run(['scontrol', 'show', 'step', '-o', old_holder], capture_output=True, text=True).stdout
            require(not any(pool._fields(line).get('StepId') == old_step for line in current_steps.splitlines()), 'Original GPU workload could not be stopped safely')
    locks = []
    try:
        for name in ('continuation.lock', 'controller.lock'):
            lock = (Path(legacy['directory']) / name).open('a+b')
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    require(time.monotonic() < deadline, 'Original generation did not relinquish its writer lock')
                    time.sleep(2)
            locks.append(lock)
        live = pool._fields(subprocess.check_output(['scontrol', 'show', 'job', '-o', holder['job_id']], text=True))
        require(live['JobState'] == 'RUNNING' and live['JobName'] == 'deep10-hold', 'Protected allocation must remain running')
        for _ in range(150):
            active_steps = subprocess.run(['scontrol', 'show', 'step', '-o', old_holder], capture_output=True, text=True).stdout
            if job_fields(old_coordinator)['JobState'] == 'CANCELLED' and not any(pool._fields(line).get('StepId') == old_step for line in active_steps.splitlines()):
                break
            time.sleep(2)
        else:
            raise ValueError('Original workload did not reach the durable stopped boundary')
        preparation = directory / ('preparation-' + os.environ.get('SLURM_RESTART_COUNT', '0') + '-' + str(time.time_ns()))
        preparation.mkdir()
        chunk_folder = preparation / 'chunks'
        chunk_folder.mkdir()
        snapshots = []
        with closing(sqlite3.connect(preparation / 'outcomes.sqlite')) as database:
            database.execute('CREATE TABLE outcomes (ordinal INTEGER PRIMARY KEY, payload BLOB NOT NULL)')
            for rank in range(8):
                source = Path(legacy['directory']) / ('worker-' + str(rank) + '.jsonl')
                stat = source.stat()
                destination = preparation / source.name
                shutil.copyfile(source, destination)
                require(source.stat().st_size == stat.st_size and source.stat().st_mtime_ns == stat.st_mtime_ns and pool.digest(source) == pool.digest(destination), 'Original journal changed while snapshotting')
                snapshots.append({'path': str(destination), 'sha256': pool.digest(destination), 'original_path': str(source), 'size_bytes': stat.st_size, **journal_boundary(destination)})
                for item in journal(destination, legacy['identity_sha256'], rank):
                    try:
                        database.execute('INSERT INTO outcomes VALUES (?, ?)', (item['ordinal'], native.canonical(item)))
                    except sqlite3.IntegrityError as error:
                        raise ValueError('Duplicate committed original outcome') from error
            reused = database.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0]
            redundant = 0
            for item in proofs:
                previous = database.execute('SELECT payload FROM outcomes WHERE ordinal=?', (item['ordinal'],)).fetchone()
                if previous is not None:
                    require(item['example_id'] == json.loads(previous[0])['example_id'], 'Overlapping qualification example differs')
                    redundant += 1
                else:
                    database.execute('INSERT INTO outcomes VALUES (?, ?)', (item['ordinal'], native.canonical(item)))
            buffers, tasks = [[] for _ in range(8)], []
            def flush(rank):
                if not buffers[rank]:
                    return
                p = chunk_folder / (str(len(tasks)) + '.jsonl')
                with p.open('xb') as stream:
                    for row in buffers[rank]:
                        stream.write(native.canonical(row) + b'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                tasks.append({'index': len(tasks), 'path': str(p), 'sha256': pool.digest(p), 'examples': len(buffers[rank])})
                buffers[rank] = []
            expected = set()
            for rank, shard in enumerate(legacy['shards']):
                require(pool.digest(shard['path']) == shard['sha256'], 'Original immutable source shard changed')
                for source in records(shard['path']):
                    ordinal = source['ordinal']
                    require(ordinal not in expected, 'Duplicate source ordinal')
                    expected.add(ordinal)
                    cached = database.execute('SELECT payload FROM outcomes WHERE ordinal=?', (ordinal,)).fetchone()
                    if cached is not None:
                        validate_outcomes([source], [json.loads(cached[0])], generator)
                    else:
                        buffers[rank].append(source)
                        if len(buffers[rank]) == 64:
                            flush(rank)
                flush(rank)
            cached_count = database.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0]
            bounds = database.execute('SELECT MIN(ordinal), MAX(ordinal) FROM outcomes').fetchone()
            require(expected == set(range(legacy['examples'])) and bounds[0] >= 0 and bounds[1] < legacy['examples'], 'Frozen source coverage differs')
            database.commit()
            base = preparation / 'outcomes.jsonl'
            with base.open('xb') as stream:
                for ordinal, payload in database.execute('SELECT ordinal, payload FROM outcomes ORDER BY ordinal'):
                    stream.write(payload + b'\n')
                stream.flush()
                os.fsync(stream.fileno())
        plan = sealed({'format': 'parallel-target-generation-v1', 'bootstrap_identity_sha256': value['identity_sha256'], 'directory': str(directory), 'root': value['root'], 'scope': 'full_training_data_preparation', 'examples': legacy['examples'], 'world_size': 16, 'original_completed_outcomes_preserved': reused, 'qualified_outcomes': len(proofs), 'redundant_qualification_outcomes_preserved_as_evidence': redundant, 'base': {'path': str(base), 'sha256': pool.digest(base), 'examples': cached_count}, 'legacy_snapshots': snapshots, 'tasks': tasks, 'remaining_examples': sum(task['examples'] for task in tasks), 'secondary_job': secondary_job, 'generation_policy': legacy['config']['target_generation'], 'target_identity_sha256': legacy['target']['model_identity_sha256']})
        plan = {k: v for k, v in plan.items() if k != 'identity_sha256'}
        plan['existing_uncommitted_artifacts'] = [{'path': str(p), 'sha256': pool.digest(p), 'size_bytes': p.stat().st_size} for p in Path(legacy['directory']).glob('*.uncommitted-*')]
        plan['second_node_qualification'] = proof_descriptors
        plan = sealed(plan)
        require(plan['base']['examples'] + plan['remaining_examples'] == legacy['examples'], 'Parallel work partition has a gap')
        atomic(directory / 'plan.json', plan, replace=False)
        return plan
    finally:
        for lock in reversed(locks):
            lock.close()


def lock_probe(directory, plan, worker, metadata):
    directory = Path(directory)
    complete = directory / 'lock-health-complete.json'
    if complete.exists():
        require(read(complete)['plan_identity_sha256'] == plan['identity_sha256'], 'Lock health belongs to another plan')
        return
    lock = (directory / 'locks/cross-node-health.lock').open('a+b')
    marker = directory / 'lock-probe-held.json'
    deadline = time.monotonic() + 1200
    try:
        if worker == 8:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            nonce = str(time.time_ns()) + '-' + str(os.getpid())
            atomic(marker, {**metadata, 'plan_identity_sha256': plan['identity_sha256'], 'nonce': nonce})
            while not all((directory / 'workers' / ('lock-' + nonce + '-' + str(index) + '.json')).exists() for index in range(16) if index != 8):
                require(time.monotonic() < deadline, 'Cross-node NFS locking proof timed out')
                time.sleep(1)
            rows = [read(directory / 'workers' / ('lock-' + nonce + '-' + str(index) + '.json')) for index in range(16) if index != 8]
            require(all(row['plan_identity_sha256'] == plan['identity_sha256'] and row['nonce'] == nonce and row['exclusive_lock_observed'] is True for row in rows) and {row['node'] for row in rows} == {'deep-chungus-10', 'deep-chungus-11'} and len({row['gpu']['uuid'] for row in [metadata, *rows]}) == 16, 'Cross-node NFS locking proof differs')
            for row in [metadata, *rows]:
                validate_metadata(bootstrap(directory), row)
            atomic(complete, {'status': 'passed', 'plan_identity_sha256': plan['identity_sha256'], 'nonce': nonce, 'owner': metadata, 'observers': rows}, replace=False)
        else:
            while True:
                require(time.monotonic() < deadline, 'NFS lock owner did not establish an exclusive held marker')
                if not marker.exists():
                    time.sleep(1)
                    continue
                witness = read(marker)
                require(witness['plan_identity_sha256'] == plan['identity_sha256'], 'NFS lock marker belongs to another plan')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    time.sleep(1)
                    continue
                except BlockingIOError:
                    pass
                if read(marker)['nonce'] == witness['nonce']:
                    break
            while not complete.exists():
                require(time.monotonic() < deadline, 'NFS lock proof did not complete')
                witness = read(marker)
                require(witness['plan_identity_sha256'] == plan['identity_sha256'], 'NFS lock marker changed plan')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(lock, fcntl.LOCK_UN)
                except BlockingIOError:
                    if read(marker)['nonce'] == witness['nonce']:
                        nonce = witness['nonce']
                        atomic(directory / 'workers' / ('lock-' + nonce + '-' + str(worker) + '.json'), {**metadata, 'plan_identity_sha256': plan['identity_sha256'], 'nonce': nonce, 'exclusive_lock_observed': True})
                time.sleep(1)
    finally:
        lock.close()


def worker(path, node_rank, local_rank):
    import torch
    from vllm import LLM, SamplingParams

    value = bootstrap(path)
    require(torch.cuda.device_count() == 1 and importlib.metadata.version('vllm') == '0.8.5.post1', 'Pinned single-GPU teacher runtime differs')
    directory, legacy = Path(path), value['legacy']
    generator = original(value)
    worker_id = node_rank * 8 + local_rank
    attempt_path = Path(os.environ['TAPS_PARALLEL_ATTEMPT'])
    attempt = read(attempt_path)
    hardware = attempt['hardware']
    binding = hardware['device_bindings'][local_rank]
    gpu = next(row for row in hardware['gpu_inventory'] if row['uuid'] == binding['uuid'])
    hostname = socket.gethostname().split('.')[0]
    require(hardware['job_id'] == os.environ['SLURM_JOB_ID'] and binding['token'] == os.environ['CUDA_VISIBLE_DEVICES'] and hardware['node'] == hostname, 'Actual GPU binding differs from the node attempt')
    metadata = {'bootstrap_identity_sha256': value['identity_sha256'], 'worker': worker_id, 'node': hostname, 'hostname': socket.gethostname(), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'slurm_restart_count': os.environ.get('SLURM_RESTART_COUNT', '0'), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'gpu': gpu, 'attempt_path': str(attempt_path), 'attempt_sha256': pool.digest(attempt_path), 'generation_settings_sha256': pool.identity(legacy['config']['target_generation']), 'original_source_hashes_sha256': pool.identity(legacy['scripts']), 'runtime': {'packages': {name: importlib.metadata.version(name) for name in ('vllm', 'torch', 'transformers')}, 'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'dtype': 'float16', 'python': sys.version, 'python_executable': sys.executable}, 'target_identity_sha256': legacy['target']['model_identity_sha256']}
    validate_metadata(value, metadata)
    llm = LLM(model=legacy['target']['path'], tokenizer=legacy['target']['path'], tensor_parallel_size=1, dtype='float16', seed=legacy['config']['target_generation']['seed'], gpu_memory_utilization=0.85, max_model_len=legacy['config']['target_generation']['max_context'], max_num_seqs=64, enable_prefix_caching=True, enforce_eager=True, disable_log_stats=True, hf_token=False)
    tokenizer = llm.get_tokenizer()
    try:
        if node_rank == 1:
            task = value['proof_inputs'][local_rank]
            proof_path = directory / ('proof-' + str(worker_id) + '.json')
            if not proof_path.exists():
                inputs = list(records(task['path']))
                result = generator.generate_batch(llm, tokenizer, inputs, legacy['config']['target_generation'], legacy['system_prompt'], SamplingParams)
                validate_outcomes(inputs, result['outcomes'], generator)
                require(result['generated_tokens'] > 0 and any(row['status'] == 'complete' for row in result['outcomes']), 'Second-node GPU did not produce a complete teacher response')
                atomic(proof_path, {**metadata, **result, 'input_sha256': task['sha256']}, replace=False)
        deadline = time.monotonic() + 3600
        while not (directory / 'plan.json').exists():
            require(time.monotonic() < deadline, 'Durable legacy handoff did not complete')
            time.sleep(2)
        plan = checked(directory / 'plan.json')
        require(plan['bootstrap_identity_sha256'] == value['identity_sha256'], 'Shared queue belongs to another bootstrap')
        lock_probe(directory, plan, worker_id, metadata)
        completed, generated, cursor = 0, 0, worker_id
        while True:
            selected = claim(directory, plan['tasks'], cursor)
            if selected is None:
                break
            index, lock = selected
            if index == -1:
                time.sleep(2)
                continue
            try:
                task = plan['tasks'][index]
                require(pool.digest(task['path']) == task['sha256'], 'Frozen task input changed')
                inputs = list(records(task['path']))
                result = generator.generate_batch(llm, tokenizer, inputs, legacy['config']['target_generation'], legacy['system_prompt'], SamplingParams)
                validate_outcomes(inputs, result['outcomes'], generator)
                commit_chunk(directory, plan, index, result, metadata)
                completed += len(result['outcomes'])
                generated += result['generated_tokens']
                atomic(directory / 'workers' / ('progress-' + str(worker_id) + '.json'), {**metadata, 'plan_identity_sha256': plan['identity_sha256'], 'completed_examples_this_attempt': completed, 'generated_tokens_this_attempt': generated, 'last_task_index': index, 'updated_at_utc': datetime.now(timezone.utc).isoformat()})
                cursor = (index + 16) % len(plan['tasks'])
            finally:
                lock.close()
        atomic(directory / 'workers' / ('complete-' + str(worker_id) + '.json'), {**metadata, 'plan_identity_sha256': plan['identity_sha256'], 'completed_examples_this_attempt': completed, 'generated_tokens_this_attempt': generated})
    finally:
        del llm


def node(path, node_rank):
    value = bootstrap(path)
    hardware = pool.allocation(8)
    require(hardware['node'] == ('deep-chungus-10' if node_rank == 0 else 'deep-chungus-11') and all(item['memory_total_mib'] >= 75000 for item in hardware['gpu_inventory']), 'Parallel teacher requires the qualified full eight-A10080 nodes')
    directory = Path(path)
    lock = (directory / ('node-' + str(node_rank) + '.lock')).open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bundle = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location('parallel_context_guard', bundle / 'context.py')
    context = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context)
    import csv
    import io

    policy = read(bundle / 'stale-context.json') if node_rank == 1 else None
    def query():
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=15)
        applications = [{'gpu_uuid': row[0].strip(), 'pid': int(row[1]), 'process_name': row[2].strip(), 'used_memory_mib': int(row[3])} for row in csv.reader(io.StringIO(text)) if row]
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15)
        memory = {row[0].strip(): int(row[1]) for row in csv.reader(io.StringIO(text)) if row}
        blocked = context.blocking_contexts(applications, memory, policy, lambda pid: Path('/proc', str(pid)).exists(), context.host_proc_visible())
        require(set(memory) == {item['uuid'] for item in hardware['gpu_inventory']}, 'GPU memory inventory differs')
        atomic(directory / ('node-' + str(node_rank) + '-gpu-context.json'), {'applications': applications, 'blocking_contexts': blocked, 'free_memory_mib': memory, 'policy': policy})
        return str(blocked) if blocked else ''
    context.wait_for_idle(query, lambda _: None)
    attempt_path = directory / 'attempts' / ('node-' + str(node_rank) + '-' + hardware['job_id'] + '-' + os.environ.get('SLURM_RESTART_COUNT', '0') + '-' + str(time.time_ns()) + '.json')
    atomic(attempt_path, {'hardware': hardware, 'bootstrap_identity_sha256': value['identity_sha256'], 'adapter_sha256': pool.digest(__file__)}, replace=False)
    python = Path(value['root']) / value['legacy']['config']['target_generation']['runtime_python']
    children = []
    try:
        for binding in hardware['device_bindings']:
            rank = binding['shard_index']
            environment = pool.worker_environment(binding['token'], bundle, Path(value['root']), python)
            ipc = Path('/tmp') / ('taps-parallel-' + str(os.getuid()) + '-' + hardware['job_id'] + '-' + str(node_rank) + '-' + str(rank))
            ipc.mkdir(exist_ok=True)
            require(ipc.stat().st_uid == os.getuid() and not ipc.is_symlink(), 'Unsafe parallel teacher IPC path')
            environment.update(PYTHONPATH=str(bundle / 'src'), VLLM_USE_V1='1', VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_RPC_BASE_PATH=str(ipc), TMPDIR=str(ipc), VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1', TAPS_PARALLEL_ATTEMPT=str(attempt_path))
            log = (directory / 'workers' / ('worker-' + str(node_rank * 8 + rank) + '.log')).open('ab')
            command = [str(python), '-B', '-m', 'taps_rebuild.corpus_parallel', 'worker', '--directory', str(directory), '--node-rank', str(node_rank), '--local-rank', str(rank)]
            children.append((subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),), start_new_session=True), log))
        while any(process.poll() is None for process, _ in children):
            require(not any(process.poll() not in (None, 0) for process, _ in children), 'A parallel teacher worker failed')
            time.sleep(2)
        require(all(process.returncode == 0 for process, _ in children), 'Parallel teacher node failed')
    finally:
        for process, _ in children:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process, log in children:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            log.close()
        lock.close()


def queue_complete(directory, plan):
    return all((Path(directory) / 'results' / (str(index) + '.json')).exists() for index in range(len(plan['tasks'])))


def retryable(state):
    return state in {'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}


def ensure_secondary(value, plan):
    directory = Path(value['directory'])
    if queue_complete(directory, plan):
        return None
    with (directory / 'recovery.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pointer = directory / 'secondary-current.json'
        current = read(pointer) if pointer.exists() else {'job_id': plan['secondary_job'], 'restarts': 0}
        job = job_fields(current['job_id'])
        require(job['JobName'] == 'taps-parallel-gen' and job['UserId'].startswith('mohamadzbib('), 'Second-node recovery job identity differs')
        if job['JobState'] in {'RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'}:
            return current['job_id']
        if job['JobState'] == 'COMPLETED':
            deadline = time.monotonic() + 120
            while not queue_complete(directory, plan):
                require(time.monotonic() < deadline, 'Completed second node has missing durable results after the NFS visibility grace')
                time.sleep(2)
            return None
        require(retryable(job['JobState']) and current['restarts'] < 20, 'Second-node nonretryable failure preserved chunks and blocked successors')
        jobs = subprocess.check_output(['scontrol', 'show', 'job', '-o'], text=True)
        schedule = read(value['secondary_schedule']['path'])
        require(schedule['directory'] == str(directory) and schedule['command'][0] == 'sbatch', 'Second-node recovery schedule differs')
        rows = [pool._fields(line) for line in jobs.splitlines()]
        owned = [row for row in rows if row.get('UserId', '').startswith('mohamadzbib(')]
        recovered = [row for row in owned if row.get('JobName') == 'taps-parallel-gen' and row.get('Comment') == schedule['comment'] and row.get('JobState') in {'RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'}]
        require(len(recovered) <= 1, 'Multiple second-node recovery jobs require attention')
        if recovered:
            job_id = recovered[0]['JobId']
        else:
            used = sum(pool._gpu_count(row.get('AllocTRES', '')) or 0 for row in owned if row.get('Partition') == 'high-priority' and row.get('JobState') in {'RUNNING', 'COMPLETING'})
            require(used + 8 <= 18, 'Second-node recovery would exceed the high-priority GPU allowance')
            require('--comment=' + schedule['comment'] in schedule['command'], 'Recovery job lacks its durable submission identity')
            job_id = subprocess.check_output(schedule['command'], text=True).strip().split(';')[0]
        require(job_id.isdigit(), 'Second-node recovery returned an invalid job identifier')
        record = {'job_id': job_id, 'previous_job_id': current['job_id'], 'previous_state': job['JobState'], 'restarts': current['restarts'] + 1, 'plan_identity_sha256': plan['identity_sha256'], 'created_at_utc': datetime.now(timezone.utc).isoformat()}
        history = directory / ('secondary-recovery-' + job_id + '.json')
        if history.exists():
            previous = read(history)
            require({k: v for k, v in previous.items() if k != 'created_at_utc'} == {k: v for k, v in record.items() if k != 'created_at_utc'}, 'Existing secondary recovery evidence differs')
            record = previous
        else:
            atomic(history, record, replace=False)
        atomic(pointer, record)
        return job_id


def attach(path):
    cpu()
    from .data_queue import holder

    value = bootstrap(path)
    plan = checked(Path(path) / 'plan.json')
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
    for key in ('CUDA_VISIBLE_DEVICES', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
        environment.pop(key, None)
    deadline = time.monotonic() + 7 * 86400
    while not queue_complete(path, plan):
        ensure_secondary(value, plan)
        require(time.monotonic() < deadline, 'Protected-node supervisor reached its bounded deadline')
        authority = read(Path(value['root']).parent / 'rl-infra/cluster/deep10/current.json')
        fields = job_fields(authority['job_id'])
        if fields['JobState'] != 'RUNNING':
            require(fields['JobName'] == 'deep10-hold' and (retryable(fields['JobState']) or fields['JobState'] in {'COMPLETED', 'PENDING', 'CONFIGURING'}), 'Protected holder requires attention; no allocation was modified')
            time.sleep(10)
            continue
        job = holder(value['root'])
        command = ['srun', '--jobid=' + job, '--exclusive', '--nodes=1', '--ntasks=1', '--gres=gpu:a100:8', '--cpus-per-task=48', '--kill-on-bad-exit=1', '--job-name=taps-parallel-data', '/usr/bin/python3', '-B', '-S', '-m', 'taps_rebuild.corpus_parallel', 'node', '--directory', str(path), '--node-rank', '0']
        process = subprocess.Popen(command, env=environment)
        while process.poll() is None:
            ensure_secondary(value, plan)
            time.sleep(10)
        if queue_complete(path, plan):
            return
        require(process.returncode == 0 or retryable(job_fields(job)['JobState']) or job_fields(job)['JobState'] == 'COMPLETED', 'Protected-node workload failed; committed chunks remain intact')
        time.sleep(5)


def finish(path, secondary_job):
    with (Path(path) / 'merge.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return finish_locked(path, secondary_job)


def finish_locked(path, secondary_job):
    cpu()
    value = bootstrap(path)
    directory, legacy = Path(path), value['legacy']
    plan = checked(directory / 'plan.json')
    output = Path(value['root']) / 'data/corpora' / (directory.name + '-target-generated')
    if (output / 'completion.json').exists():
        receipt = read(output / 'completion.json')
        require(receipt['plan_identity_sha256'] == plan['identity_sha256'] and all(pool.digest(output / name) == item['sha256'] for name, item in receipt['files'].items()), 'Existing complete corpus changed')
        atomic(directory / 'completion.json', {**receipt, 'corpus_output': str(output)})
        return receipt
    require(secondary_job == plan['secondary_job'], 'Merge second-node identity differs from its frozen plan')
    deadline = time.monotonic() + 7 * 86400
    while not queue_complete(directory, plan):
        ensure_secondary(value, plan)
        require(time.monotonic() < deadline, 'Parallel corpus did not complete within its bounded wait')
        time.sleep(5)
    generator = original(value)
    require(pool.digest(plan['base']['path']) == plan['base']['sha256'], 'Preserved base outcomes changed')
    for item in plan['second_node_qualification']:
        require(pool.digest(item['path']) == item['sha256'] and pool.digest(item['attempt']['path']) == item['attempt']['sha256'], 'Second-node qualification proof changed')
    staging = directory / ('merge-attempt-' + str(time.time_ns()))
    staging.mkdir()
    with closing(sqlite3.connect(staging / 'outcomes.sqlite')) as database:
        database.execute('CREATE TABLE outcomes (ordinal INTEGER PRIMARY KEY, payload BLOB NOT NULL)')
        def add(item):
            ordinal = item['ordinal']
            require(type(ordinal) is int and 0 <= ordinal < plan['examples'], 'Parallel corpus has an invalid ordinal')
            try:
                database.execute('INSERT INTO outcomes VALUES (?, ?)', (ordinal, native.canonical(item)))
            except sqlite3.IntegrityError as error:
                raise ValueError('Parallel corpus has a duplicate ordinal') from error
        for item in records(plan['base']['path']):
            add(item)
        result_proofs = []
        for index, task in enumerate(plan['tasks']):
            source = Path(task['path'])
            require(pool.digest(source) == task['sha256'], 'Frozen chunk changed before merge')
            result_path = directory / 'results' / (str(index) + '.json')
            result = read(result_path)
            require(result['plan_identity_sha256'] == plan['identity_sha256'] and result['task_index'] == index and result['input_sha256'] == task['sha256'], 'Chunk result belongs to another source plan')
            attempt = validate_metadata(value, result)
            validate_outcomes(list(records(source)), result['outcomes'], generator)
            for item in result['outcomes']:
                add(item)
            result_proofs.append({'path': str(result_path), 'sha256': pool.digest(result_path), 'worker': result['worker'], 'slurm_job_id': result['slurm_job_id'], 'attempt': attempt})
        require(database.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0] == plan['examples'], 'Parallel corpus has a missing source ordinal')
        database.commit()
        output.parent.mkdir(exist_ok=True, parents=True)
        publication = output.with_name(output.name + '.merge-' + str(time.time_ns()))
        publication.mkdir()
        handles = {split: ((publication / (split + '.jsonl')).open('xb'), (publication / ('native-' + split + '.jsonl')).open('xb')) for split in ('train', 'dev', 'test')}
        counts = Counter()
        try:
            for expected, (ordinal, encoded) in enumerate(database.execute('SELECT ordinal, payload FROM outcomes ORDER BY ordinal')):
                item = json.loads(encoded)
                require(item['ordinal'] == ordinal == expected, 'Merged source order differs')
                counts['processed'] += 1
                if item['status'] == 'excluded':
                    counts['excluded:' + item['reason']] += 1
                    continue
                row = item['record']
                handles[row['split']][0].write(native.canonical(row) + b'\n')
                handles[row['split']][1].write(native.canonical(generator.native_training.native_row(row, canonical_messages=True, split=row['split'])) + b'\n')
                counts['accepted:' + row['split']] += 1
        finally:
            for pair in handles.values():
                for stream in pair:
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
    require(counts['processed'] == plan['examples'], 'Merged source population differs')
    files = {p.name: {'sha256': pool.digest(p), 'size_bytes': p.stat().st_size} for p in publication.glob('*.jsonl')}
    receipt = {'status': 'complete', 'scope': 'full_training_data_preparation', 'plan_identity_sha256': plan['identity_sha256'], 'plan_path': str(directory / 'plan.json'), 'parent_plan_identity_sha256': legacy['identity_sha256'], 'target_id': legacy['target_id'], 'target_identity_sha256': legacy['target']['model_identity_sha256'], 'response_regime': 'target_generated', 'counts': dict(counts), 'files': files, 'original_committed_responses_preserved': plan['original_completed_outcomes_preserved'], 'teacher_features': 'online native training', 'native_tokenization_filter_performed': False, 'training_started': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    atomic(publication / 'result-provenance.json', result_proofs, replace=False)
    receipt['generation_provenance'] = {'path': 'result-provenance.json', 'sha256': pool.digest(publication / 'result-provenance.json'), 'qualified_plan_sha256': pool.digest(directory / 'plan.json'), 'bootstrap_sha256': pool.digest(directory / 'bootstrap.json')}
    atomic(publication / 'completion.json', receipt, replace=False)
    if output.exists():
        require(not (output / 'completion.json').exists(), 'A completed corpus must never be replaced')
        output.rename(output.with_name(output.name + '.interrupted-' + str(time.time_ns())))
    publication.rename(output)
    sync_directory(output.parent)
    atomic(directory / 'completion.json', {**receipt, 'corpus_output': str(output)})
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('initialize', 'prepare', 'node', 'worker', 'attach', 'finish'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--legacy-plan', type=Path)
    parser.add_argument('--secondary-job')
    parser.add_argument('--old-coordinator')
    parser.add_argument('--old-step')
    parser.add_argument('--node-rank', type=int, choices=(0, 1))
    parser.add_argument('--local-rank', type=int, choices=range(8))
    args = parser.parse_args()
    if args.action == 'initialize':
        initialize(args.root, args.legacy_plan, args.directory)
    elif args.action == 'prepare':
        prepare(args.directory, args.secondary_job, args.old_coordinator, args.old_step)
    elif args.action == 'node':
        node(args.directory, args.node_rank)
    elif args.action == 'worker':
        worker(args.directory, args.node_rank, args.local_rank)
    elif args.action == 'attach':
        attach(args.directory)
    else:
        finish(args.directory, args.secondary_job)


if __name__ == '__main__':
    main()
