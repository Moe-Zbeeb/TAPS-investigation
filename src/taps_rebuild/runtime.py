import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess

from . import assets, serving, sources


EXCLUDE_NEWER = '2025-05-03T00:00:00Z'
EXPECTED = {'vllm': '0.8.5.post1', 'torch': '2.6.0+cu124', 'torchvision': '0.21.0+cu124', 'torchaudio': '2.6.0+cu124', 'xformers': '0.0.29.post2', 'transformers': '4.51.1', 'tokenizers': '0.21.1', 'huggingface-hub': '0.30.2', 'numpy': '1.26.4', 'fastapi': '0.115.12', 'pydantic': '2.11.4', 'openai': '1.76.2', 'httpx': '0.28.1', 'starlette': '0.46.2', 'uvicorn': '0.34.2'}
PRIMARY_SOURCES = ['https://pypi.org/pypi/vllm/0.8.5.post1/json', 'https://github.com/vllm-project/vllm/blob/v0.8.5.post1/requirements/cuda.txt', 'https://github.com/vllm-project/vllm/blob/v0.8.5.post1/requirements/common.txt', 'https://download.pytorch.org/whl/cu124/torch/', 'https://download.pytorch.org/whl/cu124/torchvision/', 'https://download.pytorch.org/whl/cu124/torchaudio/', 'https://pypi.org/pypi/fastapi/0.115.12/json', 'https://pypi.org/pypi/pydantic/2.11.4/json', 'https://pypi.org/pypi/openai/1.76.2/json']
EXPECTED['setuptools'] = '77.0.3'
PRIMARY_SOURCES += ['https://github.com/vllm-project/vllm/blob/v0.8.5.post1/requirements/build.txt', 'https://github.com/triton-lang/triton/blob/v3.2.0/python/triton/runtime/build.py', 'https://pypi.org/pypi/setuptools/77.0.3/json']


class RuntimeError(ValueError):
    pass


def clean_environment(root, cache):
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith(('UV_', 'PIP_')) or name in ('PYTHONHOME', 'VIRTUAL_ENV', 'LD_PRELOAD', 'PYTHONPATH', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
            environment.pop(name, None)
    environment.update({'CUDA_VISIBLE_DEVICES': '', 'UV_NO_CONFIG': '1', 'UV_CACHE_DIR': str(cache), 'UV_LINK_MODE': 'copy', 'UV_PYTHON_DOWNLOADS': 'never', 'UV_KEYRING_PROVIDER': 'disabled', 'UV_HTTP_TIMEOUT': '300', 'UV_CONCURRENT_DOWNLOADS': '4', 'NETRC': '/dev/null', 'XDG_CACHE_HOME': str(cache / 'state/cache'), 'XDG_CONFIG_HOME': str(cache / 'state/config'), 'XDG_DATA_HOME': str(cache / 'state/data'), 'XDG_STATE_HOME': str(cache / 'state/state'), 'UV_CREDENTIALS_DIR': str(cache / 'state/uv-public'), 'TAPS_RUNTIME_WORKDIR': str(root), 'PIP_CONFIG_FILE': '/dev/null', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'VLLM_USE_V1': '1'})
    return environment


def execute(command, environment, log, timeout=7200, capture=False):
    with log.open('ab') as stream:
        if capture:
            result = subprocess.run(command, cwd=environment.get('TAPS_RUNTIME_WORKDIR'), env=environment, stdout=subprocess.PIPE, stderr=stream, timeout=timeout, check=False)
            stream.write(result.stdout)
        else:
            result = subprocess.run(command, cwd=environment.get('TAPS_RUNTIME_WORKDIR'), env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RuntimeError('Runtime preparation command failed; inspect ' + str(log))
    return result.stdout.decode('utf-8') if capture else None


def cache_directory(root, requested, query=None, mountinfo=None):
    if requested is None:
        return root / '.cache/runtime-vllm-uv', {'kind': 'project_shared', 'node': os.environ.get('SLURMD_NODENAME')}
    assets.require_cpu_slurm()
    node = os.environ.get('SLURMD_NODENAME', '')
    job = os.environ.get('SLURM_JOB_ID', '')
    if not node or socket.gethostname().split('.')[0] != node or not job.isdigit():
        raise RuntimeError('A node-local cache requires execution on the actual allocated Slurm node')
    query = serving.attach.command_output if query is None else query
    record = serving.attach.fields(query(['scontrol', 'show', 'job', '-o', job]))
    owner = re.fullmatch(r'[^()]+\(([0-9]+)\)', record.get('UserId', ''))
    if not owner or int(owner.group(1)) != os.getuid() or record.get('JobState') != 'RUNNING' or record.get('NodeList') != node or record.get('NumNodes') != '1':
        raise RuntimeError('The live job does not own this single-node CPU preparation step')
    path = Path(requested).absolute()
    prefix = 'taps-rebuild-vllm-' + str(os.getuid()) + '-' + job + '-' + node
    if path.parent.resolve() != Path('/tmp').resolve() or not re.fullmatch(re.escape(prefix) + r'(?:-[A-Za-z0-9_-]+)?', path.name) or path.is_symlink():
        raise RuntimeError('Use a TAPS-named direct /tmp cache for this UID, job and node')
    if path.exists() and (not path.is_dir() or path.stat().st_uid != os.getuid()):
        raise RuntimeError('Node-local cache must be an owned directory')
    contents = Path('/proc/self/mountinfo').read_text() if mountinfo is None else mountinfo
    covering = []
    target = path.parent.resolve()
    for line in contents.splitlines():
        fields, filesystem = line.split(' - ', 1)
        mounted = Path(fields.split()[4])
        if target == mounted or mounted in target.parents:
            covering.append((len(mounted.parts), filesystem.split()[0]))
    kind = max(covering)[1] if covering else None
    if kind not in ('ext4', 'xfs', 'tmpfs', 'btrfs', 'zfs', 'overlay'):
        raise RuntimeError('The requested /tmp cache is not on a verified local filesystem')
    return path, {'kind': 'node_local_tmp', 'filesystem_type': kind, 'node': node, 'job_id': job, 'uid': os.getuid(), 'path': str(path)}


def retry_intent(intent, receipt_path, identity, expected_sha256):
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256) or intent.is_symlink() or receipt_path.is_symlink():
        raise RuntimeError('Retry requires the exact prior failed intent SHA256 and regular receipt files')
    intent_bytes, receipt_bytes = intent.read_bytes(), receipt_path.read_bytes()
    prior, receipt = json.loads(intent_bytes), json.loads(receipt_bytes)
    old = prior.get('identity', {})
    if hashlib.sha256(intent_bytes).hexdigest() != expected_sha256 or receipt.get('status') != 'failed' or receipt.get('identity') != old:
        raise RuntimeError('Retry does not identify the matching failed attempt')
    if not isinstance(old.get('source_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', old['source_sha256']) or old['source_sha256'] == identity['source_sha256'] or {key: value for key, value in old.items() if key != 'source_sha256'} != {key: value for key, value in identity.items() if key != 'source_sha256'}:
        raise RuntimeError('Retry permits only the builder source SHA change; requirements, config, versions and paths must match')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = intent.parent / 'retries' / (expected_sha256 + '-' + stamp)
    archive.mkdir(parents=True, exist_ok=False)
    retained = {}
    for name, contents in (('intent.json', intent_bytes), ('runtime.json', receipt_bytes)):
        with (archive / name).open('xb') as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        retained[name] = hashlib.sha256(contents).hexdigest()
    log = intent.parent / 'build.log'
    if log.exists():
        if log.is_symlink() or not log.is_file():
            raise RuntimeError('Prior log must be a regular file')
        with log.open('rb') as source, (archive / 'build.log').open('xb') as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        retained['build.log'] = serving.digest(archive / 'build.log')
    migration = {'status': 'authorized', 'authorization': 'Explicit --retry-from under the exclusive builder lock', 'reason': 'Retry the same pinned serving environment after failed preparation using bounded HTTP concurrency/timeouts and an explicitly verified operational cache', 'prior_intent_sha256': expected_sha256, 'old_identity': old, 'new_identity': identity, 'retained_files_sha256': retained, 'archive': str(archive), 'prior_failure_record': {key: receipt[key] for key in ('failure', 'recovery_observation') if key in receipt}, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    sources.atomic_json(archive / 'migration.json', migration)
    for directory in (archive, archive.parent):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    sources.atomic_json(intent, {'identity': identity, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'retry_from_intent_sha256': expected_sha256, 'migration_path': str(archive / 'migration.json'), 'migration_sha256': serving.digest(archive / 'migration.json')})
    return migration


def prerequisite_intent(intent, receipt_path, identity, expected_sha256, requirements, prior_requirements, failure_directory):
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise RuntimeError('Prerequisite transition requires the exact prior intent SHA256')
    for path in (intent, receipt_path, requirements, prior_requirements):
        if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
            raise RuntimeError('Prerequisite evidence must be owned regular files')
    if not prior_requirements.resolve().is_relative_to(intent.parent.resolve()):
        raise RuntimeError('Prior requirements must be retained under this preparation output')
    prior_bytes, receipt_bytes = intent.read_bytes(), receipt_path.read_bytes()
    prior, receipt = json.loads(prior_bytes), json.loads(receipt_bytes)
    old = prior.get('identity', {})
    if hashlib.sha256(prior_bytes).hexdigest() != expected_sha256 or receipt.get('status') != 'complete' or receipt.get('identity') != old:
        raise RuntimeError('Prerequisite transition must identify the complete prior CPU preparation')
    changed = {'source_sha256', 'requirements_sha256', 'expected_packages'}
    if {k: v for k, v in old.items() if k not in changed} != {k: v for k, v in identity.items() if k not in changed}:
        raise RuntimeError('Prerequisite transition cannot change runtime, package pins, config or paths')
    packages = old.get('expected_packages', {})
    if 'setuptools' in packages or identity.get('expected_packages') != dict(packages, setuptools='77.0.3'):
        raise RuntimeError('Only the observed missing setuptools77.0.3 prerequisite may be added')
    before, after = prior_requirements.read_bytes(), requirements.read_bytes()
    if hashlib.sha256(before).hexdigest() != old.get('requirements_sha256') or hashlib.sha256(after).hexdigest() != identity.get('requirements_sha256') or after != before.rstrip(b'\n') + b'\nsetuptools==77.0.3\n':
        raise RuntimeError('Requirements must add only the exact missing prerequisite')
    if failure_directory.is_symlink() or failure_directory.resolve().parent != (Path(identity['root']) / 'outputs/serving-pools').resolve():
        raise RuntimeError('GPU failure evidence must be a retained serving pool run')
    evidence = {}
    for name in ('state.json', 'manifest.json', 'completion.json'):
        path = failure_directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
            raise RuntimeError('Missing regular GPU failure evidence')
        evidence[name] = path.read_bytes()
    state, manifest, completion = (json.loads(evidence[name]) for name in ('state.json', 'manifest.json', 'completion.json'))
    if state.get('status') != 'failed' or state.get('cleanup', {}).get('status') != 'complete' or manifest.get('runtime_receipt_sha256') != hashlib.sha256(receipt_bytes).hexdigest() or completion.get('controller_state_sha256') != hashlib.sha256(evidence['state.json']).hexdigest() or completion.get('compute_processes') != []:
        raise RuntimeError('GPU failure evidence does not match this prepared environment and complete cleanup')
    logs = sorted((Path(identity['root']) / 'outputs/serving/servers').glob('*' + failure_directory.name + '-gpu*.log'))
    if len(logs) != 8 or not all("ModuleNotFoundError: No module named 'setuptools'" in path.read_text(errors='replace') for path in logs):
        raise RuntimeError('All eight retained server traces must show the observed missing prerequisite')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = intent.parent / 'prerequisite-transitions' / (expected_sha256 + '-' + stamp)
    archive.mkdir(parents=True, exist_ok=False)
    retained = {'intent.json': prior_bytes, 'runtime.json': receipt_bytes, 'requirements-before.txt': before, 'requirements-after.txt': after, 'expected-before.json': json.dumps(packages, sort_keys=True).encode(), 'expected-after.json': json.dumps(identity['expected_packages'], sort_keys=True).encode()}
    retained.update({'gpu-' + name: value for name, value in evidence.items()})
    for name in ('build.log', 'freeze.txt'):
        path = intent.parent / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Prior build evidence must remain regular files')
        retained[name] = path.read_bytes()
    retained.update({'server-gpu-' + str(index) + '.log': path.read_bytes() for index, path in enumerate(logs)})
    hashes = {}
    for name, contents in retained.items():
        with (archive / name).open('xb') as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        hashes[name] = hashlib.sha256(contents).hexdigest()
    migration = {'status': 'authorized', 'authorization': 'Explicit --add-prerequisite-from and exact failed GPU evidence under the builder lock', 'reason': 'Stock Triton3.2 runtime.build imports setuptools during vLLM GPUModelRunner startup', 'old_identity': old, 'new_identity': identity, 'prior_intent_sha256': expected_sha256, 'failure_run': str(failure_directory), 'archive': str(archive), 'retained_files_sha256': hashes, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    sources.atomic_json(archive / 'migration.json', migration)
    for directory in (archive, archive.parent):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    sources.atomic_json(intent, {'identity': identity, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'prerequisite_from_intent_sha256': expected_sha256, 'migration_path': str(archive / 'migration.json'), 'migration_sha256': serving.digest(archive / 'migration.json')})
    return migration


@contextmanager
def failed_attempt(path, identity):
    try:
        yield
    except BaseException as error:
        if path.exists():
            receipt = json.loads(path.read_bytes())
            if receipt.get('identity') == identity:
                receipt.update(status='failed', failure={'recorded_by': 'runtime', 'error_type': type(error).__name__, 'message': str(error) if isinstance(error, RuntimeError) else None, 'observed_at_utc': datetime.now(timezone.utc).isoformat()})
                sources.atomic_json(path, receipt)
        raise


def validate_paths(root, base_python, uv, requirements, config):
    root = Path(root).resolve()
    base_python = Path(base_python).absolute()
    uv = Path(uv).absolute()
    requirements = Path(requirements)
    requirements = requirements if requirements.is_absolute() else root / requirements
    config = Path(config)
    config = config if config.is_absolute() else root / config
    if requirements.resolve() != (root / 'configs/runtime-vllm.txt').resolve() or config.resolve() != (root / 'configs/serving.json').resolve():
        raise RuntimeError('Use this project runtime requirements and serving configuration')
    for path in (base_python, uv):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError('Explicit managed Python and preserved uv executables are required')
    config_value, _, _ = serving.read_config(root, config)
    legacy = Path(config_value['legacy_runtime_root']).resolve()
    if legacy / 'python' not in base_python.resolve().parents:
        raise RuntimeError('Use the preserved managed Python installation under LEGACY/python')
    environment = root / 'envs/serving-vllm'
    for name in ('taps-eagle', 'taps-hass'):
        if environment.resolve() == (legacy / 'envs' / name).resolve():
            raise RuntimeError('The new serving environment must not replace a preserved runtime')
    expected_python = environment / 'bin/python'
    configured = Path(config_value['engines']['vllm']['serving_python'])
    configured = configured if configured.is_absolute() else root / configured
    if configured.absolute() != expected_python.absolute():
        raise RuntimeError('Serving configuration must select ROOT/envs/serving-vllm/bin/python')
    if requirements.is_symlink() or not requirements.is_file():
        raise RuntimeError('Runtime requirements must be a regular project file')
    return root, base_python, uv, requirements, config, environment, config_value


def install_commands(uv, base_python, environment, requirements, exists):
    commands = []
    if not exists:
        commands.append([str(uv), 'venv', '--python', str(base_python), '--no-project', str(environment)])
    commands.append([str(uv), 'pip', 'install', '--python', str(environment / 'bin/python'), '--index-url', 'https://pypi.org/simple', '--exclude-newer', EXCLUDE_NEWER, '--only-binary', 'vllm,torch,torchvision,torchaudio,xformers', '--requirement', str(requirements)])
    return commands


def physical_python(python):
    prefix = Path(python).parent.parent.resolve()
    actual = prefix / 'bin/python'
    if not (prefix / 'pyvenv.cfg').is_file() or not actual.is_file() or not os.access(actual, os.X_OK):
        raise RuntimeError('The actual executable prefix must remain a prepared virtual environment')
    return actual


def package_probe(python, environment, log, runner=execute):
    actual = physical_python(python)
    script = '\n'.join(['import faulthandler', 'faulthandler.dump_traceback_later(60,repeat=True)', 'import importlib.metadata as m,json,sys,torch,torchvision,torchaudio,fastapi,pydantic,openai,triton', 'import vllm.v1.worker.gpu_worker', 'names=' + repr(list(EXPECTED)), 'assert not torch.cuda.is_initialized()', 'print(json.dumps({"python":list(sys.version_info[:3]),"prefix":sys.prefix,"base_prefix":sys.base_prefix,"packages":{name:m.version(name) for name in names},"torch_cuda":torch.version.cuda}))', 'faulthandler.cancel_dump_traceback_later()'])
    value = json.loads(runner([str(actual), '-c', script], environment, log, timeout=600, capture=True).strip().splitlines()[-1])
    if value['python'][:2] != [3, 10] or Path(value['prefix']).resolve() != python.parent.parent.resolve() or value['prefix'] == value['base_prefix'] or value['packages'] != EXPECTED or value['torch_cuda'] != '12.4':
        raise RuntimeError('Installed package versions, virtual environment or CUDA build differ from the pinned runtime')
    value['gpu_execution_tested'] = False
    value['configured_python'] = str(python)
    value['executed_python'] = str(actual)
    return value


def prepare(root, base_python, uv, requirements='configs/runtime-vllm.txt', config='configs/serving.json', runner=execute, cache_dir=None, retry_from=None, prerequisite_from=None, prior_requirements=None, failure_run=None):
    assets.require_cpu_slurm()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('The pinned cp310 CUDA wheels require Linux x86_64')
    root, base_python, uv, requirements, config, environment_path, config_value = validate_paths(root, base_python, uv, requirements, config)
    output = root / 'outputs/preparation/runtime-vllm'
    output.mkdir(parents=True, exist_ok=True)
    log = output / 'build.log'
    cache, cache_record = cache_directory(root, cache_dir)
    cache.mkdir(parents=True, mode=0o700, exist_ok=True)
    for relative in ('tmp', 'state/cache', 'state/config', 'state/data', 'state/state', 'state/uv-public'):
        (cache / relative).mkdir(parents=True, exist_ok=True)
    environment = clean_environment(root, cache)
    environment['TMPDIR'] = str(cache / 'tmp')
    identity = {'root': str(root), 'environment': str(environment_path), 'environment_realpath': str(environment_path.resolve()), 'base_python': str(base_python), 'source_sha256': serving.digest(Path(__file__)), 'requirements_sha256': serving.digest(requirements), 'serving_source_sha256': serving.digest(root / 'src/taps_rebuild/serving.py'), 'serving_config_sha256': serving.digest(config), 'exclude_newer': EXCLUDE_NEWER, 'expected_packages': EXPECTED}
    intent = output / 'intent.json'
    receipt_path = output / 'runtime.json'
    with (output / '.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another runtime preparation process holds the lock') from error
        if prerequisite_from is not None and (retry_from is not None or prior_requirements is None or failure_run is None):
            raise RuntimeError('Prerequisite transition requires separate prior requirements and GPU failure evidence')
        if intent.exists():
            if json.loads(intent.read_bytes()).get('identity') != identity:
                if prerequisite_from is not None:
                    prerequisite_intent(intent, receipt_path, identity, prerequisite_from, requirements, Path(prior_requirements), Path(failure_run))
                elif retry_from is None:
                    raise RuntimeError('Existing runtime preparation provenance differs; preserve the environment for inspection')
                else:
                    retry_intent(intent, receipt_path, identity, retry_from)
            elif retry_from is not None or prerequisite_from is not None:
                raise RuntimeError('The retry flag requires an older failed source identity')
        elif retry_from is not None or prerequisite_from is not None:
            raise RuntimeError('Retry requires an existing failed preparation intent')
        elif environment_path.exists() or environment_path.is_symlink():
            raise RuntimeError('An existing unrecognized environment will not be modified')
        else:
            sources.atomic_json(intent, {'identity': identity, 'created_at_utc': datetime.now(timezone.utc).isoformat()})
        previous = json.loads(receipt_path.read_bytes()) if receipt_path.exists() else {}
        sources.atomic_json(receipt_path, {'status': 'checking', 'identity': identity, 'operational_cache': cache_record, 'network_settings': {'UV_HTTP_TIMEOUT': 300, 'UV_CONCURRENT_DOWNLOADS': 4}, 'job_id': os.environ['SLURM_JOB_ID'], 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'gpu_execution_tested': False, 'model_loading_tested': False, 'scientific_result': False})

        def stage(name):
            receipt = json.loads(receipt_path.read_bytes())
            receipt.update(stage=name, updated_at_utc=datetime.now(timezone.utc).isoformat())
            sources.atomic_json(receipt_path, receipt)
            print(json.dumps({'stage': name, 'status': 'started'}), flush=True)

        with failed_attempt(receipt_path, identity):
            stage('base-interpreter')
            base = json.loads(runner([str(base_python), '-I', '-c', 'import json,sys; print(json.dumps({"version":list(sys.version_info[:2]),"prefix":sys.prefix,"base_prefix":sys.base_prefix}))'], environment, log, timeout=180, capture=True))
            if base['version'] != [3, 10] or base['prefix'] != base['base_prefix']:
                raise RuntimeError('The preserved base interpreter must be managed Python3.10 rather than a virtual environment')
            uv_version = runner([str(uv), '--version'], environment, log, timeout=60, capture=True).strip()
            commands = []
            if previous.get('status') != 'complete' or previous.get('identity') != identity:
                environment_path.parent.mkdir(parents=True, exist_ok=True)
                commands = install_commands(uv, base_python, environment_path, requirements, environment_path.exists())
                for index, command in enumerate(commands):
                    stage('install-' + str(index))
                    runner(command, environment, log)
            python = environment_path / 'bin/python'
            executed_python = physical_python(python)
            stage('cpu-import-preflight')
            packages = package_probe(python, environment, log, runner)
            stage('dependency-check')
            runner([str(uv), 'pip', 'check', '--python', str(python)], environment, log, timeout=180)
            frozen = runner([str(uv), 'pip', 'freeze', '--python', str(python)], environment, log, timeout=180, capture=True)
            freeze_path = output / 'freeze.txt'
            temporary = freeze_path.with_suffix('.tmp')
            with temporary.open('w') as stream:
                stream.write(frozen)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, freeze_path)
            command_checks = {}
            stage('stock-cli-preflight')
            for name in ('vllm_ar', 'vllm_eagle3'):
                checkpoints, asset_receipt_sha = serving.checkpoint_records(root, config_value['methods'][name], verify=False)
                configured_command, _, configured_fingerprint = serving.server_command(root, config_value, name, checkpoints)
                actual_config = copy.deepcopy(config_value)
                actual_config['engines']['vllm']['serving_python'] = str(executed_python)
                command, overrides, fingerprint = serving.server_command(root, actual_config, name, checkpoints)
                checked_environment = dict(environment, **overrides)
                runtime = serving.runtime_check(command, 'vllm', checked_environment)
                command_checks[name] = {'command': command, 'configured_command': configured_command, 'environment_overrides': overrides, 'server_identity_sha256': fingerprint, 'configured_server_identity_sha256': configured_fingerprint, 'runtime': runtime, 'asset_receipt_sha256': asset_receipt_sha}
            module_environment = dict(environment, PYTHONPATH=str(root / 'src'))
            cli_checks = {}
            for name in ('vllm_ar', 'vllm_eagle3'):
                result = runner([str(executed_python), '-B', '-m', 'taps_rebuild.serving', 'command', '--root', str(root), '--config', str(config), '--method', name], module_environment, log, timeout=600, capture=True)
                parsed = json.loads(result.strip().splitlines()[-1])
                if parsed.get('server_identity_sha256') != command_checks[name]['configured_server_identity_sha256']:
                    raise RuntimeError('The installed Python module command disagrees with direct stock command construction')
                cli_checks[name] = {'status': 'passed', 'output_sha256': hashlib.sha256(result.encode()).hexdigest()}
            receipt = {'status': 'complete', 'identity': identity, 'operational_cache': cache_record, 'network_settings': {'UV_HTTP_TIMEOUT': 300, 'UV_CONCURRENT_DOWNLOADS': 4}, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'job_id': os.environ['SLURM_JOB_ID'], 'node': os.environ.get('SLURMD_NODENAME'), 'uv_version': uv_version, 'commands': commands, 'packages': packages, 'freeze_path': str(freeze_path), 'freeze_sha256': serving.digest(freeze_path), 'stock_command_preflights': command_checks, 'module_command_preflights': cli_checks, 'primary_sources': PRIMARY_SOURCES, 'log': str(log), 'verification_scope': 'CPU package imports, exact versions/CUDA build, dependency consistency, stock source hashes, CLI advertisement and module command construction only', 'gpu_execution_tested': False, 'model_loading_tested': False, 'scientific_result': False}
            sources.atomic_json(receipt_path, receipt)
    return receipt_path, receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=assets.DEFAULT_ROOT)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--uv', required=True, type=Path)
    parser.add_argument('--requirements', default='configs/runtime-vllm.txt')
    parser.add_argument('--config', default='configs/serving.json')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--retry-from')
    parser.add_argument('--add-prerequisite-from')
    parser.add_argument('--prior-requirements', type=Path)
    parser.add_argument('--gpu-failure-run', type=Path)
    arguments = parser.parse_args(argv)
    path, receipt = prepare(arguments.root, arguments.python, arguments.uv, arguments.requirements, arguments.config, cache_dir=arguments.cache_dir, retry_from=arguments.retry_from, prerequisite_from=arguments.add_prerequisite_from, prior_requirements=arguments.prior_requirements, failure_run=arguments.gpu_failure_run)
    print(json.dumps({'status': receipt['status'], 'receipt': str(path), 'gpu_execution_tested': False}), flush=True)


if __name__ == '__main__':
    main()
