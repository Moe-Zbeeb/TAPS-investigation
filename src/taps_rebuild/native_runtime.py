import argparse
import base64
import csv
import io
import math
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
import subprocess

from . import assets, pool, runtime, sources


LEGACY_ROOT = Path('/mnt/nfs/home/mohamadzbib/projects/rl-infra')
REVISION = 'cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b'
EXCLUDE_NEWER = '2025-07-12T00:00:00Z'
EXPECTED = {'torch': '2.6.0+cu124', 'torchvision': '0.21.0+cu124', 'torchaudio': '2.6.0+cu124', 'deepspeed': '0.17.1', 'transformers': '4.53.1', 'accelerate': '0.26.0', 'datasets': '3.6.0', 'pyarrow': '20.0.0', 'huggingface-hub': '0.33.4', 'tokenizers': '0.21.2', 'safetensors': '0.5.3', 'numpy': '1.26.4', 'pandas': '2.2.3', 'fsspec': '2025.3.0', 'wandb': '0.20.1', 'sentencepiece': '0.1.99', 'protobuf': '3.19.0', 'pydantic': '2.11.4', 'einops': '0.8.1', 'hjson': '3.1.0', 'msgpack': '1.1.0', 'ninja': '1.11.1.4', 'packaging': '25.0', 'psutil': '7.0.0', 'py-cpuinfo': '9.0.0', 'tqdm': '4.67.1', 'setuptools': '80.9.0', 'wheel': '0.45.1'}
NVIDIA_ML_VERSION = '12.575.51'
NVIDIA_ML_REQUIREMENT = 'nvidia-ml-py @ https://files.pythonhosted.org/packages/db/24/552ebea28f0570b9e65e62b50287a273804c9f997cc1c2dcd4e2d64b9e7d/nvidia_ml_py-12.575.51-py3-none-any.whl#sha256=eb8641800d98ce40a22f479873f34b482e214a7e80349c63be51c3919845446e\n'
EXPECTED['nvidia-ml-py'] = NVIDIA_ML_VERSION
UPSTREAM = {'cnets.py': '71295b56d8bd5648a1c630af0995f6a23a13671c8e60fe92636fffd9ee786a9c', 'configs.py': '14ae62877cd9ab12c9076fc70fe940714b9ca980865b00ba974de340df2d1ac5', 'modeling_llama_kv.py': '18adff2c21e659755eb41d6804fa6dd4b4d486b70d3c749693fbb44cff3855b5'}
PRIMARY_SOURCES = ['https://github.com/SafeAILab/EAGLE/blob/' + REVISION + '/requirements.txt', 'https://github.com/deepspeedai/DeepSpeed/blob/v0.17.1/.github/workflows/nv-nightly.yml', 'https://github.com/deepspeedai/DeepSpeed/blob/v0.17.1/setup.py', 'https://pypi.org/pypi/deepspeed/0.17.1/json', 'https://pypi.org/pypi/datasets/3.6.0/json', 'https://pypi.org/pypi/transformers/4.53.1/json', 'https://pypi.org/pypi/wandb/0.20.1/json', 'https://github.com/wandb/wandb/blob/v0.20.1/wandb/proto/wandb_internal_pb2.py', 'https://docs.astral.sh/uv/reference/environment/#netrc', 'https://github.com/deepspeedai/DeepSpeed/blob/v0.17.1/accelerator/cuda_accelerator.py#L244', 'https://github.com/deepspeedai/DeepSpeed/blob/v0.17.1/deepspeed/model_implementations/transformers/ds_transformer.py#L17', 'https://github.com/deepspeedai/DeepSpeed/blob/v0.17.1/setup.py#L97', 'https://pypi.org/pypi/nvidia-ml-py/12.575.51/json']


class NativeRuntimeError(ValueError):
    pass


def environment_for(root, state, cache=None):
    environment = runtime.clean_environment(root, state / 'cache/uv' if cache is None else cache)
    for name in list(environment):
        if name.startswith('DS_'):
            environment.pop(name)
    environment.pop('VLLM_USE_V1', None)
    environment.update(NETRC='/dev/null', XDG_CACHE_HOME=str(state / 'cache'), XDG_CONFIG_HOME=str(state / 'config'), XDG_DATA_HOME=str(state / 'data'), XDG_STATE_HOME=str(state / 'state'), UV_CREDENTIALS_DIR=str(state / 'state/uv-public'), TMPDIR=str(state / 'tmp'), UV_HTTP_TIMEOUT='300', UV_CONCURRENT_DOWNLOADS='4', DS_BUILD_OPS='0', DS_ACCELERATOR='cuda', WANDB_MODE='disabled', WANDB_DISABLED='true', CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false', GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_NOSYSTEM='1', GIT_TERMINAL_PROMPT='0')
    return environment


def retry_intent(intent, receipt_path, identity, expected_sha256, *, add_nvidia_ml_py=False, requirements=None, dependency_evidence=None):
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256) or intent.is_symlink() or receipt_path.is_symlink():
        raise NativeRuntimeError('Retry requires the exact failed intent SHA256 and regular receipt files')
    intent_bytes, receipt_bytes = intent.read_bytes(), receipt_path.read_bytes()
    old_sha = hashlib.sha256(intent_bytes).hexdigest()
    old_intent, old_receipt = json.loads(intent_bytes), json.loads(receipt_bytes)
    old = old_intent.get('identity', {})
    if old_sha != expected_sha256 or old_receipt.get('status') != 'failed' or old_receipt.get('identity') != old:
        raise NativeRuntimeError('Retry must identify the matching failed preparation attempt')
    dependency_files = None
    if add_nvidia_ml_py:
        dependency_files = validate_dependency_addition(old, identity, old_receipt, intent.parent / 'build.log', requirements, dependency_evidence)
    elif old.get('source_sha256') == identity['source_sha256'] or {key: value for key, value in old.items() if key != 'source_sha256'} != {key: value for key, value in identity.items() if key != 'source_sha256'}:
        raise NativeRuntimeError('Retry permits only a builder source SHA change; packages, inputs and paths must match exactly')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = intent.parent / 'retries' / (old_sha + '-' + stamp)
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
            raise NativeRuntimeError('Previous attempt log must be a regular file')
        with log.open('rb') as source, (archive / 'build.log').open('xb') as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        retained['build.log'] = assets.sha256(archive / 'build.log')
    if dependency_files:
        for name, contents in dependency_files.items():
            with (archive / name).open('xb') as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            retained[name] = hashlib.sha256(contents).hexdigest()
    migration = {'allowed_dependency_addition': {'name': 'nvidia-ml-py', 'version': NVIDIA_ML_VERSION} if add_nvidia_ml_py else None, 'status': 'authorized', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'authorization': 'Explicit --retry-from and --retry-add-nvidia-ml-py under exclusive builder lock' if add_nvidia_ml_py else 'Explicit --retry-from exact prior intent SHA256 under exclusive builder lock', 'reason': 'Retry the same pinned installation with updated operational builder safeguards after a failed attempt', 'prior_intent_sha256': old_sha, 'old_identity': old, 'new_identity': identity, 'retained_files_sha256': retained, 'archive': str(archive), 'prior_failure_record': {key: old_receipt[key] for key in ('failure', 'recovery_observation') if key in old_receipt}}
    sources.atomic_json(archive / 'migration.json', migration)
    for directory in (archive, archive.parent):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    sources.atomic_json(intent, {'identity': identity, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'retry_from_intent_sha256': old_sha, 'migration_path': str(archive / 'migration.json'), 'migration_sha256': assets.sha256(archive / 'migration.json')})
    return migration


def validate_dependency_addition(old, new, receipt, log_path, requirements, evidence):
    permitted = {'source_sha256', 'requirements_sha256', 'expected_packages'}
    if old.get('source_sha256') == new.get('source_sha256') or {key: value for key, value in old.items() if key not in permitted} != {key: value for key, value in new.items() if key not in permitted}:
        raise NativeRuntimeError('Additive retry cannot change model/source/config/environment identities')
    expected = dict(new.get('expected_packages', {}))
    if expected.pop('nvidia-ml-py', None) != NVIDIA_ML_VERSION or expected != old.get('expected_packages') or 'nvidia-ml-py' in old.get('expected_packages', {}):
        raise NativeRuntimeError('Additive retry permits only the declared nvidia-ml-py pin')
    if requirements is None or Path(requirements).is_symlink():
        raise NativeRuntimeError('Additive retry requires regular pinned requirements bytes')
    new_bytes = Path(requirements).read_bytes()
    addition = NVIDIA_ML_REQUIREMENT.encode()
    if not new_bytes.endswith(addition):
        raise NativeRuntimeError('The only permitted requirements addition must be the exact hashed NVIDIA wheel')
    old_bytes = new_bytes[:-len(addition)]
    if b'nvidia-ml-py' in old_bytes or hashlib.sha256(old_bytes).hexdigest() != old.get('requirements_sha256') or hashlib.sha256(new_bytes).hexdigest() != new.get('requirements_sha256'):
        raise NativeRuntimeError('All previous requirement bytes and pins must remain exactly unchanged')
    if receipt.get('stage') != 'dependency-check' or not log_path.is_file() or b'The package `deepspeed` requires `nvidia-ml-py`, but it\'s not installed' not in log_path.read_bytes():
        raise NativeRuntimeError('Additive retry requires the actual missing NVIDIA dependency failure')
    if not isinstance(evidence, dict) or evidence.get('deepspeed_version') != '0.17.1' or 'nvidia-ml-py' not in evidence.get('requires_dist', []) or evidence.get('nvidia_ml_installed_version') not in (None, NVIDIA_ML_VERSION):
        raise NativeRuntimeError('Installed DeepSpeed metadata must declare this exact missing dependency')
    try:
        metadata = base64.b64decode(evidence['metadata_base64'], validate=True)
    except (KeyError, ValueError) as error:
        raise NativeRuntimeError('Installed DeepSpeed metadata evidence bytes are missing') from error
    if hashlib.sha256(metadata).hexdigest() != evidence.get('metadata_sha256') or b'Requires-Dist: nvidia-ml-py' not in metadata:
        raise NativeRuntimeError('Installed DeepSpeed dependency metadata SHA256 differs')
    primary = {'metadata_url': 'https://pypi.org/pypi/nvidia-ml-py/12.575.51/json', 'version': NVIDIA_ML_VERSION, 'uploaded_at': '2025-05-06T20:46:36.457214Z', 'wheel_sha256': 'eb8641800d98ce40a22f479873f34b482e214a7e80349c63be51c3919845446e', 'wheel_bytes': 47547, 'cutoff_unchanged': EXCLUDE_NEWER, 'deepspeed_cuda_conditional_dependency_source': 'https://github.com/deepspeedai/DeepSpeed/blob/v0.17.1/setup.py#L97'}
    return {'requirements-before.txt': old_bytes, 'requirements-after.txt': new_bytes, 'installed-deepspeed-METADATA.txt': metadata, 'dependency-evidence.json': json.dumps(evidence, sort_keys=True, indent=2).encode() + b'\n', 'nvidia-primary-evidence.json': json.dumps(primary, sort_keys=True, indent=2).encode() + b'\n'}


def installed_dependency_evidence(python, environment, log, runner=None):
    runner = execute if runner is None else runner
    actual = runtime.physical_python(python)
    script = '\n'.join([
        'import base64,hashlib,importlib.metadata as m,json,sys',
        'from pathlib import Path',
        'd=m.distribution("deepspeed")',
        'files=[f for f in d.files if f.name=="METADATA" and f.parent.name.endswith(".dist-info")]',
        'assert len(files)==1',
        'p=Path(d.locate_file(files[0]));raw=p.read_bytes()',
        'try: nv=m.version("nvidia-ml-py")',
        'except m.PackageNotFoundError: nv=None',
        'print(json.dumps({"deepspeed_version":d.version,"requires_dist":d.requires,"metadata_path":str(p),"metadata_sha256":hashlib.sha256(raw).hexdigest(),"metadata_base64":base64.b64encode(raw).decode(),"nvidia_ml_installed_version":nv,"prefix":sys.prefix,"base_prefix":sys.base_prefix}))',
    ])
    value = json.loads(runner([str(actual), '-B', '-c', script], environment, log, timeout=300, capture=True).strip().splitlines()[-1])
    if Path(value['prefix']).resolve() != python.parent.parent.resolve() or value['prefix'] == value['base_prefix'] or not Path(value['metadata_path']).resolve().is_relative_to(actual.parent.parent):
        raise NativeRuntimeError('Declared dependency evidence came from another environment')
    value.update(configured_python=str(python), executed_python=str(actual))
    return value


@contextmanager
def failed_attempt(receipt_path, identity):
    try:
        yield
    except BaseException as error:
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_bytes())
            if receipt.get('identity') == identity:
                receipt.update(status='failed', failure={'recorded_by': 'native_runtime', 'error_type': type(error).__name__, 'message': str(error) if isinstance(error, NativeRuntimeError) else None, 'observed_at_utc': datetime.now(timezone.utc).isoformat()})
                sources.atomic_json(receipt_path, receipt)
        raise


def execute(command, environment, log, timeout=7200, capture=False):
    with log.open('ab') as handle:
        result = subprocess.run(command, cwd=environment['TAPS_RUNTIME_WORKDIR'], env=environment, stdout=subprocess.PIPE if capture else handle, stderr=handle if capture else subprocess.STDOUT, timeout=timeout, check=False)
        if capture:
            handle.write(result.stdout)
    if result.returncode:
        raise NativeRuntimeError('Native runtime command failed; inspect ' + str(log))
    return result.stdout.decode() if capture else None


def paths(root, base_python, uv, requirements, config):
    root, base_python, uv = Path(root).resolve(), Path(base_python).absolute(), Path(uv).absolute()
    requirements, config = Path(requirements), Path(config)
    requirements = requirements if requirements.is_absolute() else root / requirements
    config = config if config.is_absolute() else root / config
    if requirements.resolve() != (root / 'configs/runtime-native.txt').resolve() or config.resolve() != (root / 'configs/native-eagle3-training.json').resolve() or requirements.is_symlink():
        raise NativeRuntimeError('Use the project native runtime requirements and native training configuration')
    if not all(path.is_file() and os.access(path, os.X_OK) for path in (base_python, uv)) or (LEGACY_ROOT / 'python').resolve() not in base_python.resolve().parents:
        raise NativeRuntimeError('Explicit preserved uv and managed LEGACY/python Python3.10 are required')
    value = json.loads(config.read_bytes())
    environment = root / 'envs/native-eagle3'
    if value.get('source_revision') != REVISION or value.get('runtime_python') != 'envs/native-eagle3/bin/python':
        raise NativeRuntimeError('Native training configuration does not select the exact source and isolated environment')
    if environment.resolve() in {(LEGACY_ROOT / 'envs' / name).resolve() for name in ('taps-eagle', 'taps-hass')} or environment.resolve() == (root / 'envs/serving-vllm').resolve():
        raise NativeRuntimeError('Native runtime must remain separate from inference and serving environments')
    return root, base_python, uv, requirements, config, environment, value


def source_record(root):
    receipt_path = root / 'outputs/preparation/sources.json'
    receipt = json.loads(receipt_path.read_bytes())
    source = receipt['sources']['EAGLE']
    if receipt.get('status') != 'complete' or source.get('revision') != REVISION:
        raise NativeRuntimeError('Pinned official EAGLE source preparation is required')
    directory = Path(source['path']) / 'eagle/traineagle3'
    if any(assets.sha256(directory / name) != expected for name, expected in UPSTREAM.items()):
        raise NativeRuntimeError('Official native model source content differs')
    return {'directory': str(directory), 'revision': REVISION, 'files_sha256': UPSTREAM, 'source_receipt_sha256': assets.sha256(receipt_path)}


def bootstrap(contents):
    lines = [line for line in contents.decode().splitlines() if line.strip()]
    deep = [line for line in lines if line.startswith('deepspeed @ ')]
    if len(deep) != 1 or '#sha256=' not in deep[0] or sum(line.startswith(('torch @ ', 'torchvision @ ', 'torchaudio @ ')) and '#sha256=' in line for line in lines) != 3:
        raise NativeRuntimeError('Hashed official CUDA wheels and the pinned DeepSpeed source archive are required')
    return ('\n'.join(line for line in lines if line != deep[0]) + '\n').encode()


def commands(uv, base_python, environment, requirements, base_requirements, exists):
    values = []
    if not exists:
        values.append([str(uv), 'venv', '--python', str(base_python), '--no-project', str(environment)])
    common = [str(uv), 'pip', 'install', '--python', str(environment / 'bin/python'), '--index-url', 'https://pypi.org/simple', '--exclude-newer', EXCLUDE_NEWER, '--only-binary', 'torch,torchvision,torchaudio']
    return values + [common + ['--requirement', str(base_requirements)], common + ['--no-build-isolation', '--requirement', str(requirements)]]



def operational_command(command, environment_path):
    value = list(command)
    if '--python' in value:
        index = value.index('--python') + 1
        if value[index] == str(environment_path / 'bin/python'):
            value[index] = str(runtime.physical_python(environment_path / 'bin/python'))
    return value


def gpu_metadata_environment(root, environment, query=None):
    if environment.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise NativeRuntimeError('GPU metadata transition requires the initially CPU-only controller')
    query = pool._command if query is None else query
    inspection = dict(os.environ)
    inspection.pop('CUDA_VISIBLE_DEVICES', None)
    hardware = pool.allocation(8, environment=inspection, command=query)
    holder_path = LEGACY_ROOT / 'cluster/deep10/current.json'
    holder = json.loads(holder_path.read_bytes())
    job = hardware['slurm']['job']
    if holder.get('node') != 'deep-chungus-10' or hardware['node'] != holder['node'] or str(holder.get('job_id')) != hardware['job_id'] or job.get('JobName') != 'deep10-hold' or job.get('Account') != 'grad-students' or job.get('Partition') != 'high-priority' or job.get('QOS') != 'spec-priority':
        raise NativeRuntimeError('Fresh protected holder allocation differs')
    expected = [item['uuid'] for item in hardware['device_bindings']]
    if len(expected) != 8 or len(set(expected)) != 8 or any(not item.startswith('GPU-') for item in expected):
        raise NativeRuntimeError('Exactly eight verified physical GPU UUIDs are required')
    memory_text = query(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    observed = {}
    for row in csv.reader(io.StringIO(memory_text), skipinitialspace=True):
        if len(row) != 3 or row[0].strip() in observed:
            raise NativeRuntimeError('Invalid idle GPU inventory')
        uuid, memory, utilization = [item.strip() for item in row]
        try:
            memory, utilization = float(memory), float(utilization)
        except ValueError as error:
            raise NativeRuntimeError('GPU idle counters are unavailable') from error
        if not math.isfinite(memory) or not math.isfinite(utilization) or memory != 0 or utilization != 0:
            raise NativeRuntimeError('GPU metadata stage requires zero memory use and utilization on every GPU')
        observed[uuid] = {'memory_used_mib': memory, 'utilization_percent': utilization}
    processes = query(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits']).strip()
    if set(observed) != set(expected) or (processes and processes.lower() != 'no running processes found'):
        raise NativeRuntimeError('GPU metadata stage requires every allocated GPU idle with no compute processes')
    value = dict(environment)
    cache = Path(environment['XDG_CACHE_HOME']) / 'gpu-metadata'
    value.update(CUDA_VISIBLE_DEVICES=','.join(expected), TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'), TORCH_EXTENSIONS_DIR=str(cache / 'extensions'))
    for name in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR'):
        Path(value[name]).mkdir(parents=True, exist_ok=True)
    proof = {'hardware': hardware, 'holder_state_sha256': assets.sha256(holder_path), 'idle_gpu_counters': observed, 'compute_processes': [], 'cuda_visible_devices': expected, 'cache_paths': {name: value[name] for name in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR')}, 'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    return value, proof


def gpu_check_lines():
    return [
        'assert torch.cuda.is_available() and torch.cuda.device_count()==8',
        'uuids=__import__("os").environ["CUDA_VISIBLE_DEVICES"].split(",")',
        'assert len(uuids)==8 and len(set(uuids))==8 and all(x.startswith("GPU-") for x in uuids)',
        'gpu_checks=[]',
        'for i in range(8):',
        '    torch.cuda.set_device(i)',
        '    properties=torch.cuda.get_device_properties(i)',
        '    capability=list(torch.cuda.get_device_capability(i))',
        '    assert capability==[8,0]',
        '    tensor=torch.arange(16,device="cuda:"+str(i),dtype=torch.float32)',
        '    actual=float((tensor*tensor).sum().item())',
        '    torch.cuda.synchronize(i)',
        '    assert actual==1240.0',
        '    gpu_checks.append({"logical_index":i,"uuid_binding":uuids[i],"name":properties.name,"capability":capability,"total_memory_bytes":properties.total_memory,"arithmetic_sum_squares":actual})',
        'del tensor',
    ]


def validate_gpu_checks(value, environment):
    checks = value.get('gpu_checks', [])
    expected = environment.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(checks) != 8 or [item.get('logical_index') for item in checks] != list(range(8)) or [item.get('uuid_binding') for item in checks] != expected or len(set(expected)) != 8 or any(item.get('capability') != [8, 0] or item.get('arithmetic_sum_squares') != 1240.0 or not isinstance(item.get('name'), str) or not item['name'] or type(item.get('total_memory_bytes')) is not int or item['total_memory_bytes'] <= 0 for item in checks):
        raise NativeRuntimeError('Actual all-eight CUDA capability/arithmetic proof differs')
    if value.get('cuda_context_initialized') is not True or value.get('torch_cuda') != '12.4':
        raise NativeRuntimeError('GPU metadata proof did not initialize the expected real CUDA runtime')


def gpu_hardware_probe(python, environment, log, runner=execute):
    actual = runtime.physical_python(python)
    script = '\n'.join(['import json,torch', *gpu_check_lines(), 'print(json.dumps({"torch_cuda":torch.version.cuda,"cuda_context_initialized":torch.cuda.is_initialized(),"gpu_checks":gpu_checks,"scope":"actual_all_eight_small_cuda_arithmetic_only"}))'])
    result = json.loads(runner([str(actual), '-B', '-c', script], environment, log, timeout=300, capture=True).strip().splitlines()[-1])
    validate_gpu_checks(result, environment)
    result.update(configured_python=str(python), executed_python=str(actual))
    return result


def probe(python, environment, log, source, configuration, fixture, runner=execute, gpu_metadata=False):
    if not gpu_metadata:
        raise NativeRuntimeError('Pinned DeepSpeed0.17.1 CUDA/Triton imports require explicit --gpu-metadata and a guarded idle full node')
    actual = runtime.physical_python(python)
    script = '\n'.join([
        'import importlib.metadata as m,json,sys,torch',
        'assert not torch.cuda.is_initialized()',
        'import torchvision,torchaudio,deepspeed,accelerate,datasets,wandb,numpy,safetensors,sentencepiece,google.protobuf',
        'from packaging.specifiers import SpecifierSet',
        'from accelerate.utils import set_seed',
        'from safetensors import safe_open',
        'from deepspeed.ops.adam import FusedAdam',
        'from deepspeed.git_version_info import installed_ops,accelerator_name',
        'from deepspeed.accelerator import get_accelerator',
        'sys.path.insert(0,' + repr(source['directory']) + ')',
        'from cnets import Model,padding',
        'from configs import EConfig',
        'from modeling_llama_kv import LlamaForCausalLM',
        'data=datasets.load_dataset("json",data_files=' + repr(str(fixture)) + ',split="train",cache_dir=' + repr(str(fixture.parent / 'datasets-cache')) + ')',
        'data=data.shuffle(seed=42).map(lambda batch:{"length":[len(value) for value in batch["text"]]},batched=True,num_proc=1,load_from_cache_file=False)',
        'data.set_format("torch",columns=["length"])',
        'assert len(data)==2 and sorted(int(data[i]["length"]) for i in range(2))==[1,2]',
        *gpu_check_lines(),
        'required=' + repr(configuration.get('required_packages', {})),
        'assert all(not spec or m.version(name) in SpecifierSet(spec) for name,spec in required.items())',
        'names=' + repr(list(EXPECTED)),
        'print(json.dumps({"python":list(sys.version_info[:3]),"prefix":sys.prefix,"base_prefix":sys.base_prefix,"packages":{name:m.version(name) for name in names},"torch_cuda":torch.version.cuda,"cuda_context_initialized":torch.cuda.is_initialized(),"deepspeed_build_accelerator":accelerator_name,"deepspeed_runtime_accelerator":get_accelerator().device_name(),"deepspeed_installed_ops":installed_ops,"native_model_module_imported":Model.__module__,"dataset_fixture_rows":len(data),"gpu_checks":gpu_checks,"gpu_basic_arithmetic_tested":True,"model_loading_tested":False,"optimizer_execution_tested":False}))',
    ])
    output = runner([str(actual), '-B', '-c', script], environment, log, timeout=300, capture=True)
    value = json.loads(output.strip().splitlines()[-1])
    if value['python'][:2] != [3, 10] or Path(value['prefix']).resolve() != python.parent.parent.resolve() or value['prefix'] == value['base_prefix'] or value['packages'] != EXPECTED or value['torch_cuda'] != '12.4' or value['deepspeed_build_accelerator'] != 'cuda' or value['deepspeed_runtime_accelerator'] != 'cuda' or value['dataset_fixture_rows'] != 2:
        raise NativeRuntimeError('Native package, CPU dataset fixture, CUDA build or isolated environment preflight differs')
    validate_gpu_checks(value, environment)
    value.update(configured_python=str(python), executed_python=str(actual))
    return value


def prepare(root, base_python, uv, requirements='configs/runtime-native.txt', config='configs/native-eagle3-training.json', runner=execute, retry_from=None, cache_dir=None, gpu_metadata=False, retry_add_nvidia_ml_py=False):
    assets.require_cpu_slurm()
    if gpu_metadata is not True:
        raise NativeRuntimeError('Pinned DeepSpeed0.17.1 CUDA/Triton metadata imports initialize CUDA; rerun explicitly with --gpu-metadata after all eight allocated GPUs are idle')
    if retry_add_nvidia_ml_py and retry_from is None:
        raise NativeRuntimeError('The explicit additive dependency retry also requires --retry-from exact failed intent SHA256')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise NativeRuntimeError('Pinned native CUDA wheels require Linux x86_64')
    root, base_python, uv, requirements, config, environment_path, configuration = paths(root, base_python, uv, requirements, config)
    source = source_record(root)
    output = root / 'outputs/preparation/runtime-native'
    output.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        cache = root / '.cache/runtime-native/cache/uv'
        cache_record = {'kind': 'project_shared', 'path': str(cache), 'node': os.environ.get('SLURMD_NODENAME')}
    else:
        cache, cache_record = runtime.cache_directory(root, cache_dir)
        cache.mkdir(parents=True, mode=0o700, exist_ok=True)
    state = root / '.cache/runtime-native' if cache_dir is None else cache / 'native-state'
    for path in ('cache/uv', 'config', 'data', 'state/uv-public', 'tmp'):
        (state / path).mkdir(parents=True, exist_ok=True)
    environment = environment_for(root, state, None if cache_dir is None else cache)
    environment['TAPS_RUNTIME_WORKDIR'] = str(root)
    log, receipt_path = output / 'build.log', output / 'runtime.json'
    identity = {'root': str(root), 'environment': str(environment_path), 'environment_realpath': str(environment_path.resolve()), 'base_python': str(base_python), 'source_sha256': assets.sha256(Path(__file__)), 'requirements_sha256': assets.sha256(requirements), 'native_training_config_sha256': assets.sha256(config), 'source': source, 'expected_packages': EXPECTED, 'exclude_newer': EXCLUDE_NEWER}
    intent = output / 'intent.json'
    with (output / '.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise NativeRuntimeError('Another native runtime builder holds the lock') from error
        if intent.exists():
            if json.loads(intent.read_bytes()).get('identity') != identity:
                if retry_from is None:
                    raise NativeRuntimeError('Existing native runtime preparation provenance differs')
                evidence = None
                if retry_add_nvidia_ml_py:
                    evidence_log = output / ('dependency-evidence-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.log')
                    evidence = installed_dependency_evidence(environment_path / 'bin/python', environment, evidence_log, runner)
                    evidence.update(probe_log_path=str(evidence_log), probe_log_sha256=assets.sha256(evidence_log))
                retry_intent(intent, receipt_path, identity, retry_from, add_nvidia_ml_py=retry_add_nvidia_ml_py, requirements=requirements, dependency_evidence=evidence)
            elif retry_from is not None:
                raise NativeRuntimeError('The retry flag requires an older failed builder source identity')
        elif retry_from is not None:
            raise NativeRuntimeError('Retry requires an existing failed preparation intent')
        elif environment_path.exists() or environment_path.is_symlink():
            raise NativeRuntimeError('An unrecognized existing native environment will not be modified')
        else:
            sources.atomic_json(intent, {'identity': identity, 'created_at_utc': datetime.now(timezone.utc).isoformat()})
        previous = json.loads(receipt_path.read_bytes()) if receipt_path.exists() else {}
        receipt = {'status': 'checking', 'identity': identity, 'job_id': os.environ['SLURM_JOB_ID'], 'operational_cache': cache_record, 'state_directory': str(state), 'cache_safety_primary_source': 'https://docs.astral.sh/uv/concepts/cache/#cache-safety', 'gpu_execution_tested': False, 'optimizer_execution_tested': False, 'model_loading_tested': False, 'scientific_result': False, 'network_settings': {'UV_HTTP_TIMEOUT': 300, 'UV_CONCURRENT_DOWNLOADS': 4}, 'explicit_additive_dependency_retry': retry_add_nvidia_ml_py, 'gpu_metadata': {'explicitly_requested': True, 'stage_guards': [], 'basic_arithmetic_checks': [], 'verification_scope': 'Real all-eight CUDA metadata/arithmetic plus unchanged DeepSpeed build/import only; model and optimizer execution untested'}}

        def stage(name):
            receipt.update(stage=name, updated_at_utc=datetime.now(timezone.utc).isoformat())
            sources.atomic_json(receipt_path, receipt)
            print(json.dumps({'stage': name, 'status': 'started'}), flush=True)

        with failed_attempt(receipt_path, identity):
            stage('base-interpreter')
            base = json.loads(runner([str(base_python), '-I', '-c', 'import json,sys;print(json.dumps({"python":list(sys.version_info[:2]),"prefix":sys.prefix,"base_prefix":sys.base_prefix}))'], environment, log, timeout=180, capture=True).strip().splitlines()[-1])
            if base['python'] != [3, 10] or base['prefix'] != base['base_prefix']:
                raise NativeRuntimeError('Use preserved managed Python3.10 rather than an existing virtual environment')
            uv_version = runner([str(uv), '--version'], environment, log, timeout=60, capture=True).strip()
            base_path = output / 'bootstrap-requirements.txt'
            base_path.write_bytes(bootstrap(requirements.read_bytes()))
            executed = []
            if previous.get('status') != 'complete':
                environment_path.parent.mkdir(parents=True, exist_ok=True)
                for index, command in enumerate(commands(uv, base_python, environment_path, requirements, base_path, environment_path.exists())):
                    selected_environment = environment
                    if '--no-build-isolation' in command:
                        stage('gpu-metadata-guard-before-basic-arithmetic')
                        selected_environment, guard = gpu_metadata_environment(root, environment)
                        receipt['gpu_metadata']['stage_guards'].append({'stage': 'before-basic-arithmetic', **guard})
                        receipt['gpu_context_initialization_allowed'] = True
                        stage('gpu-metadata-all-eight-basic-arithmetic')
                        receipt['gpu_metadata']['basic_arithmetic_checks'].append(gpu_hardware_probe(environment_path / 'bin/python', selected_environment, log, runner))
                        receipt['gpu_execution_tested'] = True
                        receipt['gpu_basic_arithmetic_tested'] = True
                        selected_environment, guard = gpu_metadata_environment(root, environment)
                        receipt['gpu_metadata']['stage_guards'].append({'stage': 'before-deepspeed-build', **guard})
                        selected_environment['TORCH_CUDA_ARCH_LIST'] = '8.0'
                    stage('gpu-metadata-deepspeed-install' if '--no-build-isolation' in command else 'cpu-install-' + str(index))
                    actual_command = operational_command(command, environment_path)
                    runner(actual_command, selected_environment, log)
                    executed.append(actual_command)
            python = environment_path / 'bin/python'
            receipt['python_execution'] = {'configured_python': str(python), 'executed_python': str(runtime.physical_python(python))}
            fixture = output / 'fixture.jsonl'
            fixture.write_text('{"text":"a"}\n{"text":"bc"}\n')
            stage('gpu-metadata-guard-before-import-preflight')
            gpu_environment, guard = gpu_metadata_environment(root, environment)
            receipt['gpu_metadata']['stage_guards'].append({'stage': 'before-import-preflight', **guard})
            gpu_environment['TORCH_CUDA_ARCH_LIST'] = '8.0'
            receipt['gpu_context_initialization_allowed'] = True
            stage('gpu-metadata-import-and-native-source-preflight')
            packages = probe(python, gpu_environment, log, source, configuration, fixture, runner, gpu_metadata=True)
            _, guard = gpu_metadata_environment(root, environment)
            receipt['gpu_metadata']['stage_guards'].append({'stage': 'after-import-preflight-process-exit', **guard})
            receipt['gpu_execution_tested'] = True
            receipt['gpu_basic_arithmetic_tested'] = True
            receipt['gpu_context_initialization_allowed'] = True
            stage('dependency-check')
            runner(operational_command([str(uv), 'pip', 'check', '--python', str(python)], environment_path), environment, log, timeout=180)
            frozen = runner(operational_command([str(uv), 'pip', 'freeze', '--python', str(python)], environment_path), environment, log, timeout=180, capture=True)
            freeze_path = output / 'freeze.txt'
            temporary = freeze_path.with_suffix('.tmp')
            with temporary.open('w') as handle:
                handle.write(frozen)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, freeze_path)
            toolchain = {}
            for name in ('nvcc', 'c++', 'ninja'):
                executable = shutil.which(name, path=str(environment_path / 'bin') + os.pathsep + environment.get('PATH', ''))
                toolchain[name] = {'path': executable, 'available': executable is not None}
                if executable:
                    value = runner([executable, '--version'], environment, log, timeout=30, capture=True)
                    toolchain[name]['version_output'] = value[:4000]
            receipt.update(status='complete', stage='complete', job_id=os.environ['SLURM_JOB_ID'], node=os.environ.get('SLURMD_NODENAME'), uv_version=uv_version, commands=executed, packages=packages, freeze_path=str(freeze_path), freeze_sha256=assets.sha256(freeze_path), toolchain=toolchain, deepspeed_build_ops=False, fused_adam_cuda_jit_tested=False, source_model_imports_tested=True, primary_sources=PRIMARY_SOURCES, log=str(log), verification_scope='Guarded full-node all-eight real CUDA capability and small arithmetic checks, unchanged DeepSpeed CUDA metadata build/import, native model module import without weights, local CPU JSON dataset API, exact packages and dependency consistency; model loading, CUDA FusedAdam JIT and optimizer execution remain untested')
            sources.atomic_json(receipt_path, receipt)
    return receipt_path, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare isolated native EAGLE3 dependencies. Stock DeepSpeed0.17.1 CUDA/Triton imports require explicit idle full-node GPU metadata access; no model or optimizer is executed.')
    parser.add_argument('--root', type=Path, default=assets.DEFAULT_ROOT)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--uv', required=True, type=Path)
    parser.add_argument('--requirements', default='configs/runtime-native.txt')
    parser.add_argument('--config', default='configs/native-eagle3-training.json')
    parser.add_argument('--retry-from')
    parser.add_argument('--retry-add-nvidia-ml-py', action='store_true', help='Authorize only the declared CUDA dependency addition at its exact pinned wheel, with byte-exact prior requirements and failed DS metadata evidence')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--gpu-metadata', action='store_true', help='Explicitly permit guarded all-eight real CUDA metadata/basic arithmetic during DeepSpeed build/import; package pins and bytes stay unchanged')
    arguments = parser.parse_args(argv)
    path, receipt = prepare(arguments.root, arguments.python, arguments.uv, arguments.requirements, arguments.config, retry_from=arguments.retry_from, cache_dir=arguments.cache_dir, gpu_metadata=arguments.gpu_metadata, retry_add_nvidia_ml_py=arguments.retry_add_nvidia_ml_py)
    print(json.dumps({'status': receipt['status'], 'receipt': str(path), 'gpu_execution_tested': receipt['gpu_execution_tested'], 'model_loading_tested': False, 'optimizer_execution_tested': False}), flush=True)


if __name__ == '__main__':
    main()
