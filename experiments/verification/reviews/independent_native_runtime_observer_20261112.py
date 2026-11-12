import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys


CORE = {'native_full_body.py': '40109aae2cdd5b06fa601220175569823f94e0e63d69c44942e93854740046c1', 'native_full_body_queue.py': '1cbd366356bfcfe308aa2cdbd02c4a839b0abcb3b45b6b07db68312c58a95d5b', 'native_gradients.py': '594ee0ee196f9deceecdfe7af7a2bda6b8ee728f2aa9c222748689755c4f4d1a', 'native_training.py': '2aad19f7aefddfb70dd02b78d2eaf9709c90834d2308d055a836be83de5fec51', 'native_recovery.py': '4293c07e350a4bafbb18520050888daf7a7d84a7760c5da9e1e7db7649a34c10', 'pilot_cache.py': 'f9ad06ae2d754c4485bbd22f3e8b98f65ab9706877f40b8961e9f4f1e706ea5e', 'pool.py': 'b00f1930c4eb74ab1312426978487162f6105e5c8b2b478ae0d5894b9898b8c7'}
UPSTREAM = {'main.py': '439c4d8de398c8876f72806d7ba7e6436b5127a4751f3491505a3f51412a4eeb', 'cnets.py': '71295b56d8bd5648a1c630af0995f6a23a13671c8e60fe92636fffd9ee786a9c', 'configs.py': '14ae62877cd9ab12c9076fc70fe940714b9ca980865b00ba974de340df2d1ac5', 'modeling_llama_kv.py': '18adff2c21e659755eb41d6804fa6dd4b4d486b70d3c749693fbb44cff3855b5', 'config.json': '5d7879bc4b6abc55169b2fef99b20d36dbaebc30465a45353d98efc5f3e92d8f', 'ds_config.json': '02b432995b51d8fa1cce7685b7430de7ce14de0419a6729cd9ff0c324c1bec0b'}


def require(value, code):
    if not value:
        raise ValueError(code)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest(path)}


def readiness(directory):
    return {str(rank): {'initial_master_exists': (directory / 'attempts/attempt-1' / ('initial-masters-rank-' + str(rank) + '.pt')).exists(), 'optimizer_window_journal_exists': (directory / 'attempts/attempt-1' / ('full-body-rank-' + str(rank) + '.jsonl')).exists()} for rank in range(8)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=('pre', 'post'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'native_runtime_observer_requires_cpu_slurm')
    require(not args.output.exists(), 'runtime_observer_output_must_be_fresh')
    plan = json.loads(args.plan.read_bytes())
    require(hashlib.sha256(canonical({key: value for key, value in plan.items() if key != 'plan_identity_sha256'})).hexdigest() == plan['plan_identity_sha256'], 'runtime_plan_self_identity_mismatch')
    root, directory, work = Path(plan['project_root']), Path(plan['run_dir']), Path(plan['workdir'])
    require(plan['config']['pipeline_stage'] == 'corrected-profile-v3' and plan['config']['budget'] == {'optimizer_steps': 200} and plan['config']['full_body_v3']['source_sha256'] == CORE, 'runtime_observer_profile_or_source_scope_mismatch')
    require(Path.cwd().resolve() == root.resolve() and Path(sys.prefix).resolve() == (root / 'envs/native-eagle3').resolve(), 'runtime_observer_cwd_or_interpreter_differs')
    require(os.environ.get('PYTHONNOUSERSITE') == '1' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'runtime_observer_environment_differs')
    files = {name: record(directory / 'inputs' / name) for name in CORE}
    upstream = {name: record(work / name) for name in UPSTREAM}
    require(all(value['sha256'] == CORE[name] == plan['frozen_files']['inputs/' + name] for name, value in files.items()) and all(value['sha256'] == UPSTREAM[name] for name, value in upstream.items()), 'runtime_bound_source_changed')
    before = readiness(directory)
    started = datetime.now(timezone.utc).isoformat()
    if args.phase == 'pre':
        require(all(not any(value.values()) for value in before.values()), 'profile_execution_already_visible_before_runtime_pre_observation')
    sys.path.insert(0, str(work))
    names = ['torch', 'torch.utils.checkpoint', 'deepspeed', 'deepspeed.runtime.engine', 'deepspeed.runtime.lr_schedules', 'deepspeed.runtime.zero.stage_1_and_2', 'deepspeed.runtime.fp16.loss_scaler', 'deepspeed.ops.adam.fused_adam', 'transformers', 'accelerate', 'numpy', 'safetensors', 'datasets', 'cnets', 'modeling_llama_kv', 'configs']
    loaded = {name: importlib.import_module(name) for name in names}
    modules = {name: record(module.__file__) for name, module in loaded.items()}
    for name in ('cnets', 'modeling_llama_kv', 'configs'):
        require(Path(modules[name]['path']) == (work / (name + '.py')).resolve(), 'runtime_native_import_resolution_differs')
    torch = loaded['torch']
    require(not torch.cuda.is_initialized(), 'cpu_runtime_observer_initialized_cuda')
    shared = {}
    environment = (root / 'envs/native-eagle3').resolve()
    library_roots = [environment, *(Path(modules[name]['path']).parent.parent for name in ('torch', 'deepspeed', 'numpy'))]
    for line in Path('/proc/self/maps').read_text().splitlines():
        candidate = line.split()[-1]
        if candidate.startswith('/') and '.so' in candidate:
            path = Path(candidate).resolve()
            if path.is_file() and any(path.is_relative_to(base) for base in library_roots):
                shared[str(path)] = record(path)
    extensions = {}
    extension_root = root / 'cache/native-eagle3/deep-chungus-11/torch-extensions'
    for path in extension_root.rglob('*'):
        if path.is_file() and 'fused_adam' in str(path) and path.suffix in ('.so', '.ninja', '.cpp', '.cu', '.h'):
            extensions[str(path.resolve())] = record(path)
    for collection in (files, upstream, modules, shared, extensions):
        require(all(digest(item['path']) == item['sha256'] for item in collection.values()), 'runtime_file_changed_during_observation')
    after = readiness(directory)
    if args.phase == 'pre':
        require(all(not any(value.values()) for value in after.values()), 'profile_execution_began_before_runtime_pre_observation_completed')
    launch = json.loads((directory / 'launch.json').read_bytes())
    require(launch['plan_identity_sha256'] == plan['plan_identity_sha256'] and launch['hardware']['node'] == 'deep-chungus-11', 'runtime_launch_plan_or_physical_node_differs')
    report = {'status': 'independent_native_cpu_runtime_observation', 'phase': args.phase, 'plan': record(args.plan), 'plan_identity_sha256': plan['plan_identity_sha256'], 'observation_started_at_utc': started, 'observation_completed_at_utc': datetime.now(timezone.utc).isoformat(), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'cpu_node': socket.gethostname(), 'python': {'executable': record(sys.executable), 'prefix': str(Path(sys.prefix).resolve()), 'version': sys.version}, 'source_files': files, 'upstream_files': upstream, 'module_files': modules, 'shared_environment_libraries': shared, 'fused_adam_inventory': extensions, 'package_versions': {name: importlib.metadata.version(name) for name in ('torch', 'deepspeed', 'transformers', 'accelerate', 'numpy', 'safetensors', 'datasets')}, 'environment': {key: os.environ.get(key) for key in ('PYTHONNOUSERSITE', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}, 'profile_readiness_before': before, 'profile_readiness_after': after, 'launch_observation': launch, 'observer_source_sha256': digest(__file__), 'scientific_training_admitted': False, 'gpu_work_or_model_loading_performed': False, 'scope': 'Shared source/import/library envelope around the200profile; already completed4/direct/2+2 qualification is outside this temporal interval and needs its own retrospective worker/source bindings.'}
    report['identity_sha256'] = hashlib.sha256(canonical(report)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': digest(args.output)}), flush=True)


if __name__ == '__main__':
    main()
