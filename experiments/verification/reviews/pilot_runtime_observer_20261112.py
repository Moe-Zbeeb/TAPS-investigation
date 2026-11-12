import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys


UPSTREAM = {
    'main.py': '439c4d8de398c8876f72806d7ba7e6436b5127a4751f3491505a3f51412a4eeb',
    'cnets.py': '71295b56d8bd5648a1c630af0995f6a23a13671c8e60fe92636fffd9ee786a9c',
    'configs.py': '14ae62877cd9ab12c9076fc70fe940714b9ca980865b00ba974de340df2d1ac5',
    'modeling_llama_kv.py': '18adff2c21e659755eb41d6804fa6dd4b4d486b70d3c749693fbb44cff3855b5',
    'config.json': '5d7879bc4b6abc55169b2fef99b20d36dbaebc30465a45353d98efc5f3e92d8f',
    'ds_config.json': '02b432995b51d8fa1cce7685b7430de7ce14de0419a6729cd9ff0c324c1bec0b',
}


def require(value, code):
    if not value:
        raise ValueError(code)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path), 'size_bytes': path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--phase', choices=('pre', 'post'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extension-root', type=Path, action='append', default=[])
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'runtime_observer_requires_cpu_slurm')
    require(not args.output.exists(), 'runtime_observer_output_must_be_fresh')
    study_path = args.study / 'manifest.json' if args.study.is_dir() else args.study
    study = json.loads(study_path.read_bytes())
    common = study['common']
    root, source = Path(common['project_root']), Path(common['source_directory'])
    require(Path.cwd().resolve() == root.resolve(), 'runtime_observer_working_directory_differs')
    require(str(source.parent) in os.environ.get('PYTHONPATH', '').split(os.pathsep), 'runtime_observer_bundle_pythonpath_differs')
    require(os.environ.get('PYTHONNOUSERSITE') == '1' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'runtime_observer_environment_differs')
    require(Path(sys.executable).resolve() == (root / 'envs/native-eagle3/bin/python').resolve() and Path(sys.prefix).resolve() == (root / 'envs/native-eagle3').resolve(), 'runtime_observer_interpreter_path_differs')
    source_files = {name: record(source / name) for name in common['source_sha256']}
    require(all(item['sha256'] == common['source_sha256'][name] for name, item in source_files.items()), 'runtime_observer_bundle_source_changed')
    upstream = root / 'repos/EAGLE/eagle/traineagle3'
    upstream_files = {name: record(upstream / name) for name in UPSTREAM}
    require(all(item['sha256'] == UPSTREAM[name] for name, item in upstream_files.items()), 'runtime_observer_upstream_source_changed')
    sys.path.insert(0, str(upstream))
    modules = {}
    names = ['torch', 'deepspeed', 'transformers', 'accelerate', 'numpy', 'safetensors', 'datasets', 'cnets', 'modeling_llama_kv', 'configs', 'deepspeed.runtime.zero.stage_1_and_2', 'deepspeed.runtime.fp16.loss_scaler']
    loaded = {}
    for name in names:
        module = importlib.import_module(name)
        loaded[name] = module
        modules[name] = record(module.__file__)
    for name in ('cnets', 'modeling_llama_kv', 'configs'):
        require(Path(modules[name]['path']) == (upstream / (name + '.py')).resolve(), 'runtime_observer_import_resolution_differs')
    torch = loaded['torch']
    require(not torch.cuda.is_initialized(), 'runtime_observer_initialized_cuda')
    shared = {}
    maps = Path('/proc/self/maps')
    prefix = Path(sys.prefix).resolve()
    library_roots = [prefix, *(Path(modules[name]['path']).parent.parent for name in ('torch', 'deepspeed', 'transformers', 'accelerate', 'numpy', 'safetensors', 'datasets'))]
    if maps.exists():
        for line in maps.read_text().splitlines():
            candidate = line.split()[-1]
            if candidate.startswith('/') and '.so' in candidate:
                path = Path(candidate).resolve()
                if path.is_file() and any(path.is_relative_to(directory) for directory in library_roots):
                    shared[str(path)] = record(path)
    extension_roots = list(args.extension_root)
    if os.environ.get('TORCH_EXTENSIONS_DIR'):
        extension_roots.append(Path(os.environ['TORCH_EXTENSIONS_DIR']))
    extensions = {}
    for directory in extension_roots:
        if directory.exists():
            for path in directory.rglob('*'):
                if path.is_file() and 'fused_adam' in str(path) and path.suffix in ('.so', '.ninja', '.cpp', '.cu', '.h'):
                    extensions[str(path.resolve())] = record(path)
    flags = {'cpu_observer_import_state_only': True, 'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, 'cuda_build_version': torch.version.cuda}
    report = {'status': 'cpu_runtime_envelope_observation', 'phase': args.phase, 'study_manifest_sha256': digest(study_path), 'study_identity_sha256': study['identity_sha256'], 'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'cpu_node': socket.gethostname(), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'python': {'invoked_executable': sys.executable, 'resolved_executable': record(sys.executable), 'prefix': sys.prefix, 'version': sys.version, 'platform': platform.platform()}, 'package_versions': {name: importlib.metadata.version(name) for name in ('torch', 'deepspeed', 'transformers', 'accelerate', 'numpy', 'safetensors', 'datasets')}, 'module_files': modules, 'shared_environment_libraries': shared, 'fused_adam_inventory': extensions, 'fused_adam_search_roots': list(map(str, extension_roots)), 'source_files': source_files, 'upstream_files': upstream_files, 'cpu_observer_flags': flags, 'environment': {key: os.environ.get(key) for key in ('PYTHONPATH', 'PYTHONNOUSERSITE', 'CUBLAS_WORKSPACE_CONFIG', 'TORCH_EXTENSIONS_DIR', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}, 'probe_source_sha256': digest(Path(__file__)), 'gpu_work_or_model_loading_performed': False, 'scientific_training_admitted': False, 'limitations': ['CPU import flags are not GPU backend observations.', 'Shared environment libraries are observed before/after; newly compiled extensions must be classified separately.', 'CPU-node system executable/driver observations do not independently fingerprint GPU-node system libraries. GPU hardware/backend records remain separate evidence.']}
    for collection in (source_files, upstream_files, modules, shared, extensions):
        require(all(digest(Path(item['path'])) == item['sha256'] for item in collection.values()), 'runtime_file_changed_during_observation')
    report['identity_sha256'] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'phase': args.phase, 'report_sha256': digest(args.output)}), flush=True)


if __name__ == '__main__':
    main()
