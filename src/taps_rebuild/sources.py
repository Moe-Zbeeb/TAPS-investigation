import argparse
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


DEFAULT_ROOT = Path('/mnt/nfs/home/mohamadzbib/projects/taps-rebuild')


class SourcePreparationError(ValueError):
    pass


def atomic_bytes(path, contents):
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def require_cpu_slurm():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SourcePreparationError('Source preparation requires Slurm and empty CUDA_VISIBLE_DEVICES')


def git_environment():
    environment = os.environ.copy()
    environment.update({'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_COUNT': '2', 'GIT_CONFIG_KEY_0': 'core.excludesfile', 'GIT_CONFIG_VALUE_0': '/dev/null', 'GIT_CONFIG_KEY_1': 'core.attributesfile', 'GIT_CONFIG_VALUE_1': '/dev/null', 'GIT_TERMINAL_PROMPT': '0', 'GIT_ASKPASS': '/usr/bin/false', 'SSH_ASKPASS': '/usr/bin/false', 'GCM_INTERACTIVE': 'Never', 'GIT_OPTIONAL_LOCKS': '0', 'GIT_ALLOW_PROTOCOL': 'https'})
    for name in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_CONFIG', 'GIT_CONFIG_PARAMETERS'):
        environment.pop(name, None)
    return environment


def run_git(directory, *arguments):
    result = subprocess.run(['git', '-C', str(directory), *arguments], cwd=directory, env=git_environment(), capture_output=True, timeout=600)
    if result.returncode != 0:
        raise SourcePreparationError('Git command failed: ' + arguments[0])
    return result.stdout


def relative_path(value, prefix=None):
    path = Path(value)
    if path.is_absolute() or not path.parts or '..' in path.parts or (prefix and path.parts[0] != prefix):
        raise SourcePreparationError('Invalid relative source or benchmark path')
    return path


def validate_registry(registry):
    if registry.get('schema_version') != 1:
        raise SourcePreparationError('Unsupported source registry schema')
    sources = registry.get('sources', [])
    benchmarks = registry.get('benchmarks', [])
    names = {source['name'] for source in sources}
    if not sources or len(names) != len(sources):
        raise SourcePreparationError('Source names must be nonempty and unique')
    paths = []
    for source in sources:
        if not re.fullmatch('[A-Za-z0-9_-]+', source['name']) or not re.fullmatch('[0-9a-f]{40}', source['revision']):
            raise SourcePreparationError('Every source requires a name and immutable Git revision')
        if not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git', source['url']):
            raise SourcePreparationError('Only public HTTPS GitHub source URLs are supported')
        path = relative_path(source['path'], 'repos')
        if len(path.parts) < 2:
            raise SourcePreparationError('Source checkout must remain below repos/')
        paths.append(path)
        relative_path(source['system_prompt_file'])
    for index, path in enumerate(paths):
        if any(path == other or path in other.parents or other in path.parents for other in paths[index + 1:]):
            raise SourcePreparationError('Source checkout paths must not overlap')
    if not benchmarks or len({item['name'] for item in benchmarks}) != len(benchmarks):
        raise SourcePreparationError('Benchmark names must be nonempty and unique')
    for item in benchmarks:
        if not re.fullmatch('[a-z0-9_]+', item['name']) or item['source'] not in names:
            raise SourcePreparationError('Invalid benchmark name or source')
        relative_path(item['path'])
        if not re.fullmatch('[0-9a-f]{64}', item['sha256']) or item['rows'] != 80 or item['turns_per_row'] not in (1, 2):
            raise SourcePreparationError('Benchmarks require pinned content and the 80-question protocol')
    return sources, benchmarks


def inspect_checkout(directory, source):
    if directory.is_symlink() or not directory.is_dir():
        raise SourcePreparationError('Source checkout must be an existing directory')
    top = Path(run_git(directory, 'rev-parse', '--show-toplevel').decode().strip())
    if top.resolve() != directory.resolve():
        raise SourcePreparationError('Unexpected Git checkout root')
    head = run_git(directory, 'rev-parse', 'HEAD').decode().strip()
    origin = run_git(directory, 'remote', 'get-url', 'origin').decode().strip()
    if head != source['revision'] or origin.rstrip('/').removesuffix('.git') != source['url'].rstrip('/').removesuffix('.git'):
        raise SourcePreparationError('Existing source commit or origin differs from registry')
    if run_git(directory, 'status', '--porcelain=v1', '--untracked-files=all').strip():
        raise SourcePreparationError('Existing upstream checkout is not clean')
    return head


def ensure_checkout(root, source):
    directory = root / source['path']
    if directory.exists() or directory.is_symlink():
        inspect_checkout(directory, source)
        return directory
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + source['name'] + '-preparing-', dir=directory.parent) as temporary:
        staging = Path(temporary)
        run_git(staging, 'init', '--quiet')
        run_git(staging, 'remote', 'add', 'origin', source['url'])
        run_git(staging, 'fetch', '--quiet', '--no-tags', '--depth=1', 'origin', source['revision'])
        run_git(staging, 'checkout', '--quiet', '--detach', source['revision'])
        inspect_checkout(staging, source)
        if directory.exists() or directory.is_symlink():
            raise SourcePreparationError('Source destination appeared during preparation')
        os.rename(staging, directory)
    return directory


def tracked_manifest(directory):
    names = [name.decode() for name in run_git(directory, 'ls-files', '-z').split(b'\0') if name]
    if not names or len(names) != len(set(names)):
        raise SourcePreparationError('Tracked source inventory is empty or duplicated')
    result = {}
    for name in sorted(names):
        path = directory / relative_path(name)
        value = path.lstat()
        if stat.S_ISLNK(value.st_mode):
            content = os.fsencode(os.readlink(path))
            result[name] = {'kind': 'symlink', 'size_bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        elif stat.S_ISREG(value.st_mode):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            result[name] = {'kind': 'regular_file', 'size_bytes': value.st_size, 'sha256': digest.hexdigest()}
        else:
            raise SourcePreparationError('Unsupported tracked source entry: ' + name)
    identity = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result, identity


def extract_system_prompt(contents):
    matches = []
    for node in ast.walk(ast.parse(contents)):
        if not isinstance(node, ast.Dict):
            continue
        values = {key.value: value for key, value in zip(node.keys, node.values) if isinstance(key, ast.Constant) and isinstance(key.value, str)}
        role = values.get('role')
        content = values.get('content')
        if isinstance(role, ast.Constant) and role.value == 'system' and isinstance(content, ast.Constant) and isinstance(content.value, str):
            matches.append(content.value)
    if len(set(matches)) != 1 or not matches[0].strip():
        raise SourcePreparationError('Evaluator must contain one unique literal system prompt value')
    return matches[0]


def validate_benchmark(contents, expected):
    if hashlib.sha256(contents).hexdigest() != expected['sha256']:
        raise SourcePreparationError('Reference benchmark SHA256 differs: ' + expected['name'])
    rows = [json.loads(line) for line in contents.decode('utf-8').splitlines()]
    if len(rows) != expected['rows']:
        raise SourcePreparationError('Reference benchmark row count differs')
    identifiers = []
    for row in rows:
        identifier = row.get('question_id')
        turns = row.get('turns')
        if not isinstance(identifier, (int, str)) or isinstance(identifier, bool):
            raise SourcePreparationError('Reference question IDs must be integer or string values')
        if not isinstance(turns, list) or len(turns) != expected['turns_per_row'] or any(not isinstance(turn, str) or not turn.strip() for turn in turns):
            raise SourcePreparationError('Reference turns must be nonempty and match the published protocol')
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise SourcePreparationError('Reference question IDs must be unique')
    return identifiers


def prepare_sources(root, registry_path):
    require_cpu_slurm()
    root = Path(root).resolve()
    registry_path = Path(registry_path).resolve()
    registry_contents = registry_path.read_bytes()
    registry = json.loads(registry_contents)
    sources, benchmarks = validate_registry(registry)
    output = root / 'outputs/preparation/sources.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'preparing', 'root': str(root), 'registry': str(registry_path), 'registry_sha256': hashlib.sha256(registry_contents).hexdigest(), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'gpu_execution_performed': False, 'upstream_modified': False, 'scientific_results': False, 'sources': {}, 'benchmarks': {}}
    with (output.parent / 'sources.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SourcePreparationError('Another source preparation process holds the project lock') from None
        atomic_json(output, receipt)
        try:
            for source in sources:
                receipt['active_source'] = source['name']
                atomic_json(output, receipt)
                directory = ensure_checkout(root, source)
                head = inspect_checkout(directory, source)
                tracked, identity = tracked_manifest(directory)
                prompt_path = directory / source['system_prompt_file']
                if prompt_path.is_symlink() or source['system_prompt_file'] not in tracked:
                    raise SourcePreparationError('Evaluator must be a tracked regular source file')
                prompt_contents = prompt_path.read_bytes()
                prompt = extract_system_prompt(prompt_contents)
                if hashlib.sha256(prompt_contents).hexdigest() != tracked[source['system_prompt_file']]['sha256']:
                    raise SourcePreparationError('Evaluator changed during source preparation')
                inspect_checkout(directory, source)
                receipt['sources'][source['name']] = {'url': source['url'], 'revision': head, 'path': str(directory), 'tracked_files': tracked, 'source_identity_sha256': identity, 'system_prompt': {'text': prompt, 'sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'source_file': str(prompt_path), 'source_relative_path': source['system_prompt_file'], 'source_sha256': hashlib.sha256(prompt_contents).hexdigest()}}
                atomic_json(output, receipt)
            receipt.pop('active_source', None)
            destination = root / 'data/reference'
            destination.mkdir(parents=True, exist_ok=True)
            for benchmark in benchmarks:
                source = receipt['sources'][benchmark['source']]
                original = Path(source['path']) / benchmark['path']
                tracked = source['tracked_files'].get(benchmark['path'])
                if original.is_symlink() or not tracked or tracked['kind'] != 'regular_file':
                    raise SourcePreparationError('Benchmark must be a tracked regular source file')
                contents = original.read_bytes()
                identifiers = validate_benchmark(contents, benchmark)
                if tracked['sha256'] != benchmark['sha256']:
                    raise SourcePreparationError('Tracked benchmark identity differs from registry')
                path = destination / (benchmark['name'] + '.jsonl')
                if path.exists() or path.is_symlink():
                    if path.is_symlink() or not path.is_file() or path.read_bytes() != contents:
                        raise SourcePreparationError('Existing reference benchmark differs; refusing overwrite')
                else:
                    atomic_bytes(path, contents)
                receipt['benchmarks'][benchmark['name']] = {'path': str(path), 'sha256': benchmark['sha256'], 'rows': len(identifiers), 'turns_per_row': benchmark['turns_per_row'], 'question_ids': identifiers, 'source': benchmark['source'], 'source_file': str(original), 'source_revision': source['revision']}
                atomic_json(output, receipt)
            for source in sources:
                inspect_checkout(root / source['path'], source)
            receipt.update(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(output, receipt)
        except Exception as error:
            receipt.update(status='failed', error_type=type(error).__name__, failed_at_utc=datetime.now(timezone.utc).isoformat())
            if isinstance(error, SourcePreparationError):
                receipt['error'] = str(error)
            atomic_json(output, receipt)
            raise
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--registry', type=Path, default=Path('configs/sources.json'))
    args = parser.parse_args()
    try:
        receipt = prepare_sources(args.root, args.registry)
    except Exception as error:
        result = {'status': 'failed', 'error_type': type(error).__name__}
        if isinstance(error, SourcePreparationError):
            result['error'] = str(error)
        print(json.dumps(result), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({'status': receipt['status'], 'receipt': str(Path(receipt['root']) / 'outputs/preparation/sources.json'), 'sources': list(receipt['sources']), 'benchmarks': list(receipt['benchmarks'])}), flush=True)


if __name__ == '__main__':
    main()
