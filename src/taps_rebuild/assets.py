import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import stat


DEFAULT_ROOT = Path('/mnt/nfs/home/mohamadzbib/projects/taps-rebuild')
HUB_VERSION = '0.33.4'


class PreparationError(ValueError):
    pass


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_cpu_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise PreparationError('Asset preparation requires a Slurm allocation')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PreparationError('Asset preparation requires CUDA_VISIBLE_DEVICES to be empty')


def validate_registry(registry):
    if registry.get('schema_version') != 1 or registry.get('huggingface_hub_version') != HUB_VERSION:
        raise PreparationError('Unsupported asset registry or Hugging Face Hub version')
    assets = registry.get('assets', [])
    if not assets or len({asset['id'] for asset in assets}) != len(assets):
        raise PreparationError('Asset IDs must be nonempty and unique')
    destinations = []
    for asset in assets:
        if not re.fullmatch('[a-z0-9_]+', asset['id']):
            raise PreparationError('Invalid asset ID')
        path = Path(asset['model_path'])
        if path.is_absolute() or len(path.parts) < 2 or path.parts[0] != 'models' or '..' in path.parts:
            raise PreparationError('Model paths must remain under models/')
        destinations.append(path)
        source = asset['source']
        if not re.fullmatch(r'[^/\s]+/[^/\s]+', source['repo_id']) or not re.fullmatch('[0-9a-f]{40}', source['revision']):
            raise PreparationError('Every source must name a repository and immutable revision')
        names = [item['path'] for item in asset['files']]
        if not names or len(names) != len(set(names)):
            raise PreparationError('Required file names must be nonempty and unique')
        for item in asset['files']:
            name = item['path']
            if not name or name in ('.', '..') or any(c in name for c in '/\\*?[]'):
                raise PreparationError('Required files must be exact flat names')
            if not isinstance(item['size_bytes'], int) or item['size_bytes'] < 0:
                raise PreparationError('Invalid expected file size')
            lfs = item.get('lfs_sha256')
            blob = item.get('git_blob_id')
            if not ((lfs and re.fullmatch('[0-9a-f]{64}', lfs)) or (not lfs and blob and re.fullmatch('[0-9a-f]{40}', blob))):
                raise PreparationError('Every file requires a pinned LFS SHA256 or Git blob ID')
            if item.get('sha256') and not re.fullmatch('[0-9a-f]{64}', item['sha256']):
                raise PreparationError('Invalid expected content SHA256')
        if 'reuse' in asset:
            reuse = asset['reuse']
            if not Path(reuse['path']).is_absolute() or not Path(reuse['receipt_path']).is_absolute():
                raise PreparationError('Legacy reuse paths must be absolute')
            if not re.fullmatch('[0-9a-f]{64}', reuse['receipt_sha256']) or any(not item.get('sha256') for item in asset['files']):
                raise PreparationError('Reused assets require the original receipt and every content SHA256')
    for index, path in enumerate(destinations):
        if any(path == other or path in other.parents or other in path.parents for other in destinations[index + 1:]):
            raise PreparationError('Asset model paths must not overlap')
    return assets


def verify_file(path, expected):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected['size_bytes']:
        raise PreparationError('Required file type or size differs: ' + path.name)
    content = hashlib.sha256()
    blob = None if expected.get('lfs_sha256') else hashlib.sha1(('blob ' + str(before.st_size) + '\0').encode())
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise PreparationError('Required file changed while opening: ' + path.name)
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            content.update(block)
            if blob is not None:
                blob.update(block)
        after = os.fstat(stream.fileno())
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
        raise PreparationError('Required file changed during verification: ' + path.name)
    actual = content.hexdigest()
    identity = actual if blob is None else blob.hexdigest()
    declared = expected.get('lfs_sha256') or expected['git_blob_id']
    if identity != declared or (expected.get('sha256') and actual != expected['sha256']):
        raise PreparationError('Required file content identity differs: ' + path.name)
    return {'size_bytes': after.st_size, 'sha256': actual, 'verification_method': 'lfs_sha256' if blob is None else 'git_blob_id', 'expected_identity': declared, 'git_blob_id': None if blob is None else blob.hexdigest()}


def check_contents(directory, names):
    if not directory.is_dir():
        raise PreparationError('Asset directory is missing')
    extras = {entry.name for entry in directory.iterdir()} - set(names) - {'.cache'}
    if extras:
        raise PreparationError('Unexpected files in asset directory: ' + ', '.join(sorted(extras)))


def verify_reuse_receipt(asset):
    reuse = asset['reuse']
    path = Path(reuse['receipt_path'])
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != reuse['receipt_sha256']:
        raise PreparationError('Original preparation receipt content differs')
    receipt = json.loads(contents)
    model = receipt['models'][reuse['receipt_model_key']]
    if receipt.get('status') != 'complete' or any(model[key] != asset['source'][key] for key in ('repo_id', 'revision')):
        raise PreparationError('Original preparation receipt model identity differs')
    if Path(model['path']).resolve() != Path(reuse['path']).resolve():
        raise PreparationError('Original preparation receipt model path differs')
    if set(model['files']) != {item['path'] for item in asset['files']}:
        raise PreparationError('Original preparation receipt file selection differs')
    for item in asset['files']:
        old = model['files'][item['path']]
        if old['sha256'] != item['sha256'] or old['size_bytes'] != item['size_bytes']:
            raise PreparationError('Original preparation receipt file identity differs')


def prepare_one(root, asset, downloader):
    destination = root / asset['model_path']
    required = [item['path'] for item in asset['files']]
    reused = 'reuse' in asset
    if reused:
        verify_reuse_receipt(asset)
        directory = Path(asset['reuse']['path'])
    else:
        directory = destination
        if directory.is_symlink():
            raise PreparationError('New download destination must not be an individual model symlink')
        directory.mkdir(parents=True, exist_ok=True)
    check_contents(directory, required)
    verified = {}
    missing = []
    for item in asset['files']:
        path = directory / item['path']
        if path.exists() or path.is_symlink():
            verified[item['path']] = verify_file(path, item)
        else:
            missing.append(item['path'])
    if reused and missing:
        raise PreparationError('Required legacy model files are missing')
    if missing:
        downloaded = downloader(repo_id=asset['source']['repo_id'], revision=asset['source']['revision'], local_dir=str(directory), allow_patterns=missing, token=False, max_workers=4)
        if Path(downloaded).resolve() != directory.resolve():
            raise PreparationError('Download destination differs from the registered model path')
        check_contents(directory, required)
        verified = {item['path']: verify_file(directory / item['path'], item) for item in asset['files']}
    if reused:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != directory.resolve():
                raise PreparationError('Legacy model alias destination is already occupied')
        else:
            destination.symlink_to(directory, target_is_directory=True)
    identities = {name: {'size_bytes': value['size_bytes'], 'sha256': value['sha256']} for name, value in sorted(verified.items())}
    model_identity = hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'status': 'verified', 'source': asset['source'], 'official_identity': asset.get('official_identity'), 'path': str(destination), 'resolved_path': str(directory.resolve()), 'reused': reused, 'files': verified, 'total_bytes': sum(value['size_bytes'] for value in verified.values()), 'model_identity_sha256': model_identity}


def prepare_assets(root, registry_path, downloader=None, hub_version=None):
    require_cpu_slurm()
    root = Path(root).resolve()
    registry_path = Path(registry_path).resolve()
    registry_contents = registry_path.read_bytes()
    registry = json.loads(registry_contents)
    assets = validate_registry(registry)
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    if hub_version is None:
        hub_version = importlib.metadata.version('huggingface-hub')
    if hub_version != HUB_VERSION:
        raise PreparationError('Asset preparation requires huggingface-hub==' + HUB_VERSION)
    if downloader is None:
        from huggingface_hub import snapshot_download
        downloader = snapshot_download
    output = root / 'outputs/preparation/assets.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'preparing', 'root': str(root), 'registry': str(registry_path), 'registry_sha256': hashlib.sha256(registry_contents).hexdigest(), 'source_sha256': sha256(Path(__file__)), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'huggingface_hub_version': hub_version, 'anonymous_downloads': True, 'gpu_execution_performed': False, 'scientific_results': False, 'collection_destination': registry.get('collection_destination'), 'models': {}}
    with (output.parent / 'assets.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PreparationError('Another asset preparation process holds the project lock') from None
        atomic_json(output, receipt)
        try:
            for asset in assets:
                receipt['active_asset'] = asset['id']
                atomic_json(output, receipt)
                print(json.dumps({'asset': asset['id'], 'status': 'verifying_or_preparing', 'revision': asset['source']['revision']}), flush=True)
                receipt['models'][asset['id']] = prepare_one(root, asset, downloader)
                atomic_json(output, receipt)
            receipt.pop('active_asset', None)
            receipt.update(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(output, receipt)
        except Exception as error:
            receipt.update(status='failed', error_type=type(error).__name__, failed_at_utc=datetime.now(timezone.utc).isoformat())
            if isinstance(error, PreparationError):
                receipt['error'] = str(error)
            atomic_json(output, receipt)
            raise
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--registry', type=Path, default=Path('configs/assets.json'))
    args = parser.parse_args()
    try:
        receipt = prepare_assets(args.root, args.registry)
    except Exception as error:
        result = {'status': 'failed', 'error_type': type(error).__name__}
        if isinstance(error, PreparationError):
            result['error'] = str(error)
        print(json.dumps(result), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({'status': receipt['status'], 'receipt': str(Path(receipt['root']) / 'outputs/preparation/assets.json'), 'models': list(receipt['models'])}), flush=True)


if __name__ == '__main__':
    main()
