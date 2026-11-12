import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

from . import corpus, pool


def git_blob(path):
    size = Path(path).stat().st_size
    digest = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def prepare(root, config_path, derived_path, output):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Pilot input preparation requires CPU Slurm')
    root, derived_path = Path(root).resolve(), Path(derived_path).resolve()
    if not derived_path.is_relative_to(root / 'outputs/preparation') or derived_path.exists():
        raise ValueError('Choose a fresh derived source configuration under outputs/preparation')
    config = deepcopy(json.loads(Path(config_path).read_bytes()))
    for source in config['sources']:
        if not re.fullmatch('[A-Za-z0-9_.-]+', source['id']) or not re.fullmatch('[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', source['repo_id']) or not re.fullmatch('[0-9a-f]{40}', source['revision']):
            raise ValueError('Invalid pinned pilot source')
        for descriptor in source['files']:
            relative = Path(descriptor['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid pilot source file')
            if descriptor.get('sha256'):
                continue
            if not re.fullmatch('[0-9a-f]{40}', descriptor['git_blob_sha1']):
                raise ValueError('A small Git-backed source needs its pinned blob identity')
            destination = root / 'data/raw' / source['id'] / source['revision'] / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                temporary = destination.with_name(destination.name + '.' + str(os.getpid()) + '.tmp')
                request = urllib.request.Request('https://huggingface.co/datasets/' + source['repo_id'] + '/resolve/' + source['revision'] + '/' + descriptor['path'], headers={'User-Agent': 'TAPS-pilot-preparation'})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open('xb') as stream:
                    total = 0
                    for data in iter(lambda: response.read(1024 * 1024), b''):
                        total += len(data)
                        if total > descriptor['size_bytes']:
                            raise ValueError('Git-backed source exceeded its pinned size')
                        stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                if total != descriptor['size_bytes'] or git_blob(temporary) != descriptor['git_blob_sha1']:
                    raise ValueError('Downloaded source differs from its immutable Git blob')
                temporary.replace(destination)
            if destination.stat().st_size != descriptor['size_bytes'] or git_blob(destination) != descriptor['git_blob_sha1']:
                raise ValueError('Existing pilot raw source differs from its immutable Git blob')
            descriptor['sha256'] = pool.digest(destination)
    config['preparation_adapter'] = {'path': str(Path(__file__)), 'sha256': pool.digest(__file__), 'corpus_sha256': pool.digest(corpus.__file__), 'source_template_sha256': pool.digest(config_path)}
    derived_path.parent.mkdir(exist_ok=True, parents=True)
    pool.atomic_json(derived_path, config)
    return corpus.prepare(root, derived_path, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--derived-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.config, args.derived_config, args.output), indent=2))


if __name__ == '__main__':
    main()
