import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time


def read_file(path):
    total = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            total += len(block)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Evidence prefetch requires CPU-only Slurm execution')
    report = json.loads(args.report.read_bytes())
    if report.get('status') != 'passed' or not report.get('evidence_files'):
        raise ValueError('Evidence prefetch requires an existing passed report')
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as workers:
        sizes = list(workers.map(read_file, report['evidence_files']))
    print(json.dumps({'status': 'read_only_evidence_prefetch_complete_not_validation_or_admission', 'files': len(sizes), 'bytes': sum(sizes), 'wall_seconds': time.monotonic() - started}))


if __name__ == '__main__':
    main()
