import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def run(root, study, compiler_bundle, producer_bundle, review, review_sha256):
    sys.path.insert(0, str(compiler_bundle / 'src'))
    from taps_rebuild import pilot_teacher_admission as compiler, pool
    from taps_rebuild.pilot_cache import read, require
    from taps_rebuild.pilot_queue import holder

    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Pilot resource/review handoff requires CPU Slurm')
    require(study.resolve() == root.resolve() / 'outputs/pilot-training/rq1-pilot-24-20261113-v3' and pool.digest(review) == review_sha256 and read(review)['scientific_training_admitted'] is True, 'Pilot handoff requires the exact fresh study and independent actual review')
    require(pool.digest(Path(compiler.__file__)) == '44b8030bba92e71cf30c038481d433ab41b5f38f957807c09ba5d067bf656e12', 'Pilot handoff compiler source differs')
    report = study / 'qualification-report.json'
    require(pool.digest(report) == read(review)['qualification_report_sha256'], 'Pilot actual evidence report differs from independent approval')
    manifest = read(study / 'manifest.json')
    require(read(report)['binding'] == read(review)['binding'] and read(review)['binding']['study_manifest_sha256'] == pool.digest(study / 'manifest.json'), 'Pilot actual review has another study binding')
    producer = read(producer_bundle / 'bundle.json')
    require(Path(manifest['common']['source_directory']).resolve() == producer_bundle.resolve() / 'src/taps_rebuild' and bool(producer['files']) and all(pool.digest(producer_bundle / name) == value for name, value in producer['files'].items()), 'Frozen pilot producer bundle changed or belongs to another study')
    require(not (study / 'admission.json').exists(), 'Pilot handoff refuses an already admitted study')
    deadline = time.monotonic() + 172800
    while True:
        try:
            job = holder(root)
            break
        except (ValueError, pool.PoolError, subprocess.CalledProcessError):
            require(time.monotonic() < deadline, 'Protected pilot allocation did not become available within48hours')
            pool.atomic_json(study / 'resource-wait.json', {'status': 'actual_evidence_approved_waiting_live_protected_high_priority_allocation', 'scientific_admission_written': False, 'gpu_work_started': False, 'protected_holders_mutated': False})
            time.sleep(30)
    require(pool.digest(review) == review_sha256 and pool.digest(report) == read(review)['qualification_report_sha256'], 'Pilot approval or report changed during resource wait')
    compiler.integrate(study, report, review)
    require(holder(root) == job, 'Protected pilot allocation changed during admission integration')
    env = dict(os.environ, PYTHONPATH=str(producer_bundle / 'src'), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    command = ['/usr/bin/python3', '-B', '-S', '-m', 'taps_rebuild.pilot_pipeline', 'train', '--root', str(root), '--study', str(study), '--admission-wait-seconds', '86400']
    subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'study', 'compiler-bundle', 'producer-bundle', 'review'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    run(args.root, args.study, args.compiler_bundle, args.producer_bundle, args.review, args.review_sha256)
