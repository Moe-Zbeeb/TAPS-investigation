import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from taps_rebuild import admission, native_training as native
from taps_rebuild.pool import allocation, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    hardware = allocation(8)
    config = admission.read(args.config)
    directory = args.root / 'outputs/training-queue' / config['run_id']
    status = admission.read(directory / 'status.json')
    admission.require(status['status'] == 'passed', 'Profile preparation must pass before measurement')
    plan = native.load_plan(status['native_plan'])
    command = [sys.executable, '-B', '-m', 'taps_rebuild.training_queue', 'profile', '--root', str(args.root), '--config', str(args.config)]
    started = time.monotonic()
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    samples, errors = [], []
    try:
        with (directory / 'profile-gpu-samples.jsonl').open('x') as stream:
            while True:
                sample = {'elapsed_seconds': time.monotonic() - started, 'at_utc': datetime.now(timezone.utc).isoformat()}
                query = subprocess.run(['/usr/bin/nvidia-smi', '--query-gpu=uuid,utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
                if query.returncode:
                    errors.append({'elapsed_seconds': sample['elapsed_seconds'], 'returncode': query.returncode})
                else:
                    sample['gpus'] = [{'uuid': row[0].strip(), 'utilization_percent': float(row[1]), 'used_memory_mib': float(row[2]), 'total_memory_mib': float(row[3]), 'power_watts': float(row[4])} for row in csv.reader(io.StringIO(query.stdout)) if row]
                    admission.require({row['uuid'] for row in sample['gpus']} == {row['uuid'] for row in hardware['gpu_inventory']}, 'GPU telemetry escaped the owned node')
                    stream.write(json.dumps(sample, allow_nan=False) + '\n')
                    stream.flush()
                    samples.append(sample)
                try:
                    code = process.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    pass
        wall = time.monotonic() - started
        admission.require(code == 0, 'Measured native profile failed')
        launch = admission.read(Path(status['native_plan']).parent / 'launch.json')
        counts = launch['worker_summary']['processed_training_tokens_all_dp_ranks']
        audit = admission.read(Path(status['native_plan']).parent / 'tokenization-audit.json')
        examples = audit['splits']['train']['rows_after_official_filter']
        projected_microbatches = ((examples + 7) // 8) * 8 * plan['native_recipe']['epochs']
        ratio = projected_microbatches / counts['microbatches']
        report = {'status': 'complete', 'scope': 'Measured end-to-end native 200-update profile and sampled whole-device telemetry; no steady-state or exact framework peak-memory claim', 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'hardware': hardware, 'native_plan_sha256': admission.digest(status['native_plan']), 'launcher_wall_seconds': wall, 'allocated_gpu_hours_during_profile': wall * 8 / 3600, 'processed_token_counts_all_dp_ranks': counts, 'counts_include_distributed_sampler_padding': True, 'input_assistant_mask_tokens_per_launcher_second': counts['input_assistant_mask_tokens'] / wall, 'nonpadding_tokens_per_launcher_second': counts['nonpadding_tokens'] / wall, 'sample_count': len(samples), 'telemetry_errors': errors, 'sampled_device_memory_includes_external_or_stale_driver_contexts': True, 'full_recipe_projection': {'epochs': plan['native_recipe']['epochs'], 'train_examples_after_filter': examples, 'all_rank_microbatches': projected_microbatches, 'linear_wall_hours': wall * ratio / 3600, 'linear_gpu_hours': wall * ratio * 8 / 3600, 'scope': 'Rough projection from all profile startup/loading/audit time and its observed sequence-length mix; epoch evaluation/checkpoint costs and length drift can change actual cost'}, 'profile_completion_sha256': admission.digest(directory / 'profile-completion.json'), 'samples_sha256': admission.digest(directory / 'profile-gpu-samples.jsonl'), 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
        atomic_json(directory / 'profile-cost.json', report)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
