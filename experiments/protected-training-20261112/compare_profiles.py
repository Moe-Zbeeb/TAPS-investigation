import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phases(total, started, finished):
    native = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
    require(total > 0 and 0 < native <= total + 1, 'Inconsistent measured profile intervals')
    return {'whole_profile_wall_seconds': total, 'native_launcher_wall_seconds': native, 'outside_native_launcher_wall_seconds': max(0, total - native), 'scope': 'Native launcher interval includes model initialization and dataset work; it is not steady-state update time'}


def load(root, run):
    directory = root / 'outputs/training-queue' / run
    paths = {name: directory / filename for name, filename in {'status': 'status.json', 'cost': 'profile-cost.json', 'completion': 'profile-completion.json'}.items()}
    status, cost, completion = (read(paths[key]) for key in ('status', 'cost', 'completion'))
    require(status['status'] == 'passed' and cost['status'] == 'complete' and completion['status'] == 'complete' and completion['successful_optimizer_steps_by_rank'] == [200] * 8, 'Comparison requires two completed eight-rank profiles')
    paths['native_plan'] = Path(status['native_plan'])
    paths['native_launch'] = paths['native_plan'].parent / 'launch.json'
    plan, launch = read(paths['native_plan']), read(paths['native_launch'])
    require(digest(paths['native_plan']) == status['native_plan_sha256'] == cost['native_plan_sha256'] == completion['native_plan_sha256'] and digest(paths['completion']) == cost['profile_completion_sha256'], 'Profile evidence identity differs')
    require(launch['status'] == 'complete' and launch['plan_identity_sha256'] == plan['plan_identity_sha256'] and launch['worker_summary']['successful_optimizer_steps_by_rank'] == [200] * 8 and launch['hardware']['hardware_identity_sha256'] == cost['hardware']['hardware_identity_sha256'], 'Native launch or hardware completion differs')
    counts = cost['processed_token_counts_all_dp_ranks']
    require(counts == launch['worker_summary']['processed_training_tokens_all_dp_ranks'] and counts['microbatches'] > 0 and counts['nonpadding_tokens'] > 0, 'Profile token counters differ')
    timing = phases(cost['launcher_wall_seconds'], launch['started_at'], launch['finished_at'])
    ratio = cost['full_recipe_projection']['all_rank_microbatches'] / counts['microbatches']
    identity = {'target': plan['target']['model_identity_sha256'], 'train': plan['train']['sha256'], 'validation': plan['validation']['sha256'], 'native_helper': plan['frozen_files']['inputs/native_training.py'], 'upstream_revision': plan['source_revision'], 'world_size': plan['config']['world_size'], 'max_length': plan['native_recipe']['max_length'], 'seed': plan['native_recipe']['seed']}
    report = {'run_id': run, 'node': cost['hardware']['node'], 'hardware_identity_sha256': cost['hardware']['hardware_identity_sha256'], 'input_identity': identity, 'phases': timing, 'processed_tokens': counts, 'whole_profile_nonpadding_tokens_per_second': counts['nonpadding_tokens'] / timing['whole_profile_wall_seconds'], 'native_launcher_nonpadding_tokens_per_second': counts['nonpadding_tokens'] / timing['native_launcher_wall_seconds'], 'duration_weighted_eight_gpu_hours': cost['allocated_gpu_hours_during_profile'], 'original_rough_linear_full_recipe_hours': cost['full_recipe_projection']['linear_wall_hours'], 'one_preflight_plus_repeated_native_launcher_scenario_hours': (timing['outside_native_launcher_wall_seconds'] + timing['native_launcher_wall_seconds'] * ratio) / 3600, 'telemetry_errors': cost['telemetry_errors'], 'evidence': {key: {'path': str(path), 'sha256': digest(path)} for key, path in paths.items()}}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--primary', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Profile comparison requires CPU Slurm')
    primary, calibration = load(args.root, args.primary), load(args.root, args.calibration)
    require(primary['input_identity'] == calibration['input_identity'], 'Profiles do not share the same source/model/data/recipe identity')
    report = {'status': 'complete', 'scope': 'Same-seed engineering calibration; no quality result or isolated causal hardware comparison', 'profiles': [primary, calibration], 'native_launcher_rate_primary_over_calibration': primary['native_launcher_nonpadding_tokens_per_second'] / calibration['native_launcher_nonpadding_tokens_per_second'], 'limitations': ['Native launcher duration includes model loading and dataset work, not solely optimizer updates.', 'Both full-recipe scenarios remain rough and omit changes in sequence mix, evaluation and checkpoint cost.', 'Shared storage, cold/warm caches and the recorded stale deep11 driver context can affect this single comparison.', 'The protected-node allocation predates calibration; duration-weighted GPU hours are not incremental billing.', 'Keep the original primary profile/full-training chain authoritative; no new training is submitted by this report.'], 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
