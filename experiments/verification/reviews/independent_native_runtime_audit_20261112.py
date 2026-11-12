import argparse
import hashlib
import json
import os
from pathlib import Path
import time


PRE_SHA256 = '284f6d0c173b09a61ba605dfd8be10feb9059691fd2c71602566267ecbb3cbca'
OBSERVER_SHA256 = 'f094890cac7962d018361ed03dcbe6e708474b5769d625875111026296d13ba0'
RUNTIME_PINS = {'torch.utils.checkpoint': '7a05799e678085df1b81d512bed679c35053fb6225ea90c96661306f93140f6c', 'deepspeed.runtime.engine': '3bb0e2af63f93c105d51f120bd56886b7d45e4ad33d9cd192e5c687c8673fde8', 'deepspeed.runtime.lr_schedules': '6482b41cc5bf6d8e883020ca8fa5e7a9b1aa44f27e6f93873d12e07e573945d2', 'deepspeed.ops.adam.fused_adam': 'e7f24193f52f04572738bd3f84faf94951404e69a6287148955491d5a9beb017'}


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


class Audit:
    def __init__(self):
        self.inventory = {}
        self.phase = 'initialization'

    def artifact(self, path, expected=None):
        path = Path(path).resolve()
        item = {'sha256': digest(path), 'size_bytes': path.stat().st_size}
        require(expected is None or item['sha256'] == expected, 'native_runtime_artifact_digest_mismatch')
        require(str(path) not in self.inventory or self.inventory[str(path)] == item, 'native_runtime_artifact_changed_during_audit')
        self.inventory[str(path)] = item
        return path

    def read(self, path, expected=None):
        return json.loads(self.artifact(path, expected).read_bytes())

    def observe(self, value):
        require(hashlib.sha256(canonical({key: item for key, item in value.items() if key != 'identity_sha256'})).hexdigest() == value['identity_sha256'], 'native_runtime_observation_identity_mismatch')
        require(value['status'] == 'independent_native_cpu_runtime_observation' and value['observer_source_sha256'] == OBSERVER_SHA256 and value['gpu_work_or_model_loading_performed'] is False, 'native_runtime_observer_scope_mismatch')
        for name, expected in RUNTIME_PINS.items():
            require(value['module_files'][name]['sha256'] == expected, 'native_runtime_implementation_not_reviewed')
        for key in ('source_files', 'upstream_files', 'module_files', 'shared_environment_libraries', 'fused_adam_inventory'):
            for item in value[key].values():
                path = self.artifact(item['path'], item['sha256'])
                require(path.stat().st_size == item['size_bytes'], 'native_runtime_observed_size_mismatch')
        self.artifact(value['python']['executable']['path'], value['python']['executable']['sha256'])
        self.artifact(value['plan']['path'], value['plan']['sha256'])

    def run(self, preparation, pre_path, post_path):
        self.phase = 'native_profile_pre_post_runtime_envelope'
        pre, post = self.read(pre_path, PRE_SHA256), self.read(post_path)
        self.observe(pre)
        self.observe(post)
        require(pre['phase'] == 'pre' and post['phase'] == 'post' and pre['plan'] == post['plan'] and pre['plan_identity_sha256'] == post['plan_identity_sha256'], 'native_runtime_phase_or_profile_binding_mismatch')
        for key in ('python', 'source_files', 'upstream_files', 'module_files', 'shared_environment_libraries', 'package_versions', 'environment'):
            require(pre[key] == post[key], 'native_profile_runtime_or_source_changed_between_observations')
        require(all(not any(row.values()) for key in ('profile_readiness_before', 'profile_readiness_after') for row in pre[key].values()), 'native_pre_observer_was_after_visible_execution_started')
        require(all(all(row.values()) for key in ('profile_readiness_before', 'profile_readiness_after') for row in post[key].values()), 'native_post_observer_does_not_see_executed_profile')
        from datetime import datetime
        before, after = datetime.fromisoformat(pre['observation_completed_at_utc']), datetime.fromisoformat(post['observation_started_at_utc'])
        require(before.tzinfo is not None and after.tzinfo is not None and before <= after, 'native_runtime_observation_time_order_invalid')
        pre_launch, post_launch = pre['launch_observation'], post['launch_observation']
        require(pre_launch['status'] == 'running' and post_launch['status'] == 'complete' and pre_launch['hardware'] == post_launch['hardware'] and pre_launch['plan_identity_sha256'] == post_launch['plan_identity_sha256'] == pre['plan_identity_sha256'], 'native_profile_launch_hardware_interval_binding_mismatch')
        require(datetime.fromisoformat(post_launch['finished_at']) <= after, 'native_post_observation_precedes_profile_completion')
        require(set(pre['fused_adam_inventory']) <= set(post['fused_adam_inventory']) and all(post['fused_adam_inventory'][name] == value for name, value in pre['fused_adam_inventory'].items()), 'native_preexisting_fused_adam_cache_changed')
        require(any(name.endswith('.so') for name in post['fused_adam_inventory']), 'native_fused_adam_binary_not_observed')
        self.phase = 'native_actual_gpu_worker_package_backend_hardware_binding'
        queue = self.read(preparation / 'queue-plan.json')
        profile_descriptor = queue['plans']['profile']
        require(profile_descriptor == {'path': pre['plan']['path'], 'sha256': pre['plan']['sha256']}, 'native_runtime_profile_not_in_registered_preparation')
        cases, packages, hardware = [], None, None
        for label, descriptor in queue['plans'].items():
            plan = self.read(descriptor['path'], descriptor['sha256'])
            require(plan['config']['full_body_v3']['source_sha256'] == queue['source_sha256'], 'native_runtime_queue_source_mismatch')
            directory = Path(plan['run_dir'])
            launch = self.read(directory / 'launch.json')
            require(launch['status'] == 'complete' and launch['plan_identity_sha256'] == plan['plan_identity_sha256'] and launch['hardware']['node'] == 'deep-chungus-11' and len(launch['hardware']['device_bindings']) == 8, 'native_actual_launch_coverage_mismatch')
            actual_hardware = {key: launch['hardware'][key] for key in ('device_bindings', 'gpu_inventory', 'hardware_identity_sha256')}
            require(hardware is None or actual_hardware == hardware, 'native_physical_rank_device_mapping_changed')
            hardware = actual_hardware
            for rank in range(8):
                status = self.read(directory / ('rank-' + str(rank) + '.json'))
                require(status['rank'] == rank and status['world_size'] == 8 and status['plan_identity_sha256'] == plan['plan_identity_sha256'] and status['slurm_job_id'] == launch['slurm_job_id'] and status['status'] == 'bounded_complete', 'native_gpu_worker_identity_mismatch')
                actual_packages = status['package_versions']
                require(packages is None or packages == actual_packages, 'native_actual_gpu_package_versions_differ')
                packages = actual_packages
                require(all(actual_packages[key] == value for key, value in pre['package_versions'].items() if key in actual_packages), 'native_gpu_packages_differ_from_profile_runtime_envelope')
                attempt = directory / 'attempts' / ('attempt-' + str(launch['attempt']))
                observed = self.read(attempt / ('full-body-rank-' + str(rank) + '.json'))
                backend = observed['actual_gpu_backend']
                require(backend['matmul_tf32'] is True and backend['gradient_accumulation_steps'] == 2 and backend['capability'] == [8, 0] and backend['cublas_workspace_config'] == ':4096:8', 'native_actual_backend_contract_mismatch')
                require(observed['source_sha256'] == queue['source_sha256'] and observed['plan_identity_sha256'] == plan['plan_identity_sha256'], 'native_actual_backend_source_plan_mismatch')
                cases.append({'label': label, 'rank': rank, 'backend': backend, 'successful_updates': status['successful_optimizer_steps'], 'runtime_scope': 'pre_post_profile_interval' if label == 'profile' else 'retrospective_source_and_actual_worker_binding_outside_profile_interval'})
        require(len(cases) == 32, 'native_runtime_actual_rank_coverage_incomplete')
        return {'status': 'passed_source_bound_native_runtime_and_profile_interval', 'scope': 'profile_pre_post_shared_source_import_library_envelope_and_retrospective_short_qualification_worker_bindings', 'pre': {'path': str(pre_path), 'sha256': PRE_SHA256}, 'post': {'path': str(post_path), 'sha256': digest(post_path)}, 'profile_started_after_pre_source_order_observation': True, 'actual_gpu_cases': cases, 'actual_package_versions': packages, 'actual_rank_device_bindings': hardware['device_bindings'], 'actual_hardware_inventory': hardware, 'new_fused_adam_artifacts': sorted(set(post['fused_adam_inventory']) - set(pre['fused_adam_inventory'])), 'limitations': ['CPU pre/post observations cover all200 profile updates by source-ordered absence of initial master snapshots/journals atpre and completed launch atpost. They are observational filesystem/source evidence, not an adversarial machine attestation.', 'Short4/direct/2+2 qualification completed beforepre; their runtime is bound retrospectively to actual package/backend/launch/hardware receipts and pinned sources, not claimed inside the profile pre/post interval.', 'CPU observer does not claim independent GPU-node driver/system-library fingerprinting or direct observation of every loaded GPU process library.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--pre', type=Path, required=True)
    parser.add_argument('--post', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'native_runtime_audit_requires_cpu_slurm')
    require(not args.output.exists(), 'native_runtime_audit_output_must_be_fresh')
    audit, start = Audit(), time.monotonic()
    try:
        result = audit.run(args.preparation.resolve(), args.pre.resolve(), args.post.resolve())
        for name, item in audit.inventory.items():
            require(Path(name).stat().st_size == item['size_bytes'] and digest(name) == item['sha256'], 'native_runtime_consumed_artifact_changed')
    except Exception as error:
        result = {'status': 'failed', 'phase': audit.phase, 'error_code': str(error) if type(error) is ValueError else type(error).__name__}
    result.update(scientific_training_admitted=False, automatic_admission_emitted=False, artifact_inventory=audit.inventory, artifact_inventory_sha256=hashlib.sha256(canonical(audit.inventory)).hexdigest(), audit_source_sha256=digest(__file__), slurm_job_id=os.environ['SLURM_JOB_ID'], wall_seconds=time.monotonic() - start, cpu_only=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': result['status'], 'report_sha256': digest(args.output)}))
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
