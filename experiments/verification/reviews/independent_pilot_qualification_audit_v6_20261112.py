import argparse
import copy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


V5_SHA256 = '82a4668211870e124a7b434500b038dce75447fb63650d26a8ed10eb60c097c5'
V5_PATH = Path(__file__).with_name('independent_pilot_qualification_audit_v5_20261112.py')
NORMALIZATION_SOURCES = {
    'runtime/engine.py': '3bb0e2af63f93c105d51f120bd56886b7d45e4ad33d9cd192e5c687c8673fde8',
    'ops/adam/fused_adam.py': 'e7f24193f52f04572738bd3f84faf94951404e69a6287148955491d5a9beb017',
}


def digest(path):
    import hashlib
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


if digest(V5_PATH) != V5_SHA256:
    raise ValueError('frozen_v5_analyzer_changed')
spec = importlib.util.spec_from_file_location('frozen_independent_pilot_v5', V5_PATH)
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)
v4, v3, v2, v1 = v5.v4, v5.v3, v5.v2, v5.v1
require = v1.require


def normalized_saved_recipe(recipe):
    require(recipe['optimizer']['type'] == 'AdamW' and recipe['optimizer']['params'].get('adam_w_mode') is True, 'unsupported_deepspeed_recipe_consumption')
    require(recipe['optimizer']['params']['weight_decay'] == 0, 'adamw_consumption_policy_requires_registered_zero_decay')
    value = copy.deepcopy(recipe)
    value['optimizer']['params'].pop('adam_w_mode')
    return value


class Audit(v5.Audit):
    def normalization_sources(self):
        if hasattr(self, 'verified_normalization_sources'):
            return self.verified_normalization_sources
        import deepspeed
        package = Path(deepspeed.__file__).resolve().parent
        environment = (self.root / 'envs/native-eagle3').resolve()
        require(package.is_relative_to(environment), 'normalization_package_outside_declared_runtime')
        files = {}
        for name, expected in NORMALIZATION_SOURCES.items():
            path = package / name
            require(path.resolve() == path and digest(path) == expected, 'normalization_implementation_not_reviewed')
            item = {'sha256': expected, 'size_bytes': path.stat().st_size}
            require(str(path) not in self.inventory or self.inventory[str(path)] == item, 'normalization_source_changed_during_audit')
            self.inventory[str(path)] = item
            files[str(path)] = item
        self.verified_normalization_sources = files
        return files

    def raw_checkpoint(self, checkpoint, torch, plan=None, expected_manifest=None):
        if plan is None:
            return super().raw_checkpoint(checkpoint, torch, plan, expected_manifest)
        self.phase = 'source_bound_deepspeed_saved_config_normalization'
        source_files = self.normalization_sources()
        comparison = dict(plan)
        comparison['recipe'] = dict(plan['recipe'])
        comparison['recipe']['ds_config'] = normalized_saved_recipe(plan['recipe']['ds_config'])
        result = super().raw_checkpoint(checkpoint, torch, comparison, expected_manifest)
        result['saved_ds_config_normalization'] = {'field': 'optimizer.params.adam_w_mode', 'registered_value': True, 'saved_field': 'absent', 'normalization_source_files': source_files, 'scope': 'Pinned DeepSpeed engine consumes this key in-place and explicitly forces AdamW mode for optimizer typeAdamW. All other saved configuration entries remain compared with the unchanged recipe, and raw optimizer groups independently require weight_decay0. Actual algorithm mode is inferred from the pinned constructor path; it is not a serialized field in the raw optimizer state.'}
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--runtime-pre', type=Path)
    parser.add_argument('--runtime-post', type=Path)
    parser.add_argument('--execution-envelope', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'v6_qualification_audit_requires_cpu_slurm')
    study = args.study / 'manifest.json' if args.study.is_dir() else args.study
    root = Path(json.loads(study.read_bytes())['common']['project_root']).resolve()
    qualification = args.qualification.resolve()
    output = args.output.resolve() if args.output.suffix == '.json' else args.output.resolve() / 'report.json'
    require(output.is_relative_to(root / 'outputs') and not output.exists(), 'choose_fresh_v6_report_under_outputs')
    import torch
    import deepspeed
    torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    audit, start, completed = Audit(root), time.monotonic(), {}
    try:
        request = audit.automatic_request(args.study.resolve(), qualification)
        schedules = audit.schedules(request['study_manifest'])
        completed['schedules'] = schedules
        callbacks = audit.checkpoint_policies(qualification)
        completed['checkpoint_and_named_hook_coverage'] = callbacks
        gradients = audit.gradients(request['gradient_cases'], torch)
        completed['gradients'] = gradients
        traces = audit.traces(request['state_comparisons'])
        completed['state_comparisons'] = traces
        suffix = audit.suffix(qualification, torch)
        completed['suffix'] = suffix
        raw = audit.raw_qualifications(qualification, torch)
        completed['raw_states'] = raw
        stock = audit.stock(qualification, torch)
        completed['stock'] = stock
        pre = args.runtime_pre or qualification / 'runtime/pre.json'
        post = args.runtime_post or qualification / 'runtime/post.json'
        interval = args.execution_envelope or qualification / 'runtime/interval-accounting.json'
        runtime = audit.runtime_envelope(pre if pre.is_file() else None, post if post.is_file() else None, interval if interval.is_file() else None)
        completed['runtime'] = runtime
        checks = {'schedules': schedules['status'], 'ce_gradients': gradients['status'], 'outside_mask_gradients': gradients['outside_mask_status'], **traces['statuses'], 'suffix_probe': suffix['status'], 'raw_zero_state_reconstruction': raw['status'], 'all13_named_hooks_and_corrected_policy': callbacks['status'], 'rotary_priming': callbacks['rotary_priming']['status'], 'all13_named_adam_and_fp32_updates': 'passed' if raw['status'] == 'passed' and all(value.get('all13_named_adam_and_master_update', {}).get('status') == 'passed' for value in raw['checkpoints']) else 'partial_evidence', 'stock_export_compatibility': stock['status'], 'runtime_envelope': runtime['status']}
        report = {'status': 'all_checked_evidence_passed_requires_independent_review' if all(value == 'passed' for value in checks.values()) else 'partial_evidence', 'scope': 'independent_primed_full_body_cpu_pilot_qualification_analysis_not_admission', 'checks': checks, **completed}
        audit.finish()
    except Exception as error:
        report = {'status': 'failed', 'phase': audit.phase, 'error_code': str(error) if isinstance(error, v1.AuditFailure) else type(error).__name__, 'completed_analysis': completed}
    report.update(scientific_training_admitted=False, automatic_admission_emitted=False, artifact_inventory=audit.inventory, artifact_inventory_sha256=v1.identity(audit.inventory), analyzer_v1_sha256=v2.V1_SHA256, analyzer_v2_sha256=v3.V2_SHA256, analyzer_v3_sha256=v4.V3_SHA256, analyzer_v4_sha256=v5.V4_SHA256, analyzer_v5_sha256=V5_SHA256, analyzer_v6_sha256=digest(Path(__file__)), verifier_change_scope='Normalize only the source-proven in-place removal of registered AdamW adam_w_modeTrue under the exact reviewed DeepSpeed implementation and zero weight decay; preserve all raw tensor identities, numerical bounds and other recipe checks.', slurm_job_id=os.environ['SLURM_JOB_ID'], cpu_only=True, wall_seconds=time.monotonic() - start, project_helpers_imported=False, runtime_packages={name: importlib.metadata.version(name) for name in ('torch', 'deepspeed', 'numpy', 'transformers', 'accelerate')})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': digest(output), 'output': str(output)}), flush=True)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
