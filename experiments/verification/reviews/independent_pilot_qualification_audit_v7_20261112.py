import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


V6_SHA256 = 'bb8c6eef947390535094cea39d009e5e1cd474856eb6ae84db03f86a972024b9'
V6_PATH = Path(__file__).with_name('independent_pilot_qualification_audit_v6_20261112.py')


def digest(path):
    import hashlib
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


if digest(V6_PATH) != V6_SHA256:
    raise ValueError('frozen_v6_analyzer_changed')
spec = importlib.util.spec_from_file_location('frozen_independent_pilot_v6', V6_PATH)
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)
v5, v4, v3, v2, v1 = v6.v5, v6.v4, v6.v3, v6.v2, v6.v1
require = v1.require


def validate_prefix_parameters(parameters):
    require(set(parameters) == v3.NAMES, 'zero_lr_prefix_parameter_inventory_mismatch')
    for item in parameters.values():
        require(item['first_moment_nonzero'] > 0 and item['second_moment_nonzero'] > 0, 'zero_lr_prefix_has_no_saved_named_adam_learning')
        require(item['fp32_master_changed_elements'] == 0 and item['fp32_master_delta_l2_from_fp16_initialized_released_ancestor'] == 0.0, 'zero_lr_prefix_unexpected_parameter_change')


class Audit(v6.Audit):
    def raw_checkpoint(self, checkpoint, torch, plan=None, expected_manifest=None):
        manifest = self.json(str(checkpoint / 'manifest.json'))
        if plan is None or manifest.get('next_step') != 2:
            return super().raw_checkpoint(checkpoint, torch, plan, expected_manifest)
        self.phase = 'source_bound_expected_zero_learning_rate_prefix'
        expected = self.root / 'outputs/pilot-qualifications' / plan['study_id'] / plan['policy'] / 'split'
        require(manifest['trajectory'] == {'directory': str(expected), 'qualification': True, 'reference': False, 'alternate_pad': False}, 'zero_lr_exception_is_only_registered_split_prefix')
        require(plan['seed'] == 0 and plan['cohort'] == 'generalist' and plan['policy'] in ('original', 'target'), 'zero_lr_prefix_plan_scope_mismatch')
        scheduler = plan['recipe']['ds_config']['scheduler']
        require(scheduler == {'type': 'WarmupDecayLR', 'params': {'total_num_steps': 1536, 'warmup_max_lr': 5e-5, 'warmup_min_lr': 0, 'warmup_num_steps': 64, 'warmup_type': 'linear'}} and plan['recipe']['ds_config']['optimizer']['params']['lr'] == 0, 'zero_lr_prefix_not_registered_scheduler')
        prefix = []
        for rank in range(8):
            directory = expected / 'attempts/prefix'
            initial = self.json(str(directory / ('rank-' + str(rank) + '.initial.json')))
            rows = [row for row in v1.read_lines(self.artifact(str(directory / ('rank-' + str(rank) + '.jsonl')))) if row['event'] == 'committed_update']
            require([row['step'] for row in rows] == [0, 1] and all(row['learning_rates_before_step'] == [0.0] and row['local_fp32_master_update_l2'] == 0.0 and row['state_identity']['model'] == initial['state']['model'] and row['skipped_steps'] == 0 for row in rows), 'actual_two_step_prefix_does_not_have_zero_lr_and_unchanged_model')
            prefix.append({'rank': rank, 'actual_pre_step_learning_rates': [row['learning_rates_before_step'] for row in rows], 'actual_master_update_l2': [row['local_fp32_master_update_l2'] for row in rows]})
        original = v3.require
        def prefix_criterion(value, code):
            if code != 'named_trainable_has_no_saved_adam_or_master_update':
                return original(value, code)
        v3.require = prefix_criterion
        try:
            result = super().raw_checkpoint(checkpoint, torch, plan, expected_manifest)
        finally:
            v3.require = original
        require(result['next_step'] == 2, 'zero_lr_raw_prefix_cursor_mismatch')
        named = result['all13_named_adam_and_master_update']
        validate_prefix_parameters(named['parameters'])
        named['master_change_expectation'] = 'all13_exactly_unchanged_after_two_actual_zero_lr_updates'
        named['zero_lr_prefix_observations'] = prefix
        named['learning_claim'] = 'All13 raw Adam moments are nonzero; no parameter update is claimed at this zero-learning-rate prefix. Every step4 final still requires all13 strictly positive FP32 changes.'
        return result

    def diagnostic(self, study, torch):
        manifest = v1.verified_identity(self.json(str(study / 'manifest.json' if study.is_dir() else study)))
        results = []
        for descriptor in manifest['plans']:
            plan = v1.verified_identity(self.json(descriptor))
            if plan['seed'] != 0 or plan['cohort'] != 'generalist':
                continue
            path = (self.root / 'models').resolve() / 'trained/pilot-qualifications' / ('pilot-qualifications/' + plan['study_id'] + '/' + plan['policy'] + '/split/checkpoints/step-2')
            results.append(self.raw_checkpoint(path, torch, plan))
        require(len(results) == 2, 'zero_lr_diagnostic_policy_coverage_incomplete')
        return {'status': 'passed_zero_learning_rate_prefix_mechanism_only_not_admission', 'scope': 'both_registered_two_step_pilot_prefixes_raw_moments_and_exact_zero_master_changes_only', 'raw_prefixes': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--diagnostic-zero-lr-prefix', action='store_true')
    parser.add_argument('--runtime-pre', type=Path)
    parser.add_argument('--runtime-post', type=Path)
    parser.add_argument('--execution-envelope', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'v7_qualification_audit_requires_cpu_slurm')
    study = args.study / 'manifest.json' if args.study.is_dir() else args.study
    root = Path(json.loads(study.read_bytes())['common']['project_root']).resolve()
    qualification = args.qualification.resolve()
    output = args.output.resolve() if args.output.suffix == '.json' else args.output.resolve() / 'report.json'
    require(output.is_relative_to(root / 'outputs') and not output.exists(), 'choose_fresh_v7_report_under_outputs')
    import torch
    import deepspeed
    torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    audit, start, completed = Audit(root), time.monotonic(), {}
    try:
        if args.diagnostic_zero_lr_prefix:
            report = audit.diagnostic(args.study.resolve(), torch)
        else:
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
    report.update(scientific_training_admitted=False, automatic_admission_emitted=False, artifact_inventory=audit.inventory, artifact_inventory_sha256=v1.identity(audit.inventory), analyzer_v1_sha256=v2.V1_SHA256, analyzer_v2_sha256=v3.V2_SHA256, analyzer_v3_sha256=v4.V3_SHA256, analyzer_v4_sha256=v5.V4_SHA256, analyzer_v5_sha256=v6.V5_SHA256, analyzer_v6_sha256=V6_SHA256, analyzer_v7_sha256=digest(Path(__file__)), verifier_change_scope='At the registered split2 prefix, require both actual pre-step learning rates0, raw nonzero moments for all13, and exactly unchanged FP32 masters. Preserve all13 strictly positive master changes at all step4 finals and all prior numerical/source/state checks.', slurm_job_id=os.environ['SLURM_JOB_ID'], cpu_only=True, wall_seconds=time.monotonic() - start, project_helpers_imported=False, runtime_packages={name: importlib.metadata.version(name) for name in ('torch', 'deepspeed', 'numpy', 'transformers', 'accelerate')})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': digest(output)}), flush=True)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
