import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


V7_SHA256 = 'c13b08f0b62c9e1990423627906f2a98bfe1e666bfcbb6ea31781eef5cf50254'
V7_PATH = Path(__file__).with_name('independent_pilot_qualification_audit_v7_20261112.py')
ZERO_DIAGNOSTIC_SHA256 = '2eeb5b213449072bff78f695afe9bd600f4ddb872144fcd8cb842af7765ca4a4'
ZERO_DIAGNOSTIC_SOURCE_SHA256 = '4de754a2067f5d23274435e7b276e87c85cae5e56295c2a00ee8c96451e29166'


def digest(path):
    import hashlib
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


if digest(V7_PATH) != V7_SHA256:
    raise ValueError('frozen_v7_analyzer_changed')
spec = importlib.util.spec_from_file_location('frozen_independent_pilot_v7', V7_PATH)
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)
v6, v5, v4, v3, v2, v1 = v7.v6, v7.v5, v7.v4, v7.v3, v7.v2, v7.v1
require = v1.require


def validate_zero_fact(fact):
    require(set(fact['visible_trainable_differences']) == set(fact['fp32_master_and_adam_differences']) == v3.NAMES, 'signed_zero_fact_parameter_inventory_mismatch')
    for key in ('all13_named_adam_first_second_nonzero', 'all13_numeric_masters_unchanged', 'all_master_bit_changes_are_signed_zero', 'all_visible_bit_changes_are_signed_zero', 'all_master_casts_bitwise_match_visible'):
        require(fact[key] is True, 'signed_zero_fact_contains_unclassified_difference')
    for name in v3.NAMES:
        visible, master = fact['visible_trainable_differences'][name], fact['fp32_master_and_adam_differences'][name]
        require(visible['numeric_changed'] == master['numeric_changed'] == 0 and visible['all_finite'] is True and master['all_finite'] is True, 'signed_zero_fact_numeric_or_nonfinite_change')
        require(visible['bitwise_changed'] == visible['signed_zero_only_bit_changes'] == master['bitwise_changed'] == master['signed_zero_only_bit_changes'], 'signed_zero_fact_other_bit_change')
        require(master['first_moment_nonzero'] > 0 and master['second_moment_nonzero'] > 0 and master['fp32_master_numeric_delta_l2'] == 0.0 and master['master_cast_visible_bitwise_changes'] == 0, 'signed_zero_fact_moment_or_cast_failure')
    for value in fact['vocabulary'].values():
        require(value['dtype_equal'] is True and value['shape_equal'] is True and value['tensor_equal'] is True and value['bitwise_identity_equal'] is True, 'signed_zero_fact_vocabulary_changed')


class Audit(v7.Audit):
    def zero_facts(self):
        if hasattr(self, 'verified_zero_facts'):
            return self.verified_zero_facts
        study_id = next(iter(self.plans.values()))['study_id']
        path = self.root / 'outputs/pilot-independent-audit' / study_id / 'zero-lr-raw-mechanism.json'
        proof = self.json({'path': str(path), 'sha256': ZERO_DIAGNOSTIC_SHA256})
        require(proof['status'] == 'raw_zero_lr_prefix_mechanism_diagnostic_complete_not_admission' and proof['diagnostic_source_sha256'] == ZERO_DIAGNOSTIC_SOURCE_SHA256 and proof['scientific_training_admitted'] is False, 'signed_zero_fact_report_not_reviewed')
        require(v1.identity(proof['artifact_inventory']) == proof['artifact_inventory_sha256'], 'signed_zero_fact_inventory_identity_mismatch')
        source = Path(__file__).with_name('pilot_zero_lr_raw_diagnostic_20261112.py')
        self.artifact({'path': str(source), 'sha256': ZERO_DIAGNOSTIC_SOURCE_SHA256})
        for name, item in proof['artifact_inventory'].items():
            self.artifact({'path': name, **item})
        facts = {}
        for fact in proof['prefixes']:
            require(fact['policy'] not in facts, 'duplicate_signed_zero_fact_policy')
            validate_zero_fact(fact)
            facts[fact['policy']] = fact
        require(set(facts) == {'original', 'target'}, 'signed_zero_fact_policy_coverage_mismatch')
        self.verified_zero_facts = facts
        return facts

    def raw_checkpoint(self, checkpoint, torch, plan=None, expected_manifest=None):
        manifest = self.json(str(checkpoint / 'manifest.json'))
        if plan is None or manifest.get('next_step') != 2:
            return v6.Audit.raw_checkpoint(self, checkpoint, torch, plan, expected_manifest)
        self.phase = 'exact_reviewed_signed_zero_only_two_step_prefix'
        fact = self.zero_facts()[plan['policy']]
        require(str(checkpoint) == fact['checkpoint'] and digest(checkpoint / 'manifest.json') == fact['manifest_sha256'] and plan['identity_sha256'] == fact['registered_plan_identity_sha256'], 'signed_zero_fact_belongs_to_another_prefix')
        expected = self.root / 'outputs/pilot-qualifications' / plan['study_id'] / plan['policy'] / 'split'
        require(manifest['trajectory'] == {'directory': str(expected), 'qualification': True, 'reference': False, 'alternate_pad': False}, 'signed_zero_policy_is_only_registered_split_prefix')
        require(plan['seed'] == 0 and plan['cohort'] == 'generalist', 'signed_zero_prefix_plan_scope_mismatch')
        scheduler = plan['recipe']['ds_config']['scheduler']
        require(scheduler == {'type': 'WarmupDecayLR', 'params': {'total_num_steps': 1536, 'warmup_max_lr': 5e-5, 'warmup_min_lr': 0, 'warmup_num_steps': 64, 'warmup_type': 'linear'}} and plan['recipe']['ds_config']['optimizer']['params']['lr'] == 0, 'signed_zero_prefix_scheduler_not_registered')
        observations = []
        for rank in range(8):
            directory = expected / 'attempts/prefix'
            initial = self.json(str(directory / ('rank-' + str(rank) + '.initial.json')))
            rows = [row for row in v1.read_lines(self.artifact(str(directory / ('rank-' + str(rank) + '.jsonl')))) if row['event'] == 'committed_update']
            require(initial['state']['model'] == fact['initial_dtype_cast_model_sha256'] and [row['step'] for row in rows] == [0, 1], 'signed_zero_prefix_initial_or_step_scope_mismatch')
            require(all(row['learning_rates_before_step'] == [0.0] and row['local_fp32_master_update_l2'] == 0.0 and row['skipped_steps'] == 0 for row in rows) and rows[-1]['state_identity']['model'] == fact['raw_prefix_model_sha256'], 'signed_zero_prefix_actual_rates_or_hashes_differ')
            observations.append({'rank': rank, 'initial_model_sha256': initial['state']['model'], 'actual_model_sha256_by_step': [row['state_identity']['model'] for row in rows], 'actual_pre_step_learning_rates': [row['learning_rates_before_step'] for row in rows], 'actual_numeric_master_update_l2': [row['local_fp32_master_update_l2'] for row in rows]})
        original = v3.require
        def prefix_criterion(value, code):
            if code != 'named_trainable_has_no_saved_adam_or_master_update':
                return original(value, code)
        v3.require = prefix_criterion
        try:
            result = v6.Audit.raw_checkpoint(self, checkpoint, torch, plan, expected_manifest)
        finally:
            v3.require = original
        named = result['all13_named_adam_and_master_update']
        v7.validate_prefix_parameters(named['parameters'])
        require(result['next_step'] == 2 and all(row['model'] == fact['raw_prefix_model_sha256'] for row in result['ranks']), 'signed_zero_raw_prefix_reconstruction_hash_mismatch')
        for name, item in named['parameters'].items():
            actual = fact['fp32_master_and_adam_differences'][name]
            require(item['elements'] == actual['elements'] and item['first_moment_nonzero'] == actual['first_moment_nonzero'] and item['second_moment_nonzero'] == actual['second_moment_nonzero'], 'signed_zero_raw_named_moments_differ_from_independent_fact')
        named['master_change_expectation'] = 'zero_numeric_change_with_exactly_classified_signed_zero_representation_changes_only'
        named['exact_signed_zero_diagnostic_sha256'] = ZERO_DIAGNOSTIC_SHA256
        named['zero_lr_prefix_observations'] = observations
        named['learning_claim'] = 'All13 raw moments are nonzero. The exact two-step states retain unchanged numeric weights with signed-zero-only bit changes; recorded bitwise hashes remain unchanged in the evidence. All step4 finals still require all13 positive numeric FP32 changes.'
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
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'v8_qualification_audit_requires_cpu_slurm')
    study = args.study / 'manifest.json' if args.study.is_dir() else args.study
    root = Path(json.loads(study.read_bytes())['common']['project_root']).resolve()
    qualification = args.qualification.resolve()
    output = args.output.resolve() if args.output.suffix == '.json' else args.output.resolve() / 'report.json'
    require(output.is_relative_to(root / 'outputs') and not output.exists(), 'choose_fresh_v8_report_under_outputs')
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
    report.update(scientific_training_admitted=False, automatic_admission_emitted=False, artifact_inventory=audit.inventory, artifact_inventory_sha256=v1.identity(audit.inventory), analyzer_v1_sha256=v2.V1_SHA256, analyzer_v2_sha256=v3.V2_SHA256, analyzer_v3_sha256=v4.V3_SHA256, analyzer_v4_sha256=v5.V4_SHA256, analyzer_v5_sha256=v6.V5_SHA256, analyzer_v6_sha256=v7.V6_SHA256, analyzer_v7_sha256=V7_SHA256, analyzer_v8_sha256=digest(Path(__file__)), exact_zero_rate_signed_zero_diagnostic_sha256=ZERO_DIAGNOSTIC_SHA256, verifier_change_scope='Only the exact independently replayed two zero-LR prefix states may differ from initial bytes by classified signed-zero changes. No numeric magnitude difference, vocabulary change, missing moment, unbound file, or positive-step4 failure is permitted.', slurm_job_id=os.environ['SLURM_JOB_ID'], cpu_only=True, wall_seconds=time.monotonic() - start, project_helpers_imported=False, runtime_packages={name: importlib.metadata.version(name) for name in ('torch', 'deepspeed', 'numpy', 'transformers', 'accelerate')})
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
