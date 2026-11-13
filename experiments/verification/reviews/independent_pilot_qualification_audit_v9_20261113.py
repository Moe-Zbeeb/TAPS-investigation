import argparse
import ast
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time


V8_SHA256 = '3edbf557a8f564cf0c84b1735f37ee23aa2e1fb3d61c597e6d7425ebcda9b65e'
V8_PATH = Path(__file__).with_name('independent_pilot_qualification_audit_v8_20261112.py')
STUDY_ID = 'rq1-pilot-24-20261113-v3'
RECIPE_SHA256 = '2e0e7d0a685d6b29271676cd7ffbcbdc5b42d977df97d8eb398dbf4a73939773'
SOURCE_REVIEW_SHA256 = 'fa1485adffbd5494c196ba08efc204448af2da758511fe285f314ea4ced3162c'
PREPARER_SHA256 = '48db3d503474012a92c83c4b1467f3558f1bc52225975a9755cbb3e589cc7ea9'
QUALIFIER_SHA256 = '1a27103dab4faa14f6ead30dc5e5c1fee457e3d98783c4e86e5b61a2a121833e'
PROBE_SHA256 = '1f3ab82da730bac154d34909101624aa0d912b6bd5d7593ded4bdbb8b770ab2c'
INTERVAL_COLLECTOR_SHA256 = 'ec0da5da75253a26d1c8139d89dc22cccb9573203fff3fec86964dde437a0501'
SOURCES = {
    'native_training.py': '2aad19f7aefddfb70dd02b78d2eaf9709c90834d2308d055a836be83de5fec51',
    'pool.py': 'b00f1930c4eb74ab1312426978487162f6105e5c8b2b478ae0d5894b9898b8c7',
    'pilot_cache.py': 'f9ad06ae2d754c4485bbd22f3e8b98f65ab9706877f40b8961e9f4f1e706ea5e',
    'pilot_gate.py': '81526515203af4a0386680b79f091d4f0166aae170a12a23d1b209d9a3028470',
    'pilot_study.py': '8e9fb70fb92240d9253711cca23c0343d1f5b894c32b87eb12e0d2058872f882',
    'pilot_train.py': 'f700df226d9bd0ea3391868c24e6780efcfde0be3967bda2dba354e7cc699a4a',
    'pilot_queue.py': '0c2ec34b25496f793338abbd3495ea9553f13dde8928c4b9777f5d565506a80f',
    'pilot_pipeline.py': '45a813eb3f7063ac8224665ca8397f5a1efd390e25035b23b4c667436f320cf6',
    'pilot_smoke.py': '026aa0d7473c7adbd4df3c59bad4677847ff5fbec234aa5badf4ee6f23bd7a13',
    'native_gradients.py': '594ee0ee196f9deceecdfe7af7a2bda6b8ee728f2aa9c222748689755c4f4d1a',
}
TEACHER_POLICY = {
    'capture': 'exact32 pristine CPUFP32 inv_freq buffers before DeepSpeed initialization',
    'restore': 'original bytes and dtype on each current CUDA device after engine initialization; verify full buffer identity again after checkpoint restoration',
    'scope': 'new source-bound pilot qualification and scientific study only; old failed run and native scratch recipe unchanged',
}
BUFFER_NAMES = sorted('model.layers.' + str(layer) + '.self_attn.rotary_emb.inv_freq' for layer in range(32))
HELPERS = {'teacher_buffer_snapshot', 'restore_teacher_buffers'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


if digest(V8_PATH) != V8_SHA256:
    raise ValueError('frozen_v8_analyzer_changed')
spec = importlib.util.spec_from_file_location('frozen_independent_pilot_v8', V8_PATH)
v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v8)
v7, v6, v5, v4, v3, v2, v1 = v8.v7, v8.v6, v8.v5, v8.v4, v8.v3, v8.v2, v8.v1
require = v1.require


def helper_ast(path):
    result = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(Path(path).read_bytes()).body if isinstance(node, ast.FunctionDef) and node.name in HELPERS}
    require(set(result) == HELPERS, 'teacher_helper_inventory_mismatch')
    return result


def validate_teacher_policy(value, expected):
    require(value['policy'] == 'exact_pre_deepspeed_fp32_teacher_buffers' and value['buffer_names'] == BUFFER_NAMES, 'teacher_buffer_policy_or_complete_names_mismatch')
    require(value['pristine_buffer_identity_sha256'] == value['restored_buffer_identity_sha256'] == expected, 'teacher_buffer_bytes_differ_from_actual_standalone_equivalent_probe')
    require(value['teacher_parameters_remain_frozen'] is True and value['teacher_remains_in_evaluation_mode'] is True, 'teacher_buffer_policy_does_not_preserve_frozen_eval_state')


class Audit(v8.Audit):
    def __init__(self, root, zero_diagnostic):
        super().__init__(root)
        require(re.fullmatch('[0-9a-f]{64}', zero_diagnostic['sha256']), 'explicit_zero_diagnostic_sha256_required')
        self.zero_diagnostic = zero_diagnostic

    def source_and_teacher_proof(self, study):
        self.phase = 'new_study_exact_teacher_source_and_actual_helper_proof'
        path = study / 'manifest.json' if study.is_dir() else study
        manifest = v1.verified_identity(self.json(str(path)))
        common = manifest['common']
        require(common['study_id'] == STUDY_ID and common['source_sha256'] == SOURCES and common['recipe_sha256'] == RECIPE_SHA256 and common['recipe']['teacher_buffer_policy'] == TEACHER_POLICY, 'new_teacher_study_source_or_recipe_not_reviewed')
        self.study_path, self.study_sha256, self.study_common = path.resolve(), digest(path), common
        source = Path(common['source_directory'])
        bundle = source.parents[1]
        self.artifact({'path': str(bundle / 'configs/rq1-pilot-24-20261113.json'), 'sha256': RECIPE_SHA256})
        self.artifact({'path': str(bundle / 'prepare_teacher_study_20261113.py'), 'sha256': PREPARER_SHA256})
        self.artifact({'path': str(bundle / 'qualify_teacher_buffers_20261113.py'), 'sha256': QUALIFIER_SHA256})
        review = self.json({'path': str(bundle / 'source-review.json'), 'sha256': SOURCE_REVIEW_SHA256})
        require(review['scientific_training_admitted'] is False and review['study_id'] == STUDY_ID and review['source_sha256']['src/taps_rebuild/pilot_train.py'] == SOURCES['pilot_train.py'], 'new_teacher_source_review_scope_differs')
        preparation = self.json(str(bundle / 'preparation.json'))
        require(preparation['status'] == 'prepared_fresh_teacher_policy_study_for_bounded_engineering' and preparation['scientific_training_admitted'] is False and preparation['failed_scientific_run_resumed'] is False and preparation['source_review_sha256'] == SOURCE_REVIEW_SHA256, 'new_teacher_preparation_not_source_bound')
        require(preparation['manifest'] == {'path': str(self.study_path), 'sha256': self.study_sha256} and preparation['all40_complete_selected_logit_identities_match'] is True and preparation['all96_schedule_file_sha256_unchanged'] is True, 'new_teacher_preparation_belongs_to_another_study')
        final_ast = helper_ast(source / 'pilot_train.py')
        proof = self.root / 'outputs/diagnostics/pilot-teacher-gate-20261113-v3'
        completion = self.json({'path': str(proof / 'completion.json'), 'sha256': preparation['probe_completion_sha256']})
        require(completion['status'] == 'completed_eight_rank_engineering_probe' and completion['optimizer_updates'] == 0 and completion['scientific_training_admitted'] is False and completion['scientific_run_resumed'] is False and [item['rank'] for item in completion['ranks']] == list(range(8)), 'actual_teacher_helper_probe_incomplete')
        launch = self.json(str(proof / 'launch.json'))
        require(launch['probe_sha256'] == PROBE_SHA256 and '--restore-preinitialization-buffers' in launch['command'], 'actual_teacher_helper_collector_or_intervention_differs')
        collector = self.root / 'outputs/preparation/pilot-teacher-gate-probe-20261113-v3/pilot_teacher_gate_probe_20261113.py'
        self.artifact({'path': str(collector), 'sha256': PROBE_SHA256})
        old = v1.verified_identity(self.json(str(self.root / 'outputs/pilot-training/rq1-pilot-24-20261112-v2/manifest.json')))
        failed = v1.verified_identity(self.json(old['plans'][0]))
        require((failed['cohort'], failed['policy'], failed['seed'], failed['quota']) == ('code', 'original', 0, 64), 'actual_teacher_probe_failed_condition_differs')
        for key in ('cache_identity_sha256', 'eligibility_identity_sha256', 'vocabulary_identity_sha256', 'hardware', 'scan_runtime', 'quota'):
            require(common[key] == failed[key], 'fresh_teacher_study_changed_sealed_population_or_runtime')
        old_schedules = {name: item['sha256'] for name, item in old['files'].items() if name.startswith('schedules/')}
        new_schedules = {name: item['sha256'] for name, item in manifest['files'].items() if name.startswith('schedules/')}
        require(old_schedules == new_schedules and len(new_schedules) == 96, 'fresh_teacher_study_changed_registered_schedule_or_mask')
        expected, cases = None, []
        require(set(preparation['probe_rank_json_sha256']) == {str(proof / ('rank-' + str(rank) + '.json')) for rank in range(8)}, 'teacher_preparation_rank_inventory_incomplete')
        for item in completion['ranks']:
            rank = item['rank']
            name = str(proof / ('rank-' + str(rank) + '.json'))
            require(preparation['probe_rank_json_sha256'][name] == item['sha256'], 'teacher_preparation_rank_digest_differs')
            value = self.json({'path': name, 'sha256': item['sha256']})
            require(value['rank'] == rank and value['probe_sha256'] == PROBE_SHA256 and value['plan'] == {key: old['plans'][0][key] for key in ('path', 'sha256')} and value['hardware'] == common['hardware'] and value['runtime'] == common['scan_runtime'], 'actual_teacher_helper_probe_binding_differs')
            require(value['optimizer_updates'] == 0 and value['scientific_training_admitted'] is False and value['scientific_run_resumed'] is False and value['teacher_parameter_differences'] == [], 'actual_teacher_helper_probe_is_not_frozen_zero_update_diagnosis')
            before = value['native_before_deepspeed_buffers']
            require(before == value['standalone_scan_buffers'] and sorted(before) == BUFFER_NAMES and all(item['dtype'] == 'torch.float32' and item['shape'] == [64] for item in before.values()), 'actual_teacher_pristine_buffers_not_exact_standalone_fp32_state')
            restored = value['engineering_buffer_restoration']
            producer = self.artifact(restored['producer_source'])
            require(producer == self.root / 'outputs/preparation/pilot-teacher-gate-probe-20261113-v3/src/taps_rebuild/pilot_train.py' and helper_ast(producer) == final_ast, 'actual_gpu_teacher_helpers_differ_from_reviewed_final_source')
            expected = restored['pristine_buffer_identity_sha256'] if expected is None else expected
            validate_teacher_policy(restored, expected)
            require([case['step'] for case in value['cases']] == [0, 63, 127, 206, 207], 'actual_teacher_helper_case_coverage_differs')
            for case in value['cases']:
                context = self.gradient_contexts[(0, 'code', rank, case['step'])]
                require(case['positions'] == context['positions'][0] and case['example_id'] == context['row']['example_id'] and case['index'] == context['row']['index'], 'actual_teacher_helper_cases_differ_from_independently_rebuilt_schedule')
                left, right = [case['predictions'][name] for name in ('standalone_scan', 'native_after_deepspeed')]
                require(case['argmax_disagreements'] == 0 and len(left['covered']) == len(right['covered']) == common['quota'] and all(left['covered']) and all(right['covered']) and not left['uncovered_positions'] and not right['uncovered_positions'] and left['selected_logits_identity'] == right['selected_logits_identity'] and left['argmax'] == right['argmax'], 'actual_teacher_helper_complete_selected_logits_or_gate_differ')
                cases.append({'rank': rank, 'step': case['step'], 'selected_logits_identity': left['selected_logits_identity']})
        self.teacher_identity = expected
        return {'status': 'passed', 'study_id': STUDY_ID, 'source_sha256': SOURCES, 'recipe_sha256': RECIPE_SHA256, 'pristine_teacher_buffer_identity_sha256': expected, 'actual_helper_cases': cases, 'unchanged_schedule_files': 96, 'post_checkpoint_teacher_assertion': 'Bound to exact reviewed f700 worker bytes; successful attempt completion implies this unconditional pre-loop check passed. It is not an independent all-step teacher parameter replay.'}

    def zero_facts(self):
        if hasattr(self, 'verified_zero_facts'):
            return self.verified_zero_facts
        self.phase = 'new_study_explicit_zero_learning_rate_raw_facts'
        path = Path(self.zero_diagnostic['path']).resolve()
        require(path.is_relative_to(self.root / 'outputs/pilot-independent-audit' / STUDY_ID), 'fresh_zero_diagnostic_outside_new_study_namespace')
        proof = self.json(self.zero_diagnostic)
        require(proof['status'] == 'raw_zero_lr_prefix_mechanism_diagnostic_complete_not_admission' and proof['diagnostic_source_sha256'] == v8.ZERO_DIAGNOSTIC_SOURCE_SHA256 and proof['scientific_training_admitted'] is False, 'fresh_zero_diagnostic_scope_or_source_differs')
        require(v1.identity(proof['artifact_inventory']) == proof['artifact_inventory_sha256'] and proof['artifact_inventory'].get(str(self.study_path), {}).get('sha256') == self.study_sha256, 'fresh_zero_diagnostic_not_bound_to_new_manifest')
        self.artifact({'path': str(Path(__file__).with_name('pilot_zero_lr_raw_diagnostic_20261112.py')), 'sha256': v8.ZERO_DIAGNOSTIC_SOURCE_SHA256})
        for name, item in proof['artifact_inventory'].items():
            self.artifact({'path': name, **item})
        facts = {}
        for fact in proof['prefixes']:
            require(fact['policy'] not in facts, 'duplicate_fresh_signed_zero_fact_policy')
            v8.validate_zero_fact(fact)
            plans = [plan for plan in self.plans.values() if plan['seed'] == 0 and plan['cohort'] == 'generalist' and plan['policy'] == fact['policy']]
            require(len(plans) == 1 and fact['registered_plan_identity_sha256'] == plans[0]['identity_sha256'], 'fresh_zero_fact_belongs_to_another_study_or_plan')
            facts[fact['policy']] = fact
        require(set(facts) == {'original', 'target'}, 'fresh_signed_zero_fact_policy_coverage_mismatch')
        self.verified_zero_facts = facts
        return facts

    def raw_checkpoint(self, checkpoint, torch, plan=None, expected_manifest=None):
        result = super().raw_checkpoint(checkpoint, torch, plan, expected_manifest)
        named = result.get('all13_named_adam_and_master_update', {})
        if 'exact_signed_zero_diagnostic_sha256' in named:
            require(named['exact_signed_zero_diagnostic_sha256'] == v8.ZERO_DIAGNOSTIC_SHA256 and result['next_step'] == 2, 'inherited_signed_zero_metadata_scope_differs')
            named['exact_signed_zero_diagnostic_sha256'] = self.zero_diagnostic['sha256']
            named['exact_signed_zero_diagnostic'] = self.zero_diagnostic
        return result

    def teacher_initializations(self, qualification, callbacks):
        self.phase = 'actual_teacher_policy_across_all_new_standard_attempts'
        coverage = []
        for case in callbacks['attempt_ranks']:
            directory = qualification / case['policy'] / case['variant'] / 'attempts' / case['attempt']
            initial = self.json(str(directory / ('rank-' + str(case['rank']) + '.initial.json')))
            complete = self.json(str(directory / ('rank-' + str(case['rank']) + '.complete.json')))
            require(initial['checkpoint_policy'] == complete['checkpoint_policy'] and complete['teacher_gate_engineering_prefix'] is False, 'standard_teacher_attempt_policy_or_prefix_flag_differs')
            validate_teacher_policy(initial['checkpoint_policy']['teacher_buffers'], self.teacher_identity)
            coverage.append({key: case[key] for key in ('policy', 'variant', 'attempt', 'rank')})
        require(len(coverage) == 80, 'new_teacher_standard_attempt_rank_coverage_incomplete')
        return {'status': 'passed', 'attempt_ranks': coverage, 'pristine_teacher_buffer_identity_sha256': self.teacher_identity}

    def teacher_prefix(self, qualification, torch):
        self.phase = 'independent_208_update_teacher_gate_prefix_replay'
        plans = [plan for plan in self.plans.values() if (plan['cohort'], plan['policy'], plan['seed']) == ('code', 'original', 0)]
        require(len(plans) == 1, 'teacher_prefix_failed_condition_plan_not_unique')
        plan = plans[0]
        directory = qualification / 'teacher-gate-prefix'
        complete = self.json(str(directory / 'completion.json'))
        require(complete['status'] == 'completed208_update_engineering_teacher_gate_prefix' and complete['scientific_training_admitted'] is False and complete['plan_identity_sha256'] == plan['identity_sha256'], 'teacher_prefix_completion_scope_differs')
        process = complete['process']
        command = process['command']
        require(process['returncode'] == 0 and '--qualification' in command and '--teacher-gate-prefix' in command and command[command.index('--stop') + 1] == '208' and command[command.index('--directory') + 1] == str(directory) and command[command.index('--attempt') + 1] == 'fresh-prefix' and not any(flag in command for flag in ('--resume', '--reference', '--capture', '--alternate-pad', '--allocation-stop-epoch')), 'teacher_prefix_actual_command_not_exact_fresh_engineering')
        self.artifact({'path': process['log'], 'sha256': process['log_sha256']})
        require(complete['hardware']['node'] == 'deep-chungus-10' and [(value['index'], value['uuid']) for value in complete['hardware']['device_bindings']] == [(row[0], row[1]) for row in plan['hardware']], 'teacher_prefix_physical_rank_device_mapping_differs')
        ancestor_path = self.artifact({'path': str(Path(plan['ancestor']['path']) / 'pytorch_model.bin'), **plan['ancestor']['files']['pytorch_model.bin']})
        ancestor = torch.load(ancestor_path, map_location='cpu', weights_only=True)
        sizes = {name: ancestor[name].numel() for name in v3.NAMES}
        baseline = {name: ancestor[name].to(torch.float16) for name in v3.NAMES}
        baseline.update({name: ancestor[name] for name in ('d2t', 't2d')})
        initial_model = v2.semantic_identity(baseline, torch)
        del baseline, ancestor
        gc.collect()
        reports, coverage = [], []
        for rank in range(8):
            attempt = directory / 'attempts/fresh-prefix'
            initial = self.json(str(attempt / ('rank-' + str(rank) + '.initial.json')))
            final = self.json(str(attempt / ('rank-' + str(rank) + '.complete.json')))
            require(initial['state']['model'] == initial_model and set(initial['trainable_names']) == v3.NAMES and initial['qualification'] is True and initial['reference'] is False and initial['plan_identity_sha256'] == plan['identity_sha256'], 'teacher_prefix_not_fresh_released_initialization')
            require(final['status'] == 'completed_prefix' and final['qualification'] is True and final['reference'] is False and final['teacher_gate_engineering_prefix'] is True and final['plan_identity_sha256'] == plan['identity_sha256'] and final['source_sha256'] == SOURCES, 'teacher_prefix_rank_or_source_scope_differs')
            require(final['start_step'] == 0 and final['stop_step'] == final['successful_updates_this_attempt'] == final['logical_examples_per_rank'] == final['physical_attempts_this_process'] == 208 and final['logical_effective_contexts_per_rank'] == 208 * plan['quota'], 'teacher_prefix_successful_attempted_or_exposure_budget_differs')
            require(initial['checkpoint_policy'] == final['checkpoint_policy'] and final['all_thirteen_trainables_received_finite_nonzero_gradients_each_update'] is True, 'teacher_prefix_policy_changed_or_missing_named_hooks')
            validate_teacher_policy(final['checkpoint_policy']['teacher_buffers'], self.teacher_identity)
            v4.validate_rotary(plan['recipe'], final['checkpoint_policy']['rotary_cache'])
            journal_path = self.artifact({'path': str(attempt / ('rank-' + str(rank) + '.jsonl')), 'sha256': final['journal_sha256']})
            rows = list(v1.read_lines(journal_path))
            require(len(rows) == 416 and [row['event'] for row in rows] == ['begin', 'committed_update'] * 208, 'teacher_prefix_journal_missing_or_extra_attempts')
            schedule = list(v1.read_lines(self.artifact(plan['schedules'][rank])))
            for step in range(208):
                begin, row = rows[2 * step:2 * step + 2]
                scheduled = schedule[step]
                context = self.gradient_contexts[(0, 'code', rank, step)]
                require(begin['step'] == row['step'] == step and begin['example_id'] == scheduled['example_id'] == context['row']['example_id'] and begin['occurrence'] == scheduled['occurrence'] and begin['mask_sha256'] == scheduled['mask_sha256'][0] == v1.identity(context['positions'][0]) and begin['physical_attempt'] == row['physical_attempt'] == 'fresh-prefix', 'teacher_prefix_attempt_or_selected_mask_differs')
                require(row['global_step'] == row['logical_examples_per_rank'] == step + 1 and row['skipped_steps'] == 0 and row['effective_base_contexts'] == plan['quota'] and row['effective_head_contexts'] == [plan['quota']] * 7 and row['dense_slots'] == 2048 and row['real_tokens'] == context['row']['lengths'][0], 'teacher_prefix_committed_update_or_context_counts_differ')
                require(math.isfinite(row['loss']) and len(row['native_head_losses']) == 7 and all(math.isfinite(value) and value >= 0 for value in row['native_head_losses']), 'teacher_prefix_loss_nonfinite_or_invalid')
                expected_loss = sum(0.8 ** head * value for head, value in enumerate(row['native_head_losses'])) * 2048 / plan['quota']
                require(abs(row['loss'] - expected_loss) <= 5e-5, 'teacher_prefix_weighted_normalized_loss_differs')
                v3.validate_callbacks(row['parameter_gradient_coverage'], sizes)
                require(row['state_identity']['global_steps'] == step + 1 and row['state_identity']['skipped_steps'] == 0 and math.isfinite(row['local_fp32_master_update_l2']) and row['local_fp32_master_update_l2'] >= 0, 'teacher_prefix_state_counter_or_numeric_update_invalid')
                iteration = step - 1
                gamma = 0 if iteration < 0 else iteration / 64 if iteration < 64 else max(0.0, (1536 - iteration) / (1536 - 64))
                require(len(row['learning_rates_before_step']) == 1 and abs(row['learning_rates_before_step'][0] - 5e-5 * gamma) <= max(1e-15, 5e-5 * gamma * 1e-12), 'teacher_prefix_actual_pre_step_scheduler_differs')
                if step < 2:
                    require(row['local_fp32_master_update_l2'] == 0.0, 'teacher_prefix_zero_learning_rate_has_numeric_master_change')
            require(final['state_identity'] == rows[-1]['state_identity'] and final['state_identity']['global_steps'] == 208 and final['state_identity']['skipped_steps'] == 0, 'teacher_prefix_final_trace_or_counter_differs')
            reports.append(final)
            coverage.append({'rank': rank, 'committed_updates': 208, 'begin_records': 208, 'effective_contexts': 208 * plan['quota'], 'all13_finite_nonzero_callbacks_every_update': True})
        latest = self.json(str(directory / 'latest.json'))
        expected_checkpoint = (self.root / 'models').resolve() / 'trained/pilot-qualifications' / directory.relative_to(self.root / 'outputs') / 'checkpoints/step-208'
        require(latest['path'] == str(expected_checkpoint) and latest['next_step'] == 208, 'teacher_prefix_final_checkpoint_namespace_or_cursor_differs')
        raw = self.raw_checkpoint(expected_checkpoint, torch, plan, latest['manifest_sha256'])
        require(raw['next_step'] == 208 and raw['all13_named_adam_and_master_update']['status'] == 'passed', 'teacher_prefix_final_raw_named_learning_not_verified')
        for rank, state in enumerate(raw['ranks']):
            expected = reports[rank]['state_identity']
            require(all(state[key] == expected[key] for key in ('model', 'optimizer_scaler', 'scheduler', 'rng', 'global_steps', 'skipped_steps')) and state['micro_steps_from_rank_checkpoint'] == expected['micro_steps'], 'teacher_prefix_raw_final_state_differs_from_complete_journal')
        return {'status': 'passed', 'ranks': coverage, 'raw_checkpoint': raw, 'failed_boundary_exercised': {'rank': 2, 'step': 207, 'base_position': 320}, 'online_gate_scope': 'Successful source-bound exactQ assertions in all208updates on each of8ranks, independently rebuilt masks/counts and raw final state. No independent208target forward rerun or quality result.'}

    def runtime_envelope(self, pre, post, execution_envelope):
        result = super().runtime_envelope(pre, post, execution_envelope)
        require(result['status'] == 'passed' and execution_envelope is not None, 'new_teacher_runtime_interval_incomplete')
        interval = self.json(str(execution_envelope))
        require(interval['collector_source_sha256'] == INTERVAL_COLLECTOR_SHA256 and interval['scientific_training_admitted'] is False, 'new_cpu_normal_priority_interval_collector_not_reviewed')
        self.artifact({'path': str(Path(__file__).with_name('collect_pilot_interval_20261113.py')), 'sha256': INTERVAL_COLLECTOR_SHA256})
        self.artifact({'path': interval['submission_path'], 'sha256': interval['submission_sha256']})
        expected = {'pre_job': 'taps-runtime-pre', 'qualification_job': 'taps-pilot24-qualify', 'post_job': 'taps-runtime-post'}
        rows = [line.split('|') for line in interval['raw_sacct'].splitlines()]
        for key, name in expected.items():
            matches = [row for row in rows if len(row) >= 11 and row[0] == str(interval[key]['job_id'])]
            require(len(matches) == 1 and matches[0][1] == name and matches[0][2:8] == ['mohamadzbib', 'grad-students', 'low-priority', 'normal', 'COMPLETED', '0:0'], 'new_runtime_cpu_accounting_owner_class_or_completion_differs')
        result['cpu_coordinator_scheduling'] = 'All three CPU observer/coordinator jobs are low-priority/normal; actual GPU work remains a separately accounted high-priority/spec-priority step inside the protected full-node allocation.'
        result['interval_collector_sha256'] = INTERVAL_COLLECTOR_SHA256
        return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('study', 'qualification', 'zero-diagnostic', 'runtime-pre', 'runtime-post', 'execution-envelope', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--zero-diagnostic-sha256', required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'v9_qualification_audit_requires_cpu_slurm')
    study = args.study / 'manifest.json' if args.study.is_dir() else args.study
    root = Path(json.loads(study.read_bytes())['common']['project_root']).resolve()
    qualification = args.qualification.resolve()
    output = args.output.resolve()
    require(output.is_relative_to(root / 'outputs') and not output.exists(), 'choose_fresh_v9_report_under_outputs')
    require(qualification == root / 'outputs/pilot-qualifications' / STUDY_ID, 'v9_qualification_namespace_differs')
    import torch
    import deepspeed
    torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    zero = {'path': str(args.zero_diagnostic.resolve()), 'sha256': args.zero_diagnostic_sha256}
    audit, start, completed = Audit(root, zero), time.monotonic(), {}
    try:
        audit.artifact({'path': __file__, 'sha256': digest(__file__)})
        for module, expected in ((v8, V8_SHA256), (v7, v8.V7_SHA256), (v6, v7.V6_SHA256), (v5, v6.V5_SHA256), (v4, v5.V4_SHA256), (v3, v4.V3_SHA256), (v2, v3.V2_SHA256), (v1, v2.V1_SHA256)):
            audit.artifact({'path': module.__file__, 'sha256': expected})
        request = audit.automatic_request(args.study.resolve(), qualification)
        schedules = audit.schedules(request['study_manifest'])
        completed['schedules'] = schedules
        teacher_source = audit.source_and_teacher_proof(args.study.resolve())
        completed['teacher_source_and_actual_helper_proof'] = teacher_source
        callbacks = audit.checkpoint_policies(qualification)
        completed['checkpoint_and_named_hook_coverage'] = callbacks
        teacher_attempts = audit.teacher_initializations(qualification, callbacks)
        completed['teacher_buffer_initializations'] = teacher_attempts
        prefix = audit.teacher_prefix(qualification, torch)
        completed['teacher_gate_prefix208'] = prefix
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
        runtime = audit.runtime_envelope(args.runtime_pre, args.runtime_post, args.execution_envelope)
        completed['runtime'] = runtime
        checks = {'schedules': schedules['status'], 'ce_gradients': gradients['status'], 'outside_mask_gradients': gradients['outside_mask_status'], **traces['statuses'], 'suffix_probe': suffix['status'], 'raw_zero_state_reconstruction': raw['status'], 'all13_named_hooks_and_corrected_policy': callbacks['status'], 'rotary_priming': callbacks['rotary_priming']['status'], 'all13_named_adam_and_fp32_updates': 'passed' if raw['status'] == 'passed' and all(value.get('all13_named_adam_and_master_update', {}).get('status') == 'passed' for value in raw['checkpoints']) else 'partial_evidence', 'stock_export_compatibility': stock['status'], 'runtime_envelope': runtime['status'], 'teacher_buffer_policy': 'passed' if teacher_source['status'] == teacher_attempts['status'] == 'passed' else 'partial_evidence', 'teacher_gate_prefix208': prefix['status']}
        report = {'status': 'all_checked_evidence_passed_requires_independent_review' if all(value == 'passed' for value in checks.values()) else 'partial_evidence', 'scope': 'independent_new_study_pristine_teacher_full_body_cpu_pilot_qualification_analysis_not_admission', 'checks': checks, **completed}
        audit.finish()
    except Exception as error:
        report = {'status': 'failed', 'phase': audit.phase, 'error_code': str(error) if isinstance(error, v1.AuditFailure) else type(error).__name__, 'error_detail': str(error)[:500], 'completed_analysis': completed}
    report.update(scientific_training_admitted=False, automatic_admission_emitted=False, artifact_inventory=audit.inventory, artifact_inventory_sha256=v1.identity(audit.inventory), predecessor_analyzer_v8_sha256=V8_SHA256, analyzer_v9_sha256=digest(__file__), exact_zero_rate_signed_zero_diagnostic=zero, verifier_change_scope='Retain the exact V8 numerical/raw/signed-zero criteria; bind fresh zero-LR facts explicitly to the new study and add source-bound pristine teacher policy, actual helper proof, independently reconstructed208update prefix and normal-priority CPU interval accounting. No historical fact is relabeled as new and no scientific admission is emitted.', slurm_job_id=os.environ['SLURM_JOB_ID'], cpu_only=True, wall_seconds=time.monotonic() - start, project_helpers_imported=False, runtime_packages={name: importlib.metadata.version(name) for name in ('torch', 'deepspeed', 'numpy', 'transformers', 'accelerate')})
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
