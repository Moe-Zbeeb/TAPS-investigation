import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import ModuleType


def modules():
    if __package__:
        package = __package__
    else:
        package = '_taps_corrected_native'
        if package not in sys.modules:
            module = ModuleType(package)
            module.__path__ = [str(Path(__file__).parent)]
            sys.modules[package] = module
    return tuple(importlib.import_module(package + '.' + name) for name in ('native_training', 'native_gradients'))


native, gradients = modules()
FILES = ('native_full_body.py', 'native_gradients.py', 'native_training.py', 'native_recovery.py', 'pilot_cache.py', 'pool.py')
STAGES = {'corrected-qualification-v3': 4, 'corrected-profile-v3': 200, 'corrected-full-v3': None}


def require(value, message):
    if not value:
        raise native.NativeTrainingError(message)


def sources():
    return {name: native.digest(Path(__file__).with_name(name)) for name in FILES}


def document(root, descriptor, label):
    return native.gate_document(root, descriptor, label)


def gate(root, config):
    policy = config.get('full_body_v3', {})
    stage = config.get('pipeline_stage')
    require(stage in STAGES and config['scope'] == 'native_training' and config['world_size'] == 8 and config['initialize_from_released_draft'] is False, 'Corrected native stage must use eight-rank scratch training')
    require(config['budget'] == {'optimizer_steps': STAGES[stage]} and config.get('checkpoint_roundtrip') is False and config.get('recovery') is not None, 'Corrected native budget/recovery contract differs')
    require(policy.get('schema_version') == 3 and policy.get('source_sha256') == sources() and policy.get('rotary_capacity') == 2054, 'Corrected native source or rotary policy differs')
    mode = policy.get('checkpoint_mode')
    require(mode == 'non_reentrant' or stage == 'corrected-qualification-v3' and mode == 'off_reference', 'Direct checkpoint reference is qualification-only')
    ceiling = policy.get('maximum_attempted_optimizer_steps')
    require((stage == 'corrected-full-v3' and ceiling is None) or type(ceiling) is int and STAGES[stage] <= ceiling <= STAGES[stage] * 8, 'Engineering work requires a finite attempted-step ceiling')
    prerequisites = config['research_gate']
    require(prerequisites.get('policy') == 'pipeline-v2', 'Corrected native work requires the reviewed inference/data prerequisites')
    report = document(root, prerequisites['pipeline_certificate'], 'pipeline prerequisite')
    require(report['native_training_sha256'] == native.digest(native.__file__), 'Legacy prerequisite source differs')
    verifier = importlib.import_module(native.__package__ + '.admission_v2')
    require(all(config.get(key) == value for key, value in verifier.TRAINING_CONTRACT.items()), 'Corrected native target/runtime/recipe differs from the reviewed contract')
    prerequisite = verifier.verify_certificate(root, report, prerequisites['independent_review'])
    parent = document(root, policy['data_parent_plan'], 'data parent plan')
    require(parent['train']['sha256'] == policy['full_data_sha256']['train'] and parent['validation']['sha256'] == policy['full_data_sha256']['validation'], 'Full data identities differ from the frozen data parent')
    cache_path = Path(policy['vocabulary_cache']['path']).resolve()
    require(cache_path == Path(parent['workdir']) / 'cache.pt' and native.digest(cache_path) == policy['vocabulary_cache']['sha256'] == parent['frozen_files']['work/cache.pt'], 'Frozen training-only vocabulary cache differs')
    if stage != 'corrected-full-v3':
        return {'status': 'corrected_bounded_engineering_only', 'scientific_training_authorized': False, 'prior_inference_data_prerequisite': prerequisite['certificate_identity_sha256'], 'source_sha256': sources()}
    approval = document(root, policy['scientific_admission'], 'corrected full-body admission')
    require(approval.get('scientific_training_admitted') is True and approval.get('source_sha256') == sources() and approval.get('full_data_sha256') == policy['full_data_sha256'] and approval.get('vocabulary_cache_sha256') == policy['vocabulary_cache']['sha256'], 'Corrected full-body scientific approval is absent or mismatched')
    for label in ('qualification', 'profile'):
        evidence = document(root, approval[label], 'corrected ' + label)
        require(evidence.get('status') == 'passed' and evidence.get('source_sha256') == sources(), 'Corrected ' + label + ' did not pass for these exact sources')
    review = document(root, approval['independent_review'], 'corrected full-body independent review')
    require(review.get('scientific_training_admitted') is True and review.get('source_sha256') == sources() and review.get('qualification') == approval['qualification'] and review.get('profile') == approval['profile'] and review.get('full_data_sha256') == policy['full_data_sha256'] and review.get('vocabulary_cache_sha256') == policy['vocabulary_cache']['sha256'], 'Corrected evidence lacks an exact independent review')
    return {'status': 'corrected_full_body_admitted', 'scientific_training_authorized': True, 'source_sha256': sources()}


class AuditRouter:
    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def run(self, command, *args, **options):
        if isinstance(command, list) and len(command) >= 3 and Path(str(command[2])).name == 'native_training.py':
            require(len(command) == 6 and command[1] == '-B' and command[3:5] == ['audit', '--plan'], 'Unexpected legacy native audit invocation')
            plan = Path(command[5]).resolve()
            require(Path(command[2]).resolve() == plan.parent / 'inputs/native_training.py', 'Audit entry differs from the frozen native plan')
            command = [*command[:2], str(plan.parent / 'inputs/native_full_body.py'), *command[3:]]
        return self.delegate.run(command, *args, **options)


def install():
    native.research_gate = gate
    native.subprocess = AuditRouter(subprocess)


def prepare(root, config_path, train, validation, directory):
    install()
    plan = native.prepare(root, config_path, train, validation, directory)
    directory = Path(plan['run_dir'])
    policy = plan['config']['full_body_v3']
    for name in ('native_full_body.py', 'native_gradients.py', 'pilot_cache.py'):
        shutil.copyfile(Path(__file__).with_name(name), directory / 'inputs' / name)
    shutil.copyfile(policy['vocabulary_cache']['path'], Path(plan['workdir']) / 'cache.pt')
    if plan['config']['pipeline_stage'] != 'corrected-qualification-v3':
        require(plan['train']['sha256'] == policy['full_data_sha256']['train'] and plan['validation']['sha256'] == policy['full_data_sha256']['validation'], 'Corrected profile/full run must use the complete registered data')
    plan['frozen_files'] = {str(path.relative_to(directory)): native.digest(path) for subdir in ('inputs', 'work') for path in (directory / subdir).iterdir() if path.is_file()}
    require(all(plan['frozen_files']['inputs/' + name] == value for name, value in sources().items()), 'Frozen corrected implementation differs')
    plan['command'][plan['command'].index(str(directory / 'inputs/native_training.py'))] = str(directory / 'inputs/native_full_body.py')
    plan['compatibility_changes'] += ['Explicit nonreentrant nested-output checkpoint policy', 'CPU FP32 rotary priming to2054 before conversion; legacy FP16 extension recipe is retired', 'Frozen target forced to evaluation mode; all13 trainables observed across native GAS2 windows']
    plan['native_recipe']['corrected_full_body_v3'] = policy
    plan.pop('plan_identity_sha256')
    import hashlib
    plan['plan_identity_sha256'] = hashlib.sha256(native.canonical(plan)).hexdigest()
    native.atomic_json(directory / 'plan.json', plan)
    native.atomic_json(Path(plan['trained_models']['resolved_path']) / 'native-plan.json', {'plan_identity_sha256': plan['plan_identity_sha256'], 'plan_path': str(directory / 'plan.json')})
    return plan


def install_observers(plan, torch, rank):
    observation = {'source_sha256': sources(), 'plan_identity_sha256': plan['plan_identity_sha256'], 'rank': rank, 'policy': gradients.install_checkpoint_policy(plan['config']['full_body_v3']['checkpoint_mode'])}
    old_model_adapter, old_step_adapter = native.install_model_adapter, native.install_step_adapter

    def model_adapter(module, current, tensor_library):
        original = old_model_adapter(module, current, tensor_library)
        adapted = module.Model

        class Model(adapted):
            def scandata(self, *args, **options):
                value = super().scandata(*args, **options)
                observation['rotary_cache'] = gradients.prime_rotary_cache(self, torch)
                self.target_model.eval()
                require(all(not value.requires_grad for value in self.target_model.parameters()) and not self.embed_tokens.weight.requires_grad, 'Target/embedding must remain frozen')
                self._taps_coverage = gradients.ParameterCoverage(self, torch)
                self.target_model.register_forward_pre_hook(lambda module, args: require(not module.training, 'Frozen online teacher left evaluation mode'))
                return value

            def train(self, mode=True):
                result = super().train(mode)
                self.target_model.eval()
                return result

        module.Model = Model
        return original

    def step_adapter(deepspeed, current, tensor_library, current_rank, legacy_observation):
        original_initialize = old_step_adapter(deepspeed, current, tensor_library, current_rank, legacy_observation)
        adapted_initialize = deepspeed.initialize

        def initialize(*args, **options):
            result = adapted_initialize(*args, **options)
            engine = result[0]
            observation['engine'] = engine
            coverage = engine.module._taps_coverage
            require(engine.gradient_accumulation_steps() == 2, 'Corrected native recipe requires unchanged GAS2')
            old_step, old_forward = engine.step, engine.forward
            attempt = os.environ.get('TAPS_NATIVE_ATTEMPT_ID', '1')
            output = Path(plan['run_dir']) / 'attempts' / ('attempt-' + attempt)
            output.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            initial = [value.detach().clone() for value in engine.optimizer.single_partition_of_fp32_groups]
            torch.save({'masters': [value.cpu() for value in initial], 'param_slice_mappings': engine.optimizer._param_slice_mappings}, output / ('initial-masters-rank-' + str(rank) + '.pt'))
            observation['initial_master_snapshot_sha256'] = native.digest(output / ('initial-masters-rank-' + str(rank) + '.pt'))
            microbatches = []

            def forward(*arguments, **kwargs):
                if torch.is_grad_enabled():
                    microbatches.append({key: native.tensor_state_identity(kwargs[key], torch) for key in ('input_ids', 'attention_mask', 'loss_mask')})
                return old_forward(*arguments, **kwargs)

            def step(*arguments, **kwargs):
                before, skipped = int(engine.global_steps), int(engine.skipped_steps)
                boundary = engine.is_gradient_accumulation_boundary()
                evidence = None
                if boundary:
                    raw = engine.optimizer.averaged_gradients
                    evidence = {'scaled_gradient_partition_sha256': native.tensor_state_identity(raw, torch), 'learning_rates_before_step': [float(group['lr']) for group in engine.optimizer.param_groups], 'loss_scale_before_step': float(engine.optimizer.loss_scale)}
                    if plan['config']['full_body_v3'].get('capture_gradient_partitions'):
                        torch.save(raw, output / ('gradients-rank-' + str(rank) + '-global-' + str(before) + '.pt'))
                value = old_step(*arguments, **kwargs)
                if int(engine.global_steps) > before:
                    require(boundary and len(microbatches) == 2, 'Native optimizer window did not contain exactly two observed microbatches')
                    require(set(coverage.values) == coverage.expected, 'Corrected native backward omitted an intended trainable')
                    update_skipped = int(engine.skipped_steps) > skipped
                    if not update_skipped:
                        require(all(item['finite'] and item['nonzero'] for item in coverage.values.values()), 'Applied native update has nonfinite or entirely zero named gradients')
                    masters = engine.optimizer.single_partition_of_fp32_groups
                    evidence.update(global_step=int(engine.global_steps), skipped_steps=int(engine.skipped_steps), successful_updates=int(engine.global_steps) - int(engine.skipped_steps), update_skipped=update_skipped, loss_scale_after_step=float(engine.optimizer.loss_scale), parameter_gradient_coverage=coverage.values, actual_microbatches=microbatches, state_identity=legacy_observation['recovery'].state_identity(), rng_identity=legacy_observation['recovery'].rng_identity(), local_fp32_master_change_l2_from_initial=float(sum((left.double() - right.double()).square().sum() for left, right in zip(initial, masters)).sqrt()), loop_wall_seconds=time.monotonic() - started)
                    with (output / ('full-body-rank-' + str(rank) + '.jsonl')).open('ab') as stream:
                        stream.write(native.canonical(evidence) + b'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    coverage.begin()
                    microbatches.clear()
                    ceiling = plan['config']['full_body_v3']['maximum_attempted_optimizer_steps']
                    require(ceiling is None or int(engine.global_steps) < ceiling or int(engine.global_steps) - int(engine.skipped_steps) >= plan['config']['budget']['optimizer_steps'], 'Corrected engineering attempted-step ceiling reached before the requested successful budget')
                return value

            engine.step, engine.forward = step, forward
            return result

        deepspeed.initialize = initialize
        return original_initialize

    native.install_model_adapter, native.install_step_adapter = model_adapter, step_adapter
    return observation


def worker(path):
    install()
    plan = native.load_plan(path)
    import torch
    rank = int(os.environ['RANK'])
    observation = install_observers(plan, torch, rank)
    try:
        native.worker(path)
        observation['status'] = 'completed_corrected_native_execution_requires_independent_evidence'
    except BaseException as error:
        observation.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        observation.pop('engine', None)
        observation.update(cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_max_reserved_bytes=torch.cuda.max_memory_reserved(), scientific_training_admitted=plan['config']['pipeline_stage'] == 'corrected-full-v3')
        attempt = os.environ.get('TAPS_NATIVE_ATTEMPT_ID', '1')
        native.atomic_json(Path(plan['run_dir']) / 'attempts' / ('attempt-' + attempt) / ('full-body-rank-' + str(rank) + '.json'), observation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'audit', 'launch', 'resume', 'worker'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--train', type=Path)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    install()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.root, args.config, args.train, args.validation, args.run_dir), indent=2))
    elif args.action == 'audit':
        print(json.dumps(native.audit(args.plan), indent=2))
    elif args.action == 'worker':
        worker(args.plan)
    else:
        native.launch(args.plan, resume=args.action == 'resume')


if __name__ == '__main__':
    main()
