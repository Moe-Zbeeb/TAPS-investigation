import copy
import builtins
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import signal
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from taps_rebuild import native_training as native


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(native.canonical(value) + b'\n')


def conversation(identifier, prompt='Question', answer='Answer'):
    return {'id': identifier, 'conversations': [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': answer}]}


class Loss:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def float(self):
        return self

    def item(self):
        return self.value


class Engine:
    def __init__(self, root, skipped=()):
        self.root = root
        self.global_steps = 0
        self.skipped_steps = 0
        self.skipped = set(skipped)
        self.backward_calls = 0
        self.saved = []
        self.optimizer = types.SimpleNamespace(param_groups=[{'lr': 0.0}])

    def forward(self, **kwargs):
        return 'forward'

    def step(self):
        self.global_steps += 1
        self.skipped_steps += int(self.global_steps in self.skipped)

    def backward(self, loss):
        self.backward_calls += 1

    def get_global_grad_norm(self):
        return 1.5

    def save_16bit_model(self, path, exclude_frozen_parameters):
        self.saved.append((path, exclude_frozen_parameters))
        path = Path(path)
        path.mkdir(parents=True)
        (path / 'pytorch_model.bin').write_bytes(b'checkpoint')
        return True


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / 'project'
        self.root.mkdir()
        self.work = self.root / 'work'
        self.work.mkdir()
        write_json(self.work / 'config.json', {'fixture': True})
        self.config = json.loads((Path(__file__).resolve().parents[1] / 'configs/native-eagle3-training.json').read_bytes())

    def fixture_plan(self):
        pool_source = Path(__file__).resolve().parents[1] / 'src/taps_rebuild/pool.py'
        destination = self.root / 'src/taps_rebuild/pool.py'
        destination.parent.mkdir(parents=True)
        destination.write_bytes(pool_source.read_bytes())
        upstream = self.root / 'repos/EAGLE/eagle/traineagle3'
        upstream.mkdir(parents=True)
        for name in native.UPSTREAM:
            (upstream / name).write_text('fixture ' + name)
        model_config = {'architectures': ['LlamaForCausalLM'], 'hidden_size': 4, 'vocab_size': 8}
        write_json(upstream / 'config.json', model_config)
        hashes = {name: native.digest(upstream / name) for name in native.UPSTREAM}
        assets = {'status': 'complete', 'models': {}}
        for identifier in (self.config['target_id'], self.config['draft_id']):
            directory = self.root / 'models' / identifier
            directory.mkdir(parents=True)
            write_json(directory / 'config.json', model_config)
            (directory / 'pytorch_model.bin').write_bytes(b'weights')
            files = {name: {'size_bytes': (directory / name).stat().st_size, 'sha256': native.digest(directory / name)} for name in ['config.json', 'pytorch_model.bin']}
            assets['models'][identifier] = {'path': str(directory), 'status': 'verified', 'files': files, 'source': {'repo_id': identifier, 'revision': 'a' * 40}, 'model_identity_sha256': hashlib.sha256(native.canonical(files)).hexdigest()}
        write_json(self.root / 'outputs/preparation/assets.json', assets)
        write_json(self.root / 'outputs/preparation/sources.json', {'status': 'complete', 'sources': {'EAGLE': {'revision': native.REVISION, 'path': str(self.root / 'repos/EAGLE')}}})
        config = self.root / 'config.json'
        write_json(config, self.config)
        train, validation = self.root / 'train.jsonl', self.root / 'dev.jsonl'
        train.write_bytes(b''.join(native.canonical(native.native_row(native.engineering_row('train', index), canonical_messages=True)) + b'\n' for index in range(8)))
        validation.write_bytes(b''.join(native.canonical(native.native_row(native.engineering_row('dev', index), canonical_messages=True)) + b'\n' for index in range(8)))
        run = self.root / 'outputs/native-training/run'
        def git(arguments, **kwargs):
            return native.REVISION + '\n' if 'rev-parse' in arguments else ''
        return upstream, hashes, config, train, validation, run, git

    def test_converter_preserves_text_regime_and_official_roles(self):
        source, output = self.root / 'canonical.jsonl', self.root / 'native.jsonl'
        row = {'example_id': 'a', 'split': 'train', 'response_regime': 'target_regenerated_all_assistant_turns', 'messages': [{'role': 'system', 'content': 'native system'}, {'role': 'user', 'content': '  Unicode λ\nquestion  '}, {'role': 'assistant', 'content': ' answer\n'}]}
        source.write_bytes(native.canonical(row) + b'\n')
        receipt = native.convert(source, output, split='train', prompt='native system')
        converted = json.loads(output.read_bytes())
        self.assertEqual(converted['conversations'][0]['value'], row['messages'][1]['content'])
        self.assertEqual(converted['conversations'][1]['from'], 'gpt')
        self.assertIn('target_regenerated_all_assistant_turns', receipt['input_response_regimes'])
        self.assertFalse(receipt['regeneration_performed_by_converter'])
        with self.assertRaisesRegex(native.NativeTrainingError, 'overwritten'):
            native.convert(source, output, split='train', prompt='native system')

    def test_converter_rejects_tensor_masks_custom_system_and_incomplete_pairs(self):
        with self.assertRaisesRegex(native.NativeTrainingError, 'never prototype'):
            list(native.records(self.root / 'features.pt'))
        rows = [
            {'id': 1, 'messages': [{'role': 'system', 'content': 'custom'}, {'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]},
            {'id': 1, 'messages': [{'role': 'user', 'content': 'q'}]},
            {'id': 1, 'messages': [{'role': 'user', 'content': '<|eot_id|>'}, {'role': 'assistant', 'content': 'a'}]},
        ]
        for row in rows:
            with self.subTest(row=row), self.assertRaises(native.NativeTrainingError):
                native.native_row(row, canonical_messages=True, prompt='official')

    def test_failed_conversion_never_commits_partial_dataset(self):
        source, output = self.root / 'canonical.jsonl', self.root / 'native.jsonl'
        row = {'id': 'duplicate', 'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]}
        source.write_bytes((native.canonical(row) + b'\n') * 2)
        with self.assertRaisesRegex(native.NativeTrainingError, 'Duplicate'):
            native.convert(source, output, split='train')
        self.assertFalse(output.exists())
        self.assertFalse(list(output.parent.glob('*.tmp')))

    def test_config_view_exposes_only_two_actual_upstream_attributes(self):
        config = {'bs': 1, 'num_epochs': 40, 'num_workers': 2, 'max_len': 2048, 'config_path': 'config.json', 'gradient_checkpoint': True}
        view = native.training_view(config)
        self.assertEqual(vars(view), {'gradient_checkpointing': True, 'max_len': 2048})
        with self.assertRaises(native.NativeTrainingError):
            native.training_view({**config, 'invented': True})
        observed = []
        class Model:
            def __init__(self, model_config, ds_config, training_config, **kwargs):
                observed.append(training_config.gradient_checkpointing)
                observed.append(training_config.max_len)
            def scandata(self, datapath, tokenizerpath):
                return 'native unchanged'
        module = types.SimpleNamespace(Model=Model)
        original = native.install_model_adapter(module, {'workdir': str(self.work), 'draft': None}, None)
        self.assertIs(original, Model)
        model = module.Model(None, None, config)
        with patch.object(native, 'validate_vocabulary'):
            model.d2t, model.t2d, model.vocab_size, model.draft_vocab_size = None, None, 8, 4
            self.assertEqual(model.scandata('train', 'target'), 'native unchanged')
        self.assertEqual(observed, [True, 2048])

    def test_prepare_pins_exact_bytes_command_and_detects_frozen_tampering(self):
        upstream, hashes, config, train, validation, run, git = self.fixture_plan()
        before = {name: (upstream / name).read_bytes() for name in hashes}
        with patch.object(native, 'UPSTREAM', hashes), patch.object(native, 'require_slurm'), patch.object(native.subprocess, 'check_output', side_effect=git):
            plan = native.prepare(self.root, config, train, validation, run)
            self.assertEqual(plan['command'][0], str(self.root / 'envs/native-eagle3/bin/python'))
            self.assertIn('--nproc_per_node=8', plan['command'])
            self.assertEqual(native.load_plan(run / 'plan.json')['plan_identity_sha256'], plan['plan_identity_sha256'])
            for name in hashes:
                self.assertEqual((run / 'work' / name).read_bytes(), before[name])
                self.assertEqual((upstream / name).read_bytes(), before[name])
            (run / 'inputs/train.jsonl').write_text('changed')
            with self.assertRaisesRegex(native.NativeTrainingError, 'Frozen native input'):
                native.load_plan(run / 'plan.json')

    def test_prepare_rejects_prompt_leakage_before_creating_run(self):
        upstream, hashes, config, train, validation, run, git = self.fixture_plan()
        question = next(native.records(train))['conversations'][0]['value']
        validation.write_bytes(native.canonical(conversation('b', question, 'Different answer')) + b'\n')
        with patch.object(native, 'UPSTREAM', hashes), patch.object(native, 'require_slurm'), patch.object(native.subprocess, 'check_output', side_effect=git):
            with self.assertRaisesRegex(native.NativeTrainingError, 'overlap'):
                native.prepare(self.root, config, train, validation, run)
        self.assertFalse(run.exists())

    def test_prepare_rejects_changed_upstream_or_model_content(self):
        upstream, hashes, config, train, validation, run, git = self.fixture_plan()
        (upstream / 'main.py').write_text('changed')
        with patch.object(native, 'UPSTREAM', hashes), patch.object(native, 'require_slurm'), patch.object(native.subprocess, 'check_output', side_effect=git):
            with self.assertRaisesRegex(native.NativeTrainingError, 'source bytes'):
                native.prepare(self.root, config, train, validation, run)
        assets = json.loads((self.root / 'outputs/preparation/assets.json').read_bytes())
        target = assets['models'][self.config['target_id']]
        (Path(target['path']) / 'pytorch_model.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(native.NativeTrainingError, 'model file changed'):
            native.model_record(assets, self.config['target_id'], verify_content=True)

    def test_successful_step_budget_excludes_overflow_and_saves_before_stop(self):
        engine = Engine(self.root, skipped=[1])
        module = types.SimpleNamespace(initialize=lambda: (engine, object(), None, None))
        barriers = []
        torch = types.SimpleNamespace(distributed=types.SimpleNamespace(barrier=lambda: barriers.append(True)), is_grad_enabled=lambda: True)
        plan = {'run_dir': str(self.root), 'workdir': str(self.work), 'trained_models': {'resolved_path': str(self.root / 'models/trained/run')}, 'config': {'budget': {'optimizer_steps': 1}}}
        native.install_step_adapter(module, plan, torch, rank=0)
        module.initialize()
        engine.backward(Loss(2.0))
        engine.step()
        self.assertEqual(engine.saved, [])
        engine.backward(Loss(1.5))
        with self.assertRaisesRegex(native.BudgetComplete, 'successful native optimizer step 1'):
            engine.step()
        saved = Path(engine.saved[0][0])
        self.assertTrue((saved / 'pytorch_model.bin').exists())
        self.assertEqual((saved / 'config.json').read_bytes(), (self.work / 'config.json').read_bytes())
        self.assertEqual(barriers, [True])
        rows = [json.loads(line) for line in (self.root / 'steps-rank-0.jsonl').read_text().splitlines()]
        self.assertEqual([row['optimizer_step'] for row in rows], [0, 1])
        self.assertEqual(rows[-1]['global_step'], 2)
        self.assertTrue(rows[0]['update_skipped'])
        self.assertEqual(engine.backward_calls, 2)

    def test_nonfinite_native_loss_never_reaches_backward(self):
        engine = Engine(self.root)
        module = types.SimpleNamespace(initialize=lambda: (engine, None, None, None))
        plan = {'run_dir': str(self.root), 'config': {'budget': {'optimizer_steps': None}}}
        native.install_step_adapter(module, plan, None, rank=0)
        module.initialize()
        with self.assertRaisesRegex(native.NativeTrainingError, 'nonfinite'):
            engine.backward(Loss(math.nan))
        self.assertEqual(engine.backward_calls, 0)

    def test_budget_and_scope_do_not_imply_unsupported_matched_tokens(self):
        native.validate_config(self.config)
        for changes in [{'budget': {'optimizer_steps': 2, 'loss_tokens': 10}}, {'budget': {'optimizer_steps': None}}, {'scope': 'native_training'}]:
            with self.subTest(changes=changes), self.assertRaises(native.NativeTrainingError):
                native.validate_config({**self.config, **changes})
        full = {**self.config, 'scope': 'native_training', 'initialize_from_released_draft': False, 'budget': {'optimizer_steps': None}, 'checkpoint_roundtrip': False}
        with self.assertRaisesRegex(native.NativeTrainingError, 'research_gate'):
            native.validate_config(full)

    def test_allocation_tres_fallback_infers_only_verified_full_node_inventory(self):
        (self.root / 'inputs').mkdir()
        (self.root / 'inputs/pool.py').write_bytes((Path(__file__).resolve().parents[1] / 'src/taps_rebuild/pool.py').read_bytes())
        plan = {'run_dir': str(self.root), 'project_root': str(self.root), 'config': {'world_size': 8}}
        job = 'JobId=123 JobState=RUNNING NumNodes=1 NodeList=deep-chungus-4 Account=grad-students UserId=user(' + str(os.getuid()) + ') Partition=high-priority QOS=spec-priority TRES=cpu=64,gres/gpu=8'
        node = 'NodeName=deep-chungus-4 Gres=gpu:a100:8(S:0-1)'
        inventory = '\n'.join(str(index) + ',GPU-' + str(index) + ',A100,80000,580.65' for index in range(8))
        environment = {key: value for key, value in os.environ.items() if key != 'CUDA_VISIBLE_DEVICES'}
        environment.update(SLURM_JOB_ID='123', SLURMD_NODENAME='deep-chungus-4')
        with patch.dict(os.environ, environment, clear=True), patch.object(native.subprocess, 'check_output', side_effect=[job, node, inventory]):
            result = native.allocation(plan)
        self.assertEqual(result['gpu_count'], 8)
        self.assertEqual([binding['token'] for binding in result['device_bindings']], list(map(str, range(8))))
        for bad_job, bad_node in [(job, node.replace(':8(', ':9(')), (job.replace('RUNNING', 'PENDING'), node)]:
            with patch.dict(os.environ, environment, clear=True), patch.object(native.subprocess, 'check_output', side_effect=[bad_job, bad_node, inventory]):
                with self.assertRaises(RuntimeError):
                    native.allocation(plan)


    def test_fixture_is_own_disjoint_synthetic_32_by_8_data(self):
        destination = self.root / 'data/engineering/native'
        result = native.fixture(self.root, destination)
        self.assertEqual(result['splits']['train']['rows'], 32)
        self.assertEqual(result['splits']['dev']['rows'], 8)
        train = native.dataset_inventory(destination / 'train.jsonl')
        dev = native.dataset_inventory(destination / 'dev.jsonl')
        self.assertFalse(set(train['ids']) & set(dev['ids']))
        self.assertFalse(set(train['prompt_identities']) & set(dev['prompt_identities']))
        self.assertTrue(result['not_official_benchmark_or_training_data'])

    def checkpoint_case(self, mode='restore'):
        class Scalar:
            def __init__(self, value, dtype='float32', trainable=True):
                self.value, self.dtype, self.requires_grad = value, dtype, trainable
                self.device = 'cpu'
            def is_floating_point(self):
                return True
            def numel(self):
                return 1
            def is_contiguous(self):
                return True
            def view(self, *args):
                return self
            def __getitem__(self, index):
                return self
            def item(self):
                return self.value
            def fill_(self, value):
                self.value = value
        model, moment, master, frozen = Scalar(.25, 'float16'), Scalar(.125), Scalar(.3), Scalar(.75, trainable=False)
        base = types.SimpleNamespace(param_groups=[{'params': [master]}], state={master: {'exp_avg': moment}})
        optimizer = types.SimpleNamespace(optimizer=base, single_partition_of_fp32_groups=[master], state_dict=lambda: {'moments': [moment], 'master': [master]}, param_groups=[{'lr': 1e-8}])
        engine = types.SimpleNamespace(optimizer=optimizer, module=types.SimpleNamespace(named_parameters=lambda: [('target.frozen', frozen), ('trainable.weight', model)]), global_steps=2, skipped_steps=0)
        calls, saved = [], {}
        def save(*args, **kwargs):
            calls.append(('save', args, kwargs))
            saved.update(model=model.value, moment=moment.value)
        def load(*args, **kwargs):
            calls.append(('load', args, kwargs))
            self.assertNotEqual(model.value, saved['model'])
            self.assertNotEqual(moment.value, saved['moment'])
            if mode in ('restore', 'model_only'):
                model.value = saved['model']
            if mode in ('restore', 'optimizer_only'):
                moment.value = saved['moment']
            return 'checkpoint', {'scope': 'engineering_optimizer_roundtrip'}
        engine.save_checkpoint, engine.load_checkpoint = save, load
        distributed = types.SimpleNamespace(get_world_size=lambda: 1, all_gather_object=lambda values, value: values.__setitem__(slice(None), [value]), barrier=lambda: calls.append(('barrier',)))
        torch = types.SimpleNamespace(float32='float32', is_tensor=lambda value: isinstance(value, Scalar), no_grad=nullcontext, distributed=distributed)
        def identity(value, ignored):
            def values(item):
                if isinstance(item, Scalar):
                    return [item.dtype, item.value]
                if isinstance(item, dict):
                    return {key: values(entry) for key, entry in item.items()}
                if isinstance(item, list):
                    return [values(entry) for entry in item]
                return item
            return hashlib.sha256(native.canonical(values(value))).hexdigest()
        return engine, torch, calls, identity, frozen

    def test_checkpoint_roundtrip_requires_changed_then_exact_restored_model_and_optimizer(self):
        engine, torch, calls, identity, frozen = self.checkpoint_case()
        with patch.object(native, 'tensor_state_identity', side_effect=identity):
            result = native.checkpoint_roundtrip(engine, self.root / 'state', torch)
        self.assertEqual(result['status'], 'passed')
        for prefix in ('optimizer_state', 'trainable_model'):
            self.assertEqual(result[prefix + '_sha256_before'], result[prefix + '_sha256_after'])
            self.assertNotEqual(result[prefix + '_sha256_before'], result[prefix + '_sha256_perturbed'])
        operations = [call for call in calls if call[0] != 'barrier']
        self.assertTrue(operations[0][2]['exclude_frozen_parameters'])
        self.assertTrue(operations[1][2]['load_optimizer_states'])
        self.assertFalse(operations[1][2]['load_module_strict'])
        self.assertEqual(sum(call[0] == 'barrier' for call in calls), 3)
        self.assertEqual(result['perturbation']['model_parameter'], 'trainable.weight')
        self.assertEqual(frozen.value, .75)

    def test_checkpoint_roundtrip_rejects_noop_and_partial_recovery(self):
        for mode in ('noop', 'model_only', 'optimizer_only'):
            with self.subTest(mode=mode):
                engine, torch, calls, identity, frozen = self.checkpoint_case(mode)
                with patch.object(native, 'tensor_state_identity', side_effect=identity):
                    with self.assertRaisesRegex(native.NativeTrainingError, 'recovery differed'):
                        native.checkpoint_roundtrip(engine, self.root / 'state', torch)
                self.assertEqual(frozen.value, .75)

    def test_checkpoint_recovery_rejects_unowned_moments_and_rank_name_disagreement(self):
        engine, torch, calls, identity, frozen = self.checkpoint_case()
        engine.optimizer.single_partition_of_fp32_groups = []
        with self.assertRaisesRegex(native.NativeTrainingError, 'owned local FP32'):
            native.recovery_targets(engine, torch)
        torch.distributed.get_world_size = lambda: 2
        torch.distributed.all_gather_object = lambda values, value: values.__setitem__(slice(None), [value, 'other.weight'])
        with self.assertRaisesRegex(native.NativeTrainingError, 'different replicated'):
            native.recovery_targets(engine, torch)

    def test_loss_scaler_identity_hashes_dtype_exactly_and_rejects_unknown_objects(self):
        class Dtype:
            def __init__(self, name):
                self.name = name
            def __str__(self):
                return self.name
        scaler_type = type('DynamicLossScaler', (), {'__module__': 'deepspeed.runtime.fp16.loss_scaler'})
        scaler = scaler_type()
        scaler.dtype = Dtype('torch.float16')
        scaler.cur_scale = 16384
        scaler.cur_iter = 2
        torch = types.SimpleNamespace(is_tensor=lambda value: False, dtype=Dtype)
        first = native.tensor_state_identity({'loss_scaler': scaler}, torch)
        self.assertEqual(first, native.tensor_state_identity({'loss_scaler': scaler}, torch))
        scaler.dtype = Dtype('torch.bfloat16')
        self.assertNotEqual(first, native.tensor_state_identity({'loss_scaler': scaler}, torch))
        with self.assertRaises(TypeError):
            native.tensor_state_identity({'unsupported': object()}, torch)

    def test_zero_fragment_identity_supports_only_exact_proven_two_integer_fields(self):
        fragment_type = type('fragment_address', (), {'__module__': 'deepspeed.utils.tensor_fragment'})
        fragment = fragment_type()
        fragment.numel, fragment.start = 10, 0
        torch = types.SimpleNamespace(is_tensor=lambda value: False, dtype=type('Dtype', (), {}))
        before = native.tensor_state_identity({'param_slice_mappings': [{'weight': fragment}]}, torch)
        fragment.start = 1
        self.assertNotEqual(before, native.tensor_state_identity({'param_slice_mappings': [{'weight': fragment}]}, torch))
        fragment.extra = 3
        with self.assertRaisesRegex(native.NativeTrainingError, 'fragment-address'):
            native.tensor_state_identity(fragment, torch)
        del fragment.extra
        fragment.start = True
        with self.assertRaisesRegex(native.NativeTrainingError, 'fragment-address'):
            native.tensor_state_identity(fragment, torch)

    def test_overflow_infinite_gradient_can_recover_without_false_success(self):
        engine = Engine(self.root, skipped=[1])
        engine.get_global_grad_norm = lambda: math.inf
        module = types.SimpleNamespace(initialize=lambda: (engine, None, None, None))
        plan = {'run_dir': str(self.root), 'config': {'budget': {'optimizer_steps': None}}}
        native.install_step_adapter(module, plan, None, rank=0)
        module.initialize()
        engine.backward(Loss(1.0))
        engine.step()
        row = json.loads((self.root / 'steps-rank-0.jsonl').read_bytes())
        self.assertTrue(row['update_skipped'])
        self.assertTrue(row['gradient_norm_nonfinite'])
        self.assertIsNone(row['global_grad_norm'])

    def test_native_vocabulary_shortfall_fails_before_training(self):
        torch = types.SimpleNamespace(long='long', bool='bool')
        d2t = types.SimpleNamespace(shape=(3,), dtype='long')
        t2d = types.SimpleNamespace(shape=(8,), dtype='bool')
        with self.assertRaisesRegex(native.NativeTrainingError, 'fewer than'):
            native.validate_vocabulary(torch, d2t, t2d, 8, 4)


    def test_trained_storage_accepts_top_level_models_alias_and_rejects_children(self):
        storage = self.root.parent / 'model-storage'
        storage.mkdir()
        (self.root / 'models').symlink_to(storage, target_is_directory=True)
        paths = native.trained_paths(self.root, 'native-run')
        self.assertEqual(paths['resolved_path'], str(storage / 'trained/native-eagle3/native-run'))
        (storage / 'trained').symlink_to(self.root / 'work', target_is_directory=True)
        with self.assertRaisesRegex(native.NativeTrainingError, 'child symlink'):
            native.trained_paths(self.root, 'another-run')

    def test_training_token_counters_exclude_no_grad_validation(self):
        counts = []
        engine = Engine(self.root)
        module = types.SimpleNamespace(initialize=lambda: (engine, None, None, None))
        enabled = [True]
        torch = types.SimpleNamespace(is_grad_enabled=lambda: enabled[0])
        plan = {'run_dir': str(self.root), 'config': {'budget': {'optimizer_steps': None}}}
        observation = {}
        native.install_step_adapter(module, plan, torch, rank=0, observation=observation)
        module.initialize()
        tensor = types.SimpleNamespace(numel=lambda: 12, sum=lambda: Loss(8))
        mask = types.SimpleNamespace(sum=lambda: Loss(5))
        engine.forward(input_ids=tensor, attention_mask=tensor, loss_mask=mask)
        enabled[0] = False
        engine.forward(input_ids=tensor, attention_mask=tensor, loss_mask=mask)
        self.assertEqual(observation['token_counts'], {'microbatches': 1, 'input_slots': 12, 'nonpadding_tokens': 8, 'input_assistant_mask_tokens': 5})


    def test_native_audit_executes_frozen_preprocessor_and_rejects_empty_supervision(self):
        upstream, hashes, config, train, validation, run, git = self.fixture_plan()
        source = upstream / 'main.py'
        source.write_text('def build_dataset_rank(tokenizer, datapath):\n    return torch.audit_examples\n')
        hashes['main.py'] = native.digest(source)
        class Binary:
            def __or__(self, other):
                return self
            def all(self):
                return True
        class Tensor:
            shape = (1, 3)
            ndim = 2
            def __init__(self, count):
                self.count = count
            def __eq__(self, other):
                return Binary()
            def sum(self):
                return Loss(self.count)
        torch = types.ModuleType('torch')
        mask = Tensor(2)
        torch.audit_examples = [{'input_ids': Tensor(3), 'loss_mask': mask}]
        datasets = types.ModuleType('datasets')
        datasets.load_dataset = lambda *args, **kwargs: None
        transformers = types.ModuleType('transformers')
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
        with patch.object(native, 'UPSTREAM', hashes), patch.object(native, 'require_slurm'), patch.object(native.subprocess, 'check_output', side_effect=git):
            native.prepare(self.root, config, train, validation, run)
        with patch.object(native, 'UPSTREAM', hashes), patch.object(native, 'require_slurm'), patch.dict(sys.modules, {'torch': torch, 'datasets': datasets, 'transformers': transformers}):
            output = native.audit(run / 'plan.json')
            self.assertEqual(output['splits']['train']['token_slots'], 3)
            self.assertEqual(output['splits']['validation']['supervised_mask_tokens'], 2)
            mask.count = 0
            with self.assertRaisesRegex(native.NativeTrainingError, 'no supervised tokens'):
                native.audit(run / 'plan.json')

    def test_native_launcher_interrupt_cleans_only_its_child_process_group(self):
        child = types.SimpleNamespace(pid=12345, wait=lambda *args, **kwargs: None, poll=lambda: 0)
        calls = []
        waits = [KeyboardInterrupt(), None]
        def wait(*args, **kwargs):
            value = waits.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value
        child.wait = wait
        def kill(group, number):
            calls.append((group, number))
            if len(calls) > 2:
                raise ProcessLookupError()
        with patch.object(native.subprocess, 'Popen', return_value=child) as creation, patch.object(native.os, 'killpg', side_effect=kill), patch.object(native.signal, 'signal', return_value=signal.SIG_DFL):
            with self.assertRaises(KeyboardInterrupt):
                native.run_training(['native-worker'], str(self.root), {}, 9)
        self.assertTrue(creation.call_args.kwargs['start_new_session'])
        self.assertEqual(creation.call_args.kwargs['pass_fds'], (9,))
        self.assertIn((12345, signal.SIGTERM), calls)
        self.assertTrue(all(group == 12345 for group, number in calls))

    def test_cuda_compiler_guard_sets_only_verified_project_cache(self):
        with patch.object(Path, 'is_file', return_value=True), patch.object(native.os, 'access', return_value=True), patch.object(native.subprocess, 'check_output', return_value='Cuda compilation tools, release 12.4, V12.4.131'):
            environment = native.cuda_environment(self.root, 'deep-chungus-10', {'cuda_home': '/usr/local/cuda-12.4'})
        self.assertEqual(environment['CUDA_HOME'], '/usr/local/cuda-12.4')
        self.assertEqual(environment['TORCH_CUDA_ARCH_LIST'], '8.0')
        self.assertTrue(Path(environment['TORCH_EXTENSIONS_DIR']).is_relative_to(self.root / 'cache'))
        with patch.object(Path, 'is_file', return_value=True), patch.object(native.os, 'access', return_value=True), patch.object(native.subprocess, 'check_output', return_value='release 12.8,'):
            with self.assertRaisesRegex(native.NativeTrainingError, 'compiler version differs'):
                native.cuda_environment(self.root, 'node', {})


    def test_jit_environment_selects_installed_native_ninja_before_inherited_path(self):
        physical = self.root / 'physical-environment'
        binary = physical / 'bin/ninja'
        binary.parent.mkdir(parents=True)
        binary.write_text('fixture')
        binary.chmod(0o755)
        (physical / 'pyvenv.cfg').write_text('fixture')
        logical = self.root / 'environment'
        logical.symlink_to(physical, target_is_directory=True)
        original = {'PATH': '/old/environment/bin:/usr/bin', 'CUDA_HOME': '/usr/local/cuda-12.4', 'MAX_JOBS': '4'}
        with patch.object(native.subprocess, 'check_output', return_value='1.11.1.git.kitware.jobserver-1\n') as command:
            environment, proof = native.jit_environment(logical / 'bin/python', original)
        self.assertEqual(environment['PATH'].split(os.pathsep)[:2], [str(physical / 'bin'), '/usr/local/cuda-12.4/bin'])
        self.assertEqual(environment['MAX_JOBS'], '4')
        self.assertEqual(original['PATH'], '/old/environment/bin:/usr/bin')
        self.assertEqual(proof['ninja_executable'], str(binary))
        self.assertEqual(command.call_args.args[0], [str(binary), '--version'])
        self.assertEqual(command.call_args.kwargs['env'], environment)
        self.assertEqual(command.call_args.kwargs['timeout'], 30)
        binary.unlink()
        with self.assertRaisesRegex(native.NativeTrainingError, 'installed Ninja'):
            native.jit_environment(logical / 'bin/python', original)


    def test_real_spawn_executes_exact_parent_setup_once_and_unpickles_collator(self):
        upstream, hashes, config, train, validation, run, git = self.fixture_plan()
        marker, output = self.root / 'setup-count.txt', self.root / 'child-output.txt'
        source = upstream / 'main.py'
        source.write_text('import multiprocessing\nfrom pathlib import Path\nPath(' + repr(str(marker)) + ').open("a").write("setup\\n")\nclass DataCollatorWithPadding:\n    def __call__(self, output):\n        from pathlib import Path\n        Path(output).write_text("child-collation")\nchild=multiprocessing.get_context("spawn").Process(target=DataCollatorWithPadding(),args=(' + repr(str(output)) + ',))\nchild.start()\nchild.join(10)\nassert child.exitcode==0,child.exitcode\n')
        hashes['main.py'] = native.digest(source)
        with patch.object(native, 'UPSTREAM', hashes), patch.object(native, 'require_slurm'), patch.object(native.subprocess, 'check_output', side_effect=git):
            native.prepare(self.root, config, train, validation, run)
        stubs = self.root / 'import-stubs'
        stubs.mkdir()
        (stubs / 'torch.py').write_text('')
        environment = os.environ.copy()
        environment.update(PYTHONPATH=os.pathsep.join([str(stubs), str(Path(native.__file__).resolve().parents[1])]), PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-B', '-c', 'import sys;from taps_rebuild import native_training as native;plan=native.load_plan(sys.argv[1]);before=sys.modules["__main__"];native.run_native_main(plan);assert sys.modules["__main__"] is before;assert "TAPS_NATIVE_SPAWN_PLAN_IDENTITY" not in native.os.environ', str(run / 'plan.json')]
        result = subprocess.run(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(marker.read_text(), 'setup\n')
        self.assertEqual(output.read_text(), 'child-collation')
        self.assertEqual(native.digest(run / 'work/main.py'), hashes['main.py'])
        definition, identity = native.collator_definition(source, hashes['main.py'])
        self.assertTrue(identity['child_exposes_only_collator'])
        self.assertEqual(len(definition.body), 1)
        self.assertEqual(identity['native_loader_workers_unchanged'], 4)

    def test_collator_exposure_rejects_source_mismatch_duplicate_class_and_file_dependency(self):
        source = self.root / 'main.py'
        for content in ('class DataCollatorWithPadding:\n    pass\nclass DataCollatorWithPadding:\n    pass\n', 'class DataCollatorWithPadding:\n    pass\nprint(__file__)\n'):
            source.write_text(content)
            with self.assertRaises(native.NativeTrainingError):
                native.collator_definition(source, native.digest(source))
        with self.assertRaisesRegex(native.NativeTrainingError, 'Frozen native main differs'):
            native.collator_definition(source, '0' * 64)


    def test_real_data_cannot_claim_engineering_exemption(self):
        upstream, hashes, config, train, validation, run, git = self.fixture_plan()
        train.write_bytes(native.canonical(conversation('real-data', 'A real training prompt', 'An answer')) + b'\n')
        with patch.object(native, 'UPSTREAM', hashes), patch.object(native, 'require_slurm'), patch.object(native.subprocess, 'check_output', side_effect=git):
            with self.assertRaisesRegex(native.NativeTrainingError, 'real data require'):
                native.prepare(self.root, config, train, validation, run)
        self.assertFalse(run.exists())

    def test_claimed_research_pass_cannot_replace_missing_full_model_validator(self):
        registry = {'reference_cells': [{'method': method, 'benchmark': 'fixture', 'temperature': 0, 'published_tau': 2.0, 'source_url': 'primary', 'expected_questions': 1, 'expected_turns': 1, 'timing_repeats': 5} for method in ['eagle3', 'eagle2', 'hass']]}
        path = self.root / 'configs/reproduction.json'
        write_json(path, registry)
        report = {'reproduction_gate': {'status': 'passed', 'passed': True, 'reference_cells': 3, 'passed_reference_cells': 3}, 'cells': [{**cell, 'reference': cell, 'reproduction_gate': {'passed': True}} for cell in registry['reference_cells']]}
        write_json(self.root / 'report.json', report)
        write_json(self.root / 'full-model.json', {'status': 'passed', 'models_loaded': True, 'automatic_pipeline_gate_pass': True})
        evidence = {'reproduction_config_sha256': native.digest(path), 'reproduction_report': {'path': 'report.json', 'sha256': native.digest(self.root / 'report.json')}, 'full_model_temperature_one_report': {'path': 'full-model.json', 'sha256': native.digest(self.root / 'full-model.json')}}
        config = {**self.config, 'scope': 'native_training', 'initialize_from_released_draft': False, 'research_gate': evidence}
        native.validate_config(config)
        with self.assertRaisesRegex(native.NativeTrainingError, 'claimed passed metadata'):
            native.research_gate(self.root, config)
        evidence['reproduction_report']['sha256'] = '0' * 64
        with self.assertRaisesRegex(native.NativeTrainingError, 'SHA256 differs'):
            native.research_gate(self.root, config)


    def test_deepspeed_import_observes_rank_device_after_torch_geometry_validation(self):
        class Observed(Exception):
            pass
        original_import = builtins.__import__
        for rank in (0, 7):
            with self.subTest(rank=rank):
                state = {}
                torch = types.ModuleType('torch')
                torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 8,
                    set_device=lambda index: state.update(selected=index), get_device_capability=lambda index: (8, 0))
                package = types.ModuleType('packaging')
                package.__path__ = []
                specifiers = types.ModuleType('packaging.specifiers')
                specifiers.SpecifierSet = object
                plan = {'command': [str(Path(sys.prefix) / 'bin/python')], 'config': {'required_packages': {}, 'world_size': 8}, 'run_dir': str(self.root), 'plan_identity_sha256': 'a' * 64}
                write_json(self.root / 'tokenization-audit.json', {'status': 'passed', 'plan_identity_sha256': 'a' * 64})
                def importing(name, *args, **kwargs):
                    if name == 'deepspeed':
                        self.assertEqual(state.get('selected'), rank)
                        self.assertEqual(os.environ.get('CUBLAS_WORKSPACE_CONFIG'), ':4096:8')
                        raise Observed()
                    return original_import(name, *args, **kwargs)
                with patch.object(native, 'load_plan', return_value=plan), patch.object(native, 'require_slurm'), patch.dict(os.environ, {'RANK': str(rank), 'LOCAL_RANK': str(rank), 'WORLD_SIZE': '8'}), patch.dict(sys.modules, {'torch': torch, 'packaging': package, 'packaging.specifiers': specifiers}), patch.object(builtins, '__import__', side_effect=importing):
                    with self.assertRaises(Observed):
                        native.worker(self.root / 'plan.json')

    def test_runtime_probe_checks_virtual_environment_prefix(self):
        with patch.object(native.subprocess, 'check_output', return_value=json.dumps({'prefix': '/old/environment', 'versions': {'torch': '2.6.0'}})):
            with self.assertRaisesRegex(native.NativeTrainingError, 'prefix differs'):
                native.runtime_probe('/new/environment/bin/python', '/new/environment', {'torch': '==2.6.0'})


if __name__ == '__main__':
    unittest.main()
