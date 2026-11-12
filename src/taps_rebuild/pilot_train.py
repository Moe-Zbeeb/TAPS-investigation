import argparse
import ast
import fcntl
import importlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time

from . import native_training as native, pool
from .pilot_cache import read, records, require
from .pilot_gate import backend, devices, upstream, verify_cache
from .pilot_study import mask_positions, verify_identity


def admission(plan, verify_content=True):
    directory = Path(plan['study_directory'])
    binding = {'study_manifest_sha256': pool.digest(directory / 'manifest.json'), 'source_sha256': plan['source_sha256'], 'recipe_sha256': plan['recipe_sha256'], 'cache_identity_sha256': plan['cache_identity_sha256'], 'eligibility_identity_sha256': plan['eligibility_identity_sha256'], 'hardware': plan['hardware'], 'vocabulary_identity_sha256': plan['vocabulary_identity_sha256']}
    receipt = verify_identity(read(directory / 'admission.json'))
    require(receipt['scientific_training_admitted'] is True and receipt['binding'] == binding, 'Pilot training has no exact study-bound admission')
    report_path = Path(receipt['qualification_report']['path'])
    require(pool.digest(report_path) == receipt['qualification_report']['sha256'], 'Pilot qualification evidence changed')
    report = read(report_path)
    require(report['status'] == 'passed' and report['binding'] == binding and all(report['checks'][name] == 'passed' for name in plan['recipe']['admission_required']), 'Pilot qualifications did not all pass')
    require(bool(report['evidence_files']), 'Pilot qualification has no retained evidence inventory')
    if verify_content:
        require(all(Path(name).stat().st_size == item['size_bytes'] and pool.digest(name) == item['sha256'] for name, item in report['evidence_files'].items()), 'Retained pilot qualification evidence changed')
    review_path = Path(receipt['independent_review']['path'])
    require(pool.digest(review_path) == receipt['independent_review']['sha256'], 'Pilot independent evidence review changed')
    review = read(review_path)
    require(review['scientific_training_admitted'] is True and review['qualification_report_sha256'] == receipt['qualification_report']['sha256'] and review['binding'] == binding, 'Pilot evidence has not been independently admitted for this exact study')
    return receipt


def load_plan(path, qualification):
    plan = verify_identity(read(path))
    require(plan['scope'] == 'scientific_pilot_requires_admission' and plan['steps'] == 1536 and plan['quota'] in (64, 32), 'Pilot plan contract differs')
    source = Path(__file__).parent
    require(source.resolve() == Path(plan['source_directory']), 'Pilot source directory differs from the frozen study')
    require(all(pool.digest(source / name) == expected for name, expected in plan['source_sha256'].items()), 'Pilot executing source differs from frozen plan')
    if not qualification:
        admission(plan, verify_content=int(os.environ.get('RANK', '0')) == 0)
    return plan


def rng_state(torch):
    import numpy as np
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]], 'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state()}


def restore_rng(value, torch):
    import numpy as np
    random.setstate(value['python'])
    state = value['numpy']
    np.random.set_state((state[0], np.array(state[1], dtype='uint32'), *state[2:]))
    torch.set_rng_state(value['torch_cpu'])
    torch.cuda.set_rng_state(value['torch_cuda'])


def state_identity(engine, torch):
    parameters = {name: value for name, value in engine.module.named_parameters() if value.requires_grad}
    parameters.update({key: getattr(engine.module, key) for key in ('d2t', 't2d')})
    return {'model': native.tensor_state_identity(parameters, torch), 'optimizer_scaler': native.tensor_state_identity(engine.optimizer.state_dict(), torch), 'scheduler': native.tensor_state_identity(engine.lr_scheduler.state_dict(), torch), 'rng': native.tensor_state_identity(rng_state(torch), torch), 'global_steps': int(engine.global_steps), 'skipped_steps': int(engine.skipped_steps), 'micro_steps': int(engine.micro_steps)}


def model_directory(plan, directory, qualification):
    storage = (Path(plan['project_root']) / 'models').resolve()
    result = storage / 'trained/pilot-qualifications' / directory.relative_to(Path(plan['project_root']) / 'outputs') if qualification else Path(plan['trained_models'])
    require(result.is_relative_to(storage) and result.resolve() == result, 'Pilot checkpoint storage is outside project models/')
    return result


def save(engine, plan, directory, model_root, trajectory, step, torch, rank):
    path = model_root / 'checkpoints' / ('step-' + str(step))
    staging = path.with_name('.' + path.name + '-' + trajectory['attempt'] + '.pending')
    if rank == 0:
        require(not path.exists(), 'A committed checkpoint already occupies this step')
        staging.mkdir(parents=True, exist_ok=False)
    torch.distributed.barrier()
    state = {'plan_identity_sha256': plan['identity_sha256'], 'trajectory': {key: value for key, value in trajectory.items() if key != 'attempt'}, 'next_step': step, 'rng': rng_state(torch), 'state_identity': state_identity(engine, torch)}
    engine.save_checkpoint(str(staging), tag='state', client_state={'plan_identity_sha256': plan['identity_sha256'], 'trajectory': state['trajectory'], 'next_step': step}, exclude_frozen_parameters=True)
    temporary = staging / ('rank-' + str(rank) + '.tmp')
    torch.save(state, temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(staging / ('rank-' + str(rank) + '.pt'))
    torch.distributed.barrier()
    if rank == 0:
        files = {}
        for item in sorted(staging.rglob('*')):
            if item.is_file():
                with item.open('rb') as stream:
                    os.fsync(stream.fileno())
                files[str(item.relative_to(staging))] = pool.digest(item)
        manifest = {'plan_identity_sha256': plan['identity_sha256'], 'trajectory': state['trajectory'], 'source_sha256': plan['source_sha256'], 'hardware': plan['hardware'], 'next_step': step, 'world_size': 8, 'files': files}
        manifest['identity_sha256'] = pool.identity(manifest)
        pool.atomic_json(staging / 'manifest.json', manifest)
        staging.replace(path)
        pool.atomic_json(directory / 'latest.json', {'path': str(path), 'manifest_sha256': pool.digest(path / 'manifest.json'), 'next_step': step})
        committed = []
        for candidate in path.parent.glob('step-*'):
            if candidate.resolve() == candidate and (candidate / 'manifest.json').is_file():
                receipt = verify_identity(read(candidate / 'manifest.json'))
                require(receipt['plan_identity_sha256'] == plan['identity_sha256'], 'Another trajectory occupies pilot checkpoint storage')
                committed.append((receipt['next_step'], candidate))
        for _, candidate in sorted(committed, reverse=True)[2:]:
            archive = directory / 'checkpoint-receipts'
            archive.mkdir(exist_ok=True)
            shutil.copyfile(candidate / 'manifest.json', archive / (candidate.name + '.json'))
            shutil.rmtree(candidate)
    torch.distributed.barrier()
    return path


def restore(engine, plan, trajectory, path, torch, rank):
    manifest = verify_identity(read(path / 'manifest.json'))
    require(manifest['plan_identity_sha256'] == plan['identity_sha256'] and manifest['trajectory'] == {key: value for key, value in trajectory.items() if key != 'attempt'} and manifest['source_sha256'] == plan['source_sha256'] and manifest['hardware'] == plan['hardware'] and manifest['world_size'] == 8, 'Pilot checkpoint trajectory binding differs')
    if rank == 0:
        require(all(pool.digest(path / name) == expected for name, expected in manifest['files'].items()), 'Committed pilot checkpoint bytes changed')
    torch.distributed.barrier()
    loaded, client = engine.load_checkpoint(str(path), tag='state', load_module_strict=False, load_optimizer_states=True, load_lr_scheduler_states=True)
    require(loaded and client['plan_identity_sha256'] == plan['identity_sha256'] and client['trajectory'] == manifest['trajectory'] and client['next_step'] == manifest['next_step'], 'Pilot engine checkpoint restore failed')
    state = torch.load(path / ('rank-' + str(rank) + '.pt'), map_location='cpu', weights_only=False)
    require(state['plan_identity_sha256'] == plan['identity_sha256'] and state['trajectory'] == manifest['trajectory'] and state['next_step'] == manifest['next_step'], 'Pilot rank checkpoint differs')
    engine.micro_steps = state['state_identity']['micro_steps']
    restore_rng(state['rng'], torch)
    require(state_identity(engine, torch) == state['state_identity'], 'Fresh-process pilot state differs after restore')
    return state['next_step']


class Guard:
    def __init__(self, model, torch, quota, capture=False):
        self.model, self.torch, self.quota, self.capture = model, torch, quota, capture
        self.positions, self.heads, self.targets, self.logits, self.gradients, self.sparse_gradients = [], 0, None, [], {}, {}
        self.handles = [model.target_model.register_forward_hook(self.target), model.lm_head.register_forward_hook(self.head)]

    def begin(self, positions):
        require(len(positions) == self.quota and len(set(positions)) == self.quota and min(positions) >= 6, 'Pilot selected-mask geometry differs')
        self.positions, self.heads, self.targets, self.logits, self.gradients, self.sparse_gradients = positions, 0, None, [], {}, {}

    def target(self, module, inputs, output):
        torch = self.torch
        require(not module.training and all(not child.training for child in module.modules()), 'Online teacher left evaluation mode')
        selected = output.logits[0, [value + 1 for value in self.positions]]
        require(bool(torch.isfinite(selected).all()) and bool(self.model.t2d[selected.argmax(-1)].all()), 'Online teacher gate no longer covers exactlyQ selected contexts')
        if self.capture:
            self.targets = selected.detach().cpu()

    def head(self, module, inputs, output):
        torch, head = self.torch, self.heads
        require(head < 7 and output.shape == (1, 2048, 32000) and output.dtype == torch.float16, 'Native seven-head output contract differs')
        self.heads += 1
        selected = [value - head for value in self.positions]
        if self.capture:
            self.logits.append(output[0, selected].detach().cpu())
        def gradient(value):
            keep = torch.zeros(2048, dtype=torch.bool, device=value.device)
            keep[selected] = True
            require(bool(torch.isfinite(value).all()) and bool((value[0, ~keep] == 0).all()), 'Native head has nonfinite or outside-selected logit gradients')
            if self.capture:
                self.gradients[head] = value[0, selected].detach().cpu()
                self.sparse_gradients[head] = value[0].detach().to_sparse(1).cpu()
            return value
        output.register_hook(gradient)

    def finish(self):
        require(self.heads == 7 and (not self.capture or set(self.gradients) == set(range(7))), 'Native effective seven-head coverage differs')

    def evidence(self, losses, torch):
        require(self.capture, 'Gradient evidence capture was not enabled')
        return {'base_positions': self.positions, 'head_grid_indices': [[value - head for value in self.positions] for head in range(7)], 'teacher_full_vocab_logits_original_s_plus_one': self.targets, 'head_logits_fp16_before_float_cast': self.logits, 'head_logit_gradients_fp16_scaled128_before_unscale_and_dp_reduce': [self.gradients[head] for head in range(7)], 'full_row_sparse_head_logit_gradients_fp16': [self.sparse_gradients[head] for head in range(7)], 'native_head_mean_losses': [float(value.detach()) for value in losses], 'loss_scale': 128, 'head_weights': [0.8 ** head for head in range(7)], 'normalization': 2048 / self.quota, 'quota': self.quota, 'vocabulary_t2d': self.model.t2d.detach().cpu(), 'dtype_units': 'FP16 lm_head output gradient hook before FP32 cast backward, loss scaling128, unscale and DP reduction not yet applied'}

    def close(self):
        for handle in self.handles:
            handle.remove()


def native_reference_step(source, engine, batch, quota, rank):
    require(pool.digest(source) == native.UPSTREAM['main.py'], 'Native reference source changed')
    tree = ast.parse(Path(source).read_bytes())
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.For) and ast.dump(node.iter) == ast.dump(ast.parse('enumerate(tqdm(train_loader))', mode='eval').body)]
    require(len(candidates) == 1, 'Native reference training loop changed')
    body = []
    for node in candidates[0].body:
        body.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'loss' for target in node.targets):
            body.append(ast.parse('loss = loss * normalization').body[0])
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'step':
            break
    require(len(body) == 8 and isinstance(body[-1], ast.Expr) and ast.unparse(body[-1]) == 'model_engine.step()', 'Native reference step AST changed')
    namespace = {'model': engine.module, 'model_engine': engine, 'data': batch, 'rank': rank, 'normalization': 2048 / quota}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), 'exec'), namespace)
    return namespace['plosses'], namespace['loss']


def update(engine, batch, quota):
    engine.module.zero_grad()
    losses, _, _ = engine(**batch)
    require(len(losses) == 7, 'Native training head count differs')
    loss = sum((0.8 ** head) * value for head, value in enumerate(losses)) * (2048 / quota)
    engine.backward(loss)
    engine.step()
    return losses, loss


def worker(plan_path, directory, attempt, stop, resume=None, qualification=False, reference=False, capture=False, alternate_pad=False, allocation_stop_epoch=None):
    require(os.environ.get('SLURM_JOB_ID') and int(os.environ['WORLD_SIZE']) == 8, 'Pilot worker requires eight-GPU Slurm')
    import numpy as np
    import torch
    import deepspeed
    from accelerate.utils import set_seed

    plan = load_plan(plan_path, qualification)
    rank = int(os.environ['LOCAL_RANK'])
    require(rank == int(os.environ['RANK']) and torch.cuda.device_count() == 8, 'Pilot worker rank geometry differs')
    torch.cuda.set_device(rank)
    runtime, hardware = backend(torch), devices()
    require(runtime == plan['scan_runtime'] and hardware == plan['hardware'], 'Training differs from eligibility runtime or ordered physical hardware')
    require(0 < stop <= (4 if qualification else 1536) and (qualification or stop % plan['recipe']['checkpoint_interval'] == 0), 'Pilot stop boundary is not registered')
    directory = Path(directory).resolve()
    expected = Path(plan['project_root']) / 'outputs'
    require(directory.is_relative_to(expected), 'Pilot journals must stay under project outputs/')
    require(directory.is_relative_to(expected / 'pilot-qualifications' / plan['study_id']) if qualification else directory == Path(plan['output']), 'Pilot execution directory differs from its registered namespace')
    directory.mkdir(parents=True, exist_ok=True)
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', attempt), 'Pilot attempt name is invalid')
    model_root = model_directory(plan, directory, qualification)
    model_root.mkdir(parents=True, exist_ok=True)
    run_lock = None
    if rank == 0:
        run_lock = (directory / 'run.lock').open('a+')
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    trajectory = {'directory': str(directory), 'qualification': qualification, 'reference': reference, 'alternate_pad': alternate_pad, 'attempt': attempt}
    attempt_dir = directory / 'attempts' / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    journal = attempt_dir / ('rank-' + str(rank) + '.jsonl')
    require(not journal.exists(), 'Pilot attempt journal already exists')
    deepspeed.init_distributed(dist_backend='nccl')
    cache = Path(plan['cache'])
    cache_receipt = verify_cache(cache, rank)
    require(cache_receipt['identity_sha256'] == plan['cache_identity_sha256'], 'Pilot cache binding changed')
    eligible = verify_identity(read(Path(plan['eligibility']) / 'completion.json'))
    require(eligible['identity_sha256'] == plan['eligibility_identity_sha256'], 'Pilot eligibility binding changed')
    eligible_path = Path(plan['eligibility']) / ('rank-' + str(rank) + '.eligible.jsonl')
    require(pool.digest(eligible_path) == eligible['files'][eligible_path.name]['sha256'], 'Pilot effective positions changed')
    rows = {row['index']: row for row in records(eligible_path)}
    descriptor = plan['schedules'][rank]
    require(pool.digest(descriptor['path']) == descriptor['sha256'], 'Pilot schedule bytes changed')
    schedules = list(records(descriptor['path']))
    require(len(schedules) == 1536, 'Pilot rank schedule coverage differs')
    source = upstream(plan['project_root'])
    require(plan['ancestor']['files']['config.json']['sha256'] == native.UPSTREAM['config.json'], 'Released ancestor architecture differs from pinned native configuration')
    for name, identifier in (('target', 'llama31_instruct_8b'), ('ancestor', 'eagle3_llama31_8b')):
        record = native.model_record(read(Path(plan['project_root']) / 'outputs/preparation/assets.json'), identifier, verify_content=rank == 0)
        require(record == plan[name], 'Pilot prepared model inventory changed')
    set_seed(plan['seed'])
    cnets = importlib.import_module('cnets')
    native.install_model_adapter(cnets, {'draft': plan['ancestor'], 'workdir': str(directory)}, torch)
    config = importlib.import_module('configs').EConfig.from_pretrained(str(source / 'config.json'))
    training = {'bs': 1, 'num_epochs': 1, 'num_workers': 0, 'max_len': 2048, 'config_path': str(source / 'config.json'), 'gradient_checkpoint': True}
    model = cnets.Model(config, plan['recipe']['ds_config'], training, path=plan['target']['path'], load_emb=True, load_head=True)
    model.scandata(None, plan['target']['path'])
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=plan['recipe']['ds_config'])
    engine.train()
    model.target_model.eval()
    initial = state_identity(engine, torch)
    initial_names = sorted(name for name, value in model.named_parameters() if value.requires_grad)
    require(not any(name.startswith(('target_model.', 'embed_tokens.')) for name in initial_names), 'Frozen teacher or embedding is trainable')
    released = torch.load(Path(plan['ancestor']['path']) / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    expected_initial = {name: released[name].to(value.dtype) for name, value in model.named_parameters() if value.requires_grad}
    expected_initial.update({key: released[key] for key in ('d2t', 't2d')})
    require(native.tensor_state_identity(expected_initial, torch) == initial['model'], 'Pilot initialization differs from complete released common ancestor')
    del released, expected_initial
    pool.atomic_json(attempt_dir / ('rank-' + str(rank) + '.initial.json'), {'state': initial, 'trainable_names': initial_names, 'plan_identity_sha256': plan['identity_sha256'], 'qualification': qualification, 'reference': reference})
    if resume:
        require(Path(resume).resolve() == Path(resume) and Path(resume).parent == model_root / 'checkpoints', 'Pilot resume checkpoint is outside its model run')
    start = restore(engine, plan, trajectory, Path(resume), torch, rank) if resume else 0
    require(start < stop, 'Pilot checkpoint cursor is beyond requested stop')
    count = cache_receipt['rank_examples'][rank]
    ids = np.memmap(cache / ('rank-' + str(rank) + '.ids'), mode='r', dtype='<i4', shape=(count, 2, 2048))
    guard = Guard(model, torch, plan['quota'], capture)
    gradient_observation = {}
    if qualification:
        original_step = engine.step
        def observed_step(*args, **kwargs):
            gradients = engine.optimizer.averaged_gradients
            require(isinstance(gradients, dict) and gradients and any(torch.is_tensor(value) and value.numel() for values in gradients.values() for value in values), 'Owned ZeRO2 pre-step gradient partitions are unavailable')
            gradient_observation['identity_sha256'] = native.tensor_state_identity(gradients, torch)
            masters = engine.optimizer.single_partition_of_fp32_groups
            before = [value.detach().clone() for value in masters]
            gradient_observation['learning_rates_before_step'] = [float(group['lr']) for group in engine.optimizer.param_groups]
            result = original_step(*args, **kwargs)
            gradient_observation['local_fp32_master_update_l2'] = float(sum((left.double() - right.double()).square().sum() for left, right in zip(before, masters)).sqrt())
            return result
        engine.step = observed_step
    cuda_seconds, checkpoint_seconds = 0.0, 0.0
    torch.distributed.barrier()
    loop_started = time.monotonic()
    for step in range(start, stop):
        item = schedules[step]
        row = rows[item['index']]
        require(item['step'] == step and item['rank'] == rank and row['rank'] == rank and row['split'] == 'train' and row['example_id'] == item['example_id'] and row['domain'] == item['domain'], 'Pilot scheduled example/split differs')
        policy = plan['policy_index']
        positions = mask_positions(row, plan['quota'], plan['seed'], item['occurrence'], policy)
        require(pool.identity(positions) == item['mask_sha256'][policy] and max(positions) + 1 < row['lengths'][policy], 'Pilot selected mask differs from frozen schedule')
        torch.cuda.synchronize(rank)
        started = time.monotonic()
        with journal.open('ab', buffering=0) as stream:
            stream.write(native.canonical({'event': 'begin', 'step': step, 'example_id': row['example_id'], 'occurrence': item['occurrence'], 'mask_sha256': item['mask_sha256'][policy], 'physical_attempt': attempt}) + b'\n')
            os.fsync(stream.fileno())
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        tokens = np.array(ids[item['index'], policy], dtype='int64')
        if alternate_pad:
            tokens[row['lengths'][policy]:] = 0
        batch = {'input_ids': torch.as_tensor(tokens, device=rank)[None], 'attention_mask': torch.ones((1, 2048), dtype=torch.long, device=rank), 'loss_mask': torch.zeros((1, 2048), dtype=torch.long, device=rank)}
        batch['loss_mask'][0, positions] = 1
        guard.begin(positions)
        skipped = int(engine.skipped_steps)
        old_step = int(engine.global_steps)
        losses, loss = native_reference_step(source / 'main.py', engine, batch, plan['quota'], rank) if reference else update(engine, batch, plan['quota'])
        end.record()
        torch.cuda.synchronize(rank)
        guard.finish()
        require(bool(torch.isfinite(loss).all()) and int(engine.skipped_steps) == skipped == 0 and int(engine.global_steps) == old_step + 1 == step + 1, 'Pilot nonfinite loss, overflow or successful-step budget mismatch')
        elapsed = time.monotonic() - started
        event_seconds = begin.elapsed_time(end) / 1000
        cuda_seconds += event_seconds
        record = {'event': 'committed_update', 'step': step, 'global_step': int(engine.global_steps), 'skipped_steps': int(engine.skipped_steps), 'loss': float(loss.detach()), 'native_head_losses': [float(value.detach()) for value in losses], 'effective_base_contexts': plan['quota'], 'effective_head_contexts': [plan['quota']] * 7, 'dense_slots': 2048, 'real_tokens': row['lengths'][policy], 'cuda_default_stream_seconds': event_seconds, 'rank_loop_seconds': elapsed, 'logical_examples_per_rank': step + 1, 'physical_attempt': attempt}
        if qualification:
            record['state_identity'] = state_identity(engine, torch)
            record['pre_step_scaled_gradient_partition_sha256'] = gradient_observation['identity_sha256']
            record['learning_rates_before_step'] = gradient_observation['learning_rates_before_step']
            record['local_fp32_master_update_l2'] = gradient_observation['local_fp32_master_update_l2']
        with journal.open('ab', buffering=0) as stream:
            stream.write(native.canonical(record) + b'\n')
            os.fsync(stream.fileno())
        if capture:
            evidence = guard.evidence(losses, torch)
            evidence.update({'rank': rank, 'step': step, 'domain': row['domain'], 'policy': plan['policy'], 'example_id': row['example_id'], 'cache_index': item['index'], 'plan_identity_sha256': plan['identity_sha256'], 'input_ids': batch['input_ids'].detach().cpu(), 'input_token_sha256': native.tensor_state_identity(batch['input_ids'], torch), 'real_length': row['lengths'][policy], 'alternate_pad': alternate_pad, 'native_reference': reference, 'actual_normalized_total_loss': float(loss.detach()), 'learning_rates_before_step': gradient_observation['learning_rates_before_step'], 'local_fp32_master_update_l2': gradient_observation['local_fp32_master_update_l2']})
            torch.save(evidence, attempt_dir / ('rank-' + str(rank) + '-step-' + str(step) + '-gradients.pt'))
        if (step + 1) % plan['recipe']['checkpoint_interval'] == 0 or step + 1 == stop:
            checkpoint_started = time.monotonic()
            save(engine, plan, directory, model_root, trajectory, step + 1, torch, rank)
            checkpoint_seconds += time.monotonic() - checkpoint_started
        if rank == 0:
            print(json.dumps({'run_id': plan['run_id'], 'step': step + 1, 'stop': stop, 'loss': record['loss']}), flush=True)
        if not qualification and allocation_stop_epoch is not None and (step + 1) % plan['recipe']['checkpoint_interval'] == 0:
            pause = torch.tensor(int(time.time() >= allocation_stop_epoch), dtype=torch.int32, device=rank)
            torch.distributed.all_reduce(pause, op=torch.distributed.ReduceOp.MAX)
            if int(pause):
                stop = step + 1
                break
    guard.close()
    torch.distributed.barrier()
    loop_seconds = time.monotonic() - loop_started - checkpoint_seconds
    final = state_identity(engine, torch)
    report = {'status': 'completed_prefix' if stop < 1536 else 'completed_fixed_budget', 'qualification': qualification, 'reference': reference, 'plan_identity_sha256': plan['identity_sha256'], 'source_sha256': plan['source_sha256'], 'rank': rank, 'start_step': start, 'stop_step': stop, 'successful_updates_this_attempt': stop - start, 'logical_examples_per_rank': stop, 'logical_effective_contexts_per_rank': stop * plan['quota'], 'cuda_default_stream_seconds_this_attempt': cuda_seconds, 'rank_loop_seconds_this_attempt': loop_seconds, 'checkpoint_seconds_this_attempt': checkpoint_seconds, 'physical_attempts_this_process': stop - start, 'state_identity': final, 'journal_sha256': pool.digest(journal), 'vocabulary_identity_sha256': native.tensor_state_identity({key: getattr(model, key) for key in ('d2t', 't2d')}, torch)}
    pool.atomic_json(attempt_dir / ('rank-' + str(rank) + '.complete.json'), report)
    if stop == 1536 or qualification:
        export = model_root / 'exports' / ('step-' + str(stop))
        export.mkdir(parents=True, exist_ok=True)
        engine.save_16bit_model(str(export), exclude_frozen_parameters=True)
        if rank == 0:
            (export / 'config.json').write_bytes((source / 'config.json').read_bytes())
            exported = torch.load(export / 'pytorch_model.bin', map_location='cpu', weights_only=True)
            expected = {name: value for name, value in model.named_parameters() if value.requires_grad}
            expected.update({key: getattr(model, key) for key in ('d2t', 't2d')})
            require(set(expected) <= set(exported) and not any(name.startswith('target_model.') for name in exported) and all(torch.equal(exported[name], value.detach().cpu()) for name, value in expected.items()), 'Export differs from trained parameters or frozen vocabulary')
            export_identity = native.tensor_state_identity({name: exported[name] for name in expected}, torch)
            require(export_identity == final['model'], 'Export trainable identity differs from final committed state')
            inventory = {item.name: {'sha256': pool.digest(item), 'size_bytes': item.stat().st_size} for item in export.iterdir() if item.is_file()}
            pool.atomic_json(export / 'export.json', {'status': 'exported_requires_stock_inference_check', 'plan_identity_sha256': plan['identity_sha256'], 'state_identity_sha256': export_identity, 'files': inventory, 'inference_validated': False})
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if run_lock:
        run_lock.close()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--stop', type=int, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--qualification', action='store_true')
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--alternate-pad', action='store_true')
    parser.add_argument('--allocation-stop-epoch', type=float)
    args = parser.parse_args()
    require(not (args.reference or args.capture or args.alternate_pad) or args.qualification, 'Numerical/reference modes are engineering qualifications only')
    print(json.dumps(worker(args.plan, args.directory, args.attempt, args.stop, args.resume, args.qualification, args.reference, args.capture, args.alternate_pad, args.allocation_stop_epoch), indent=2))


if __name__ == '__main__':
    main()
