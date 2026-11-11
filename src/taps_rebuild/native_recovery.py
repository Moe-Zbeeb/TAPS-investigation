import ast
import hashlib
import json
import os
from pathlib import Path
import random
import shutil


def transform(source, expected):
    data = Path(source).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Native restart transformation requires the pinned upstream bytes')
    tree = ast.parse(data)
    changed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            names = [value.id for value in node.targets[0].elts if isinstance(value, ast.Name)]
            if names == ['checkpoint_path', 'start_epoch']:
                if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name) or node.value.func.id != 'find_max_state_with_file':
                    raise ValueError('Native epoch checkpoint discovery changed')
                node.value = ast.Tuple(elts=[ast.Constant(None), ast.Constant(0)], ctx=ast.Load())
                changed.add('epoch_discovery')
        if not isinstance(node, ast.For):
            continue
        if isinstance(node.target, ast.Name) and node.target.id == 'epoch':
            expected_iter = ast.parse('range(start_epoch, num_epochs)', mode='eval').body
            if ast.dump(node.iter) != ast.dump(expected_iter):
                raise ValueError('Native epoch loop changed')
            node.iter = ast.parse('_taps_recovery.epochs(start_epoch, num_epochs)', mode='eval').body
            changed.add('epoch_loop')
        expected_iter = ast.parse('enumerate(tqdm(train_loader))', mode='eval').body
        if ast.dump(node.iter) == ast.dump(expected_iter):
            node.iter = ast.parse('_taps_recovery.batches(train_loader, epoch)', mode='eval').body
            node.body.append(ast.parse('_taps_recovery.commit_batch()').body[0])
            changed.add('batch_loop')
    if changed != {'epoch_discovery', 'epoch_loop', 'batch_loop'}:
        raise ValueError('Native restart hooks did not match every pinned loop')
    return ast.fix_missing_locations(tree)


class Recovery:
    def __init__(self, plan, torch, rank, observation, helpers, stop, finish):
        self.plan, self.torch, self.rank = plan, torch, rank
        self.observation, self.helpers = observation, helpers
        self.stop, self.finish = stop, finish
        self.engine = observation['engine']
        self.directory = Path(plan['trained_models']['resolved_path']) / 'recovery'
        self.directory.mkdir(exist_ok=True)
        self.namespace = None
        self.cursor = {'epoch': 0, 'next_batch': 0}
        self.restored = None
        self.pending_checkpoint = False
        self.completed_steps = 0
        self.resume_path = os.environ.get('TAPS_NATIVE_RESUME_CHECKPOINT')
        if self.resume_path:
            self.restore(Path(self.resume_path))

    def state_identity(self):
        engine, h, torch = self.engine, self.helpers, self.torch
        return {'trainable_model': h.tensor_state_identity({name: parameter for name, parameter in engine.module.named_parameters() if parameter.requires_grad}, torch), 'optimizer_and_scaler': h.tensor_state_identity(engine.optimizer.state_dict(), torch), 'scheduler': h.tensor_state_identity(engine.lr_scheduler.state_dict(), torch), 'global_steps': int(engine.global_steps), 'skipped_steps': int(engine.skipped_steps), 'micro_steps': int(engine.micro_steps), 'learning_rates': [float(group['lr']) for group in engine.optimizer.param_groups]}

    def rng_state(self):
        import numpy

        return {'python': random.getstate(), 'numpy': numpy.random.get_state(), 'torch_cpu': self.torch.get_rng_state(), 'torch_cuda': self.torch.cuda.get_rng_state()}

    def restore_rng(self, state):
        import numpy

        random.setstate(state['python'])
        numpy.random.set_state(state['numpy'])
        self.torch.set_rng_state(state['torch_cpu'])
        self.torch.cuda.set_rng_state(state['torch_cuda'])

    def rng_identity(self):
        state = self.rng_state()
        numpy = state['numpy']
        state['numpy'] = [numpy[0], numpy[1].tolist(), *numpy[2:]]
        return self.helpers.tensor_state_identity(state, self.torch)

    def epochs(self, start, stop):
        yield from range(max(start, self.cursor['epoch']), stop)

    def sampler(self):
        sampler = self.namespace['train_sampler']
        return {key: getattr(sampler, key) for key in ('epoch', 'seed', 'rank', 'num_replicas', 'shuffle', 'drop_last', 'num_samples', 'total_size')}

    def batches(self, loader, epoch):
        offset = self.cursor['next_batch'] if epoch == self.cursor['epoch'] else 0
        if not 0 <= offset <= len(loader):
            raise ValueError('Restored native data cursor is outside this epoch')
        iterator = iter(loader)
        for _ in range(offset):
            next(iterator)
        if self.restored is not None:
            if self.sampler() != self.restored['sampler']:
                raise ValueError('Restored distributed sampler identity differs')
            self.namespace.update(self.restored['epoch_metrics'])
            self.restore_rng(self.restored['rng'])
            self.observation['restored_rng_and_cursor'] = True
            self.restored = None
        for index, data in enumerate(iterator, start=offset):
            self.cursor = {'epoch': epoch, 'next_batch': index + 1}
            path = Path(self.plan['run_dir']) / 'attempts' / ('attempt-' + os.environ.get('TAPS_NATIVE_ATTEMPT_ID', '1')) / ('batches-rank-' + str(self.rank) + '.jsonl')
            row = {'epoch': epoch, 'batch_index': index, 'input_ids_sha256': self.helpers.tensor_state_identity(data['input_ids'], self.torch), 'attention_mask_sha256': self.helpers.tensor_state_identity(data['attention_mask'], self.torch)}
            with path.open('ab') as handle:
                handle.write(self.helpers.canonical(row) + b'\n')
                handle.flush()
                os.fsync(handle.fileno())
            yield index, data

    def after_step(self, before, skipped_before):
        self.completed_steps = int(self.engine.global_steps) - int(self.engine.skipped_steps)
        policy = self.plan['config']['recovery']
        self.pending_checkpoint = int(self.engine.global_steps) > before and int(self.engine.skipped_steps) == skipped_before and self.completed_steps > 0 and self.completed_steps % policy['checkpoint_interval'] == 0

    def save(self):
        h, engine, torch = self.helpers, self.engine, self.torch
        path = self.directory / ('attempt-' + os.environ.get('TAPS_NATIVE_ATTEMPT_ID', '1') + '-step-' + str(engine.global_steps))
        path.mkdir(exist_ok=False)
        state = {'plan_identity_sha256': self.plan['plan_identity_sha256'], 'cursor': dict(self.cursor), 'sampler': self.sampler(), 'rng': self.rng_state(), 'epoch_metrics': {key: self.namespace[key] for key in ('epoch_acces', 'epoch_plosses')}, 'token_counts': dict(self.observation['token_counts']), 'state_identity': self.state_identity()}
        engine.save_checkpoint(str(path), tag='state', client_state={'plan_identity_sha256': self.plan['plan_identity_sha256'], 'cursor': self.cursor}, exclude_frozen_parameters=True)
        rank_path = path / ('rank-' + str(self.rank) + '.pt')
        temporary = rank_path.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(rank_path)
        torch.distributed.barrier()
        if self.rank == 0:
            files = {str(file.relative_to(path)): h.digest(file) for file in sorted(path.rglob('*')) if file.is_file()}
            launch = json.loads((Path(self.plan['run_dir']) / 'launch.json').read_bytes())
            manifest = {'format': 'native-process-recovery-v1', 'plan_identity_sha256': self.plan['plan_identity_sha256'], 'world_size': self.plan['config']['world_size'], 'hardware_identity_sha256': launch['hardware']['hardware_identity_sha256'], 'files': files, 'global_step': int(engine.global_steps), 'cursor': self.cursor}
            h.atomic_json(path / 'manifest.json', manifest)
            h.atomic_json(self.directory / 'latest.json', {'path': str(path), 'manifest_sha256': h.digest(path / 'manifest.json'), 'plan_identity_sha256': self.plan['plan_identity_sha256']})
            committed = []
            for candidate in self.directory.glob('attempt-*-step-*'):
                if candidate.resolve() != candidate or not (candidate / 'manifest.json').is_file():
                    continue
                receipt = json.loads((candidate / 'manifest.json').read_bytes())
                if receipt['plan_identity_sha256'] == self.plan['plan_identity_sha256']:
                    committed.append((receipt['global_step'], int(candidate.name.split('-')[1]), candidate))
            for _, _, candidate in sorted(committed, reverse=True)[2:]:
                archive = Path(self.plan['run_dir']) / 'recovery-receipts'
                archive.mkdir(exist_ok=True)
                shutil.copyfile(candidate / 'manifest.json', archive / (candidate.name + '.json'))
                shutil.rmtree(candidate)
        torch.distributed.barrier()
        return path

    def restore(self, path):
        h, torch = self.helpers, self.torch
        if path.resolve() != path or path.parent != self.directory or not path.is_dir():
            raise ValueError('Native recovery path is outside the original trained-model run')
        manifest = json.loads((path / 'manifest.json').read_bytes())
        if manifest['plan_identity_sha256'] != self.plan['plan_identity_sha256'] or manifest['world_size'] != self.plan['config']['world_size']:
            raise ValueError('Native checkpoint plan or world size differs')
        if self.rank == 0:
            if any(h.digest(path / name) != digest for name, digest in manifest['files'].items()):
                raise ValueError('Committed native checkpoint content changed')
        torch.distributed.barrier()
        loaded, client = self.engine.load_checkpoint(str(path), tag='state', load_module_strict=False, load_optimizer_states=True, load_lr_scheduler_states=True)
        if not loaded or client['plan_identity_sha256'] != self.plan['plan_identity_sha256']:
            raise ValueError('Native engine did not load the bound process checkpoint')
        state = torch.load(path / ('rank-' + str(self.rank) + '.pt'), map_location='cpu', weights_only=False)
        if state['plan_identity_sha256'] != self.plan['plan_identity_sha256'] or client['cursor'] != state['cursor']:
            raise ValueError('Native rank state and engine checkpoint disagree')
        self.engine.micro_steps = state['state_identity']['micro_steps']
        actual = self.state_identity()
        expected = state['state_identity']
        if any(actual[key] != expected[key] for key in actual if key != 'learning_rates') or len(actual['learning_rates']) != len(expected['learning_rates']) or any(abs(a-b) > max(1e-15, abs(b)*1e-12) for a,b in zip(actual['learning_rates'], expected['learning_rates'])):
            raise ValueError('Native model, optimizer, scaler, scheduler or counters differ after process restart')
        self.cursor, self.restored = state['cursor'], state
        self.observation['token_counts'].update(state['token_counts'])
        self.observation['process_recovery'] = {'status': 'loaded_exact_state', 'path': str(path), 'state_identity': actual, 'cursor': self.cursor, 'rng_restore_deferred_until_loader_skip': True}

    def commit_batch(self):
        if not self.pending_checkpoint:
            return
        path = self.save()
        self.pending_checkpoint = False
        policy = self.plan['config']['recovery']
        if not self.resume_path and self.completed_steps == policy['stop_after_optimizer_step']:
            self.stop(str(path))
        cap = self.plan['config']['budget']['optimizer_steps']
        if cap is not None and self.completed_steps >= cap:
            self.finish(self.completed_steps)


def validate_policy(value):
    if not isinstance(value, dict) or set(value) != {'checkpoint_interval', 'stop_after_optimizer_step'}:
        raise ValueError('Native recovery requires an explicit checkpoint interval and optional stop boundary')
    interval, stop = value['checkpoint_interval'], value['stop_after_optimizer_step']
    if type(interval) is not int or interval < 1 or (stop is not None and (type(stop) is not int or stop < 1 or stop % interval)):
        raise ValueError('Native recovery stop must be a positive checkpoint boundary')
    return value
