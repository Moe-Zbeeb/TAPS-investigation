import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import re
import stat
import tempfile
from pathlib import Path


IMPLEMENTATION = 'REFERENCE_PROTOTYPE'


class LearningError(ValueError):
    pass


def _torch():
    try:
        import torch
    except ImportError as error:
        raise LearningError('Torch is required for tensor operations; no dependency installation is performed') from error
    return torch


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise LearningError(name + ' must be an integer >= ' + str(minimum))
    return value


def _number(value, name, minimum=0, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum or (positive and value == minimum):
        raise LearningError('Invalid finite value for ' + name)
    return float(value)


def model_config(value):
    config = copy.deepcopy(value)
    defaults = {'num_hidden_layers': 1, 'num_experts': 2, 'expert_mode': 'separate_heads', 'lora_rank': 4, 'max_sequence_length': 128, 'dropout': 0.0, 'pad_token_id': 0}
    for name, default in defaults.items():
        config.setdefault(name, default)
    for name in ('feature_dim', 'vocab_size', 'hidden_size', 'num_attention_heads', 'num_hidden_layers', 'num_experts', 'lora_rank', 'max_sequence_length'):
        _integer(config.get(name), name)
    if config['hidden_size'] % config['num_attention_heads'] or config['num_hidden_layers'] not in (1, 2, 4):
        raise LearningError('Hidden size must divide attention heads and layers must be 1, 2 or 4')
    if config['expert_mode'] not in ('separate_heads', 'lora_output'):
        raise LearningError('expert_mode must be separate_heads or lora_output')
    if config['lora_rank'] > min(config['hidden_size'], config['vocab_size']):
        raise LearningError('LoRA rank must not exceed hidden or vocabulary dimension')
    config['dropout'] = _number(config['dropout'], 'dropout')
    if config['dropout'] >= 1:
        raise LearningError('dropout must be smaller than one')
    _integer(config['pad_token_id'], 'pad_token_id', 0)
    if config['pad_token_id'] >= config['vocab_size']:
        raise LearningError('pad_token_id is outside the vocabulary')
    config['lora_alpha'] = _number(config.get('lora_alpha', config['lora_rank']), 'lora_alpha', positive=True)
    return config


def normalize_config(value):
    config = copy.deepcopy(value)
    config.setdefault('scope', 'engineering')
    if config['scope'] not in ('engineering', 'research'):
        raise LearningError('scope must be engineering or research')
    for name in ('experiment', 'run_id'):
        if not isinstance(config.get(name), str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', config[name]):
            raise LearningError('Invalid experiment or run_id')
    config['model'] = model_config(config['model'])
    config.setdefault('seed', 20261109)
    _integer(config['seed'], 'seed', 0)
    training = config.setdefault('training', {})
    for name, default in {'optimizer_steps': 2, 'batch_size': 2, 'gradient_accumulation_steps': 1, 'learning_rate': 0.001, 'weight_decay': 0.0, 'max_grad_norm': 1.0, 'checkpoint_every_steps': 1, 'shuffle': True}.items():
        training.setdefault(name, default)
    for name in ('optimizer_steps', 'batch_size', 'gradient_accumulation_steps', 'checkpoint_every_steps'):
        _integer(training[name], name)
    for name in ('learning_rate', 'max_grad_norm'):
        training[name] = _number(training[name], name, positive=True)
    training['weight_decay'] = _number(training['weight_decay'], 'weight_decay')
    if type(training['shuffle']) is not bool:
        raise LearningError('shuffle must be a boolean')
    loss = config.setdefault('loss', {})
    for name, default in {'objective': 'hard_ce', 'temperature': 1.0, 'feature_mse_weight': 0.0, 'feature_mse_target': 'next_features', 'domain_gate_weight': 0.0}.items():
        loss.setdefault(name, default)
    if loss['objective'] not in ('hard_ce', 'teacher_kl') or loss['feature_mse_target'] not in ('next_features', 'input_reconstruction'):
        raise LearningError('Unsupported loss objective or feature target')
    loss['temperature'] = _number(loss['temperature'], 'temperature', positive=True)
    for name in ('feature_mse_weight', 'domain_gate_weight'):
        loss[name] = _number(loss[name], name)
    if loss['feature_mse_weight'] and loss['feature_mse_target'] == 'input_reconstruction' and config['scope'] != 'engineering':
        raise LearningError('Input-feature reconstruction is an explicitly engineering-only objective')
    data = config.setdefault('data', {})
    if 'synthetic' in data and type(data['synthetic']) is not bool:
        raise LearningError('data.synthetic must be a boolean')
    if bool(data.get('synthetic')) == bool(data.get('path')):
        raise LearningError('Choose exactly one synthetic dataset or local data.path')
    if data.get('synthetic') and config['scope'] != 'engineering':
        raise LearningError('Synthetic data is engineering-only')
    if data.get('synthetic'):
        _integer(data.get('examples', 4), 'synthetic examples')
        _integer(data.get('sequence_length', 8), 'synthetic sequence_length')
        if data.get('sequence_length', 8) > config['model']['max_sequence_length']:
            raise LearningError('Synthetic sequence exceeds model context')
    else:
        data.setdefault('split', 'train')
        if not isinstance(data['split'], str) or not data['split'].strip():
            raise LearningError('data.split must identify a nonempty feature-cache split')
    config.setdefault('budget', {})
    canonical(config)
    return config


def capabilities():
    return {'implementation': IMPLEMENTATION, 'shared_body_expert_training': True, 'dense_teacher_kl': True, 'native_official_eagle3_retraining': False, 'native_recipe_status': 'not_implemented; no claim of reproducing the native official EAGLE-3 architecture or training recipe', 'sparse_teacher_kl': False, 'benchmark_scores_produced': False}


def build_model(value):
    config = model_config(value)
    torch = _torch()
    nn = torch.nn

    class SharedBody(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.TransformerEncoderLayer(config['hidden_size'], config['num_attention_heads'], dim_feedforward=config['hidden_size'] * 4, dropout=config['dropout'], activation='gelu', batch_first=True, norm_first=True) for _ in range(config['num_hidden_layers'])])
            self.forward_calls = 0

        def forward(self, hidden, attention_mask):
            self.forward_calls += 1
            batch, length = attention_mask.shape
            future = torch.triu(torch.ones(length, length, dtype=torch.bool, device=hidden.device), diagonal=1)
            forbidden = future.unsqueeze(0) | ~attention_mask.unsqueeze(1)
            self_only = ~torch.eye(length, dtype=torch.bool, device=hidden.device).unsqueeze(0).expand(batch, -1, -1)
            forbidden = torch.where((~attention_mask).unsqueeze(-1), self_only, forbidden)
            mask = forbidden.repeat_interleave(config['num_attention_heads'], dim=0)
            for layer in self.layers:
                hidden = layer(hidden, src_mask=mask)
                hidden = hidden.masked_fill((~attention_mask).unsqueeze(-1), 0)
            return hidden

    class SharedBodyExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = copy.deepcopy(config)
            self.implementation = IMPLEMENTATION
            self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'], padding_idx=config['pad_token_id'])
            self.feature_projection = nn.Linear(config['feature_dim'], config['hidden_size'])
            self.position_embedding = nn.Embedding(config['max_sequence_length'], config['hidden_size'])
            self.input_norm = nn.LayerNorm(config['hidden_size'])
            self.body = SharedBody()
            self.final_norm = nn.LayerNorm(config['hidden_size'])
            self.gate = nn.Linear(config['hidden_size'], config['num_experts'])
            self.feature_head = nn.Linear(config['hidden_size'], config['feature_dim'])
            if config['expert_mode'] == 'separate_heads':
                self.expert_heads = nn.ModuleList([nn.Linear(config['hidden_size'], config['vocab_size']) for _ in range(config['num_experts'])])
            else:
                self.base_head = nn.Linear(config['hidden_size'], config['vocab_size'])
                self.expert_A = nn.Parameter(torch.empty(config['num_experts'], config['hidden_size'], config['lora_rank']))
                self.expert_B = nn.Parameter(torch.zeros(config['num_experts'], config['lora_rank'], config['vocab_size']))
                nn.init.normal_(self.expert_A, std=0.02)

        def forward(self, input_ids, features, attention_mask=None):
            if input_ids.ndim != 2 or features.shape != (*input_ids.shape, config['feature_dim']):
                raise LearningError('Expected input_ids[B,T] and aligned features[B,T,F]')
            batch, length = input_ids.shape
            if not length or length > config['max_sequence_length']:
                raise LearningError('Sequence length is outside model context')
            if input_ids.dtype not in (torch.int32, torch.int64) or bool((input_ids < 0).any()) or bool((input_ids >= config['vocab_size']).any()):
                raise LearningError('Input token IDs are outside the vocabulary')
            if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
                raise LearningError('Features must be finite floating-point tensors')
            features = features.to(dtype=self.feature_projection.weight.dtype)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask.shape != input_ids.shape or not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise LearningError('attention_mask must be aligned and binary')
            attention_mask = attention_mask.bool()
            positions = torch.arange(length, device=input_ids.device)
            hidden = self.token_embedding(input_ids.long()) + self.feature_projection(features) + self.position_embedding(positions).unsqueeze(0)
            hidden = self.input_norm(hidden).masked_fill((~attention_mask).unsqueeze(-1), 0)
            hidden = self.final_norm(self.body(hidden, attention_mask))
            if config['expert_mode'] == 'separate_heads':
                logits = torch.stack([head(hidden) for head in self.expert_heads], dim=2)
            else:
                low_rank = torch.einsum('bth,khr->btkr', hidden, self.expert_A)
                delta = torch.einsum('btkr,krv->btkv', low_rank, self.expert_B) * (config['lora_alpha'] / config['lora_rank'])
                logits = self.base_head(hidden).unsqueeze(2) + delta
            gate_logits = self.gate(hidden)
            mixture = torch.logsumexp(torch.log_softmax(logits.float(), dim=-1) + torch.log_softmax(gate_logits.float(), dim=-1).unsqueeze(-1), dim=2)
            return {'expert_logits': logits, 'gate_logits': gate_logits, 'mixture_log_probs': mixture, 'normalized_mixture_log_probs': mixture, 'feature_predictions': self.feature_head(hidden), 'shared_hidden': hidden}

    return SharedBodyExperts()


def validate_example(example, model, loss=None, scope='engineering'):
    torch = _torch()
    model = model_config(model)
    loss = loss or {'objective': 'hard_ce', 'feature_mse_weight': 0, 'domain_gate_weight': 0}
    required = ('input_ids', 'features', 'labels', 'loss_mask', 'attention_mask', 'domain_id')
    if not isinstance(example, dict) or any(name not in example for name in required):
        raise LearningError('Example is missing required aligned tensor fields')
    value = dict(example)
    for name in ('feature_targets', 'teacher_logits'):
        if value.get(name) is None:
            value.pop(name, None)
    for name in required[:-1]:
        if not torch.is_tensor(value[name]):
            raise LearningError(name + ' must be a tensor')
    ids = value['input_ids']
    if ids.ndim != 1 or not 1 <= len(ids) <= model['max_sequence_length'] or ids.dtype not in (torch.int32, torch.int64):
        raise LearningError('input_ids must be a nonempty integer sequence within context')
    length = len(ids)
    if value['features'].shape != (length, model['feature_dim']) or value['labels'].shape != (length,):
        raise LearningError('Feature/label alignment differs from the token sequence')
    if value['labels'].dtype not in (torch.int32, torch.int64):
        raise LearningError('labels must be integer IDs already aligned without an internal shift')
    for name in ('attention_mask', 'loss_mask'):
        mask = value[name]
        if mask.shape != (length,) or not bool(((mask == 0) | (mask == 1)).all()):
            raise LearningError(name + ' must be an aligned binary mask')
    attention = value['attention_mask'].bool()
    scoring = value['loss_mask'].bool()
    if not bool(scoring.any()) or bool((scoring & ~attention).any()):
        raise LearningError('Loss mask must score nonpadding tokens and cannot be empty')
    if bool((ids < 0).any()) or bool((ids >= model['vocab_size']).any()):
        raise LearningError('Input IDs exceed the configured vocabulary')
    labels = value['labels']
    if bool(((labels < 0) & (labels != -100)).any()) or bool((labels >= model['vocab_size']).any()) or (loss.get('objective', 'hard_ce') == 'hard_ce' and bool((labels[scoring] < 0).any())):
        raise LearningError('Scored hard labels must be valid aligned vocabulary IDs')
    for name in ('features', 'feature_targets', 'teacher_logits'):
        if name not in value:
            continue
        tensor = value[name]
        expected = (length, model['vocab_size']) if name == 'teacher_logits' else (length, model['feature_dim'])
        if not torch.is_tensor(tensor) or tensor.shape != expected or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise LearningError(name + ' must be a finite aligned floating-point tensor')
    if loss.get('objective') == 'teacher_kl' and 'teacher_logits' not in value:
        raise LearningError('Dense teacher_logits are required for teacher_kl')
    if loss.get('feature_mse_weight', 0):
        mode = loss.get('feature_mse_target', 'next_features')
        if mode == 'input_reconstruction' and scope != 'engineering':
            raise LearningError('Input reconstruction is engineering-only')
        if mode != 'input_reconstruction' and 'feature_targets' not in value:
            raise LearningError('feature_targets are required for next-feature MSE')
    domain = value['domain_id']
    if torch.is_tensor(domain):
        if domain.ndim != 0 or domain.dtype not in (torch.int32, torch.int64):
            raise LearningError('domain_id must be an integer scalar')
        domain = int(domain.item())
    _integer(domain, 'domain_id', 0)
    if loss.get('domain_gate_weight', 0) and domain >= model['num_experts']:
        raise LearningError('Supervised domain_id must identify one configured expert')
    for name in ('input_ids', 'labels'):
        value[name] = value[name].detach().cpu().long().contiguous()
    for name in ('attention_mask', 'loss_mask'):
        value[name] = value[name].detach().cpu().bool().contiguous()
    for name in ('features', 'feature_targets', 'teacher_logits'):
        if name in value:
            value[name] = value[name].detach().cpu().float().contiguous()
    value['domain_id'] = domain
    return value


def collate_examples(examples, model, loss=None, scope='engineering'):
    torch = _torch()
    model = model_config(model)
    values = [validate_example(example, model, loss, scope) for example in examples]
    if not values:
        raise LearningError('Cannot collate an empty batch')
    batch, length = len(values), max(len(value['input_ids']) for value in values)
    result = {'input_ids': torch.full((batch, length), model['pad_token_id'], dtype=torch.long), 'features': torch.zeros(batch, length, model['feature_dim']), 'labels': torch.full((batch, length), -100, dtype=torch.long), 'loss_mask': torch.zeros(batch, length, dtype=torch.bool), 'attention_mask': torch.zeros(batch, length, dtype=torch.bool), 'domain_id': torch.tensor([value['domain_id'] for value in values], dtype=torch.long)}
    for name, dimension in (('teacher_logits', model['vocab_size']), ('feature_targets', model['feature_dim'])):
        if all(name in value for value in values):
            result[name] = torch.zeros(batch, length, dimension)
    for index, value in enumerate(values):
        end = len(value['input_ids'])
        for name in result.keys() - {'domain_id'}:
            result[name][index, :end] = value[name]
    return result


def compute_loss(outputs, batch, value=None):
    torch = _torch()
    functional = torch.nn.functional
    config = {'objective': 'hard_ce', 'temperature': 1.0, 'feature_mse_weight': 0.0, 'feature_mse_target': 'next_features', 'domain_gate_weight': 0.0}
    config.update(value or {})
    mixture = outputs['mixture_log_probs']
    labels = batch['labels']
    if mixture.ndim != 3 or labels.shape != mixture.shape[:2] or batch['loss_mask'].shape != labels.shape or batch['attention_mask'].shape != labels.shape:
        raise LearningError('Loss tensors must be aligned [B,T] and [B,T,V]')
    mask = batch['loss_mask'].bool()
    attention = batch['attention_mask'].bool()
    tokens = int(mask.sum().item())
    if not tokens or bool((mask & ~attention).any()):
        raise LearningError('Masked loss must contain scored nonpadding tokens')
    if config['objective'] == 'hard_ce':
        if bool((labels[mask] < 0).any()) or bool((labels[mask] >= mixture.shape[-1]).any()):
            raise LearningError('Hard labels are outside the vocabulary')
        main = -mixture.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)[mask].mean()
    elif config['objective'] == 'teacher_kl':
        if 'teacher_logits' not in batch or batch['teacher_logits'].shape != mixture.shape:
            raise LearningError('Aligned dense teacher_logits are required')
        temperature = _number(config['temperature'], 'temperature', positive=True)
        teacher_log = torch.log_softmax(batch['teacher_logits'].detach().float() / temperature, dim=-1)
        student_log = torch.logsumexp(torch.log_softmax(outputs['expert_logits'].float() / temperature, dim=-1) + torch.log_softmax(outputs['gate_logits'].float(), dim=-1).unsqueeze(-1), dim=2)
        main = (teacher_log.exp() * (teacher_log - student_log)).sum(-1)[mask].mean() * temperature * temperature
    else:
        raise LearningError('Unsupported token objective')
    feature_loss = main.new_zeros(())
    if config['feature_mse_weight']:
        target = batch['features'] if config['feature_mse_target'] == 'input_reconstruction' else batch.get('feature_targets')
        if target is None or target.shape != outputs['feature_predictions'].shape:
            raise LearningError('Aligned feature_targets are required for next-feature MSE')
        feature_loss = (outputs['feature_predictions'].float() - target.detach().float()).square().mean(-1)[mask].mean()
    gate_loss = main.new_zeros(())
    if config['domain_gate_weight']:
        domains = batch['domain_id'].long()
        if domains.shape != (labels.shape[0],) or bool((domains < 0).any()) or bool((domains >= outputs['gate_logits'].shape[-1]).any()):
            raise LearningError('domain_id must identify one expert per batch example')
        targets = domains[:, None].expand_as(labels)
        gate_loss = functional.cross_entropy(outputs['gate_logits'][mask], targets[mask])
    total = main + config['feature_mse_weight'] * feature_loss + config['domain_gate_weight'] * gate_loss
    if not bool(torch.isfinite(total)):
        raise LearningError('Training loss is nonfinite')
    return {'loss': total, 'token_loss': main, 'feature_mse': feature_loss, 'gate_loss': gate_loss, 'loss_tokens': tokens, 'nonpadding_tokens': int(attention.sum().item()), 'input_slots': labels.numel()}


def budget_eligibility(observed, declared):
    reasons = []
    for name in ('loss_tokens', 'optimizer_steps'):
        if name not in declared:
            reasons.append('missing_declared_' + name)
        elif type(declared[name]) is not int or declared[name] < 1 or observed.get(name) != declared[name]:
            reasons.append(name + '_does_not_match_declared_budget')
    for name in ('nonpadding_tokens', 'input_slots'):
        if name in declared and observed.get(name) != declared[name]:
            reasons.append(name + '_does_not_match_declared_budget')
    return {'eligible': not reasons, 'reasons': reasons, 'observed': dict(observed), 'declared': dict(declared), 'qualification': 'Eligibility matches declared loss-token and optimizer-step budgets; it does not establish equal FLOPs, GPU time or native-recipe comparability.'}


def compare_budgets(left, right):
    a = budget_eligibility(left['counters'], left['declared_budget'])
    b = budget_eligibility(right['counters'], right['declared_budget'])
    same = all(name in left['declared_budget'] and name in right['declared_budget'] and left['declared_budget'][name] == right['declared_budget'][name] for name in ('loss_tokens', 'optimizer_steps'))
    return {'eligible': a['eligible'] and b['eligible'] and same, 'left': a, 'right': b, 'same_declared_token_step_budget': same}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('wb') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def trained_directory(root, config):
    root = Path(root).resolve()
    for name in ('experiment', 'run_id'):
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', config[name]):
            raise LearningError('Invalid trained-model path identifier')
    storage = (root / 'models').resolve()
    directory = storage / 'trained' / config['experiment'] / config['run_id']
    if directory.resolve() != directory:
        raise LearningError('Trained output children cannot use symlink aliases or escape their resolved model storage root')
    return directory


def _synthetic(torch, config):
    model = config['model']
    data = config['data']
    generator = torch.Generator(device='cpu').manual_seed(config['seed'] + 1)
    result = []
    for index in range(data.get('examples', 4)):
        length = data.get('sequence_length', 8)
        ids = torch.randint(0, model['vocab_size'], (length + 1,), generator=generator)
        features = torch.randn(length + 1, model['feature_dim'], generator=generator)
        value = {'input_ids': ids[:-1], 'features': features[:-1], 'feature_targets': features[1:], 'labels': ids[1:], 'attention_mask': torch.ones(length, dtype=torch.bool), 'loss_mask': torch.ones(length, dtype=torch.bool), 'domain_id': index % model['num_experts']}
        if data.get('include_teacher_logits', True):
            value['teacher_logits'] = torch.randn(length, model['vocab_size'], generator=generator)
        result.append(validate_example(value, model, config['loss'], config['scope']))
    encoded = [{name: {'dtype': str(tensor.dtype), 'shape': list(tensor.shape), 'values': tensor.tolist()} if torch.is_tensor(tensor) else tensor for name, tensor in value.items()} for value in result]
    return result, {'sha256': identity(encoded), 'synthetic': True, 'scope': 'engineering', 'alignment': 'Synthetic prealigned one-step token/feature targets; not native teacher features'}


def inspect_feature_manifest(root, path, split='train'):
    path = Path(path)
    if not path.is_absolute():
        path = Path(root) / path
    if path.is_dir():
        path = path / 'manifest.json'
    path = path.resolve(strict=True)
    expected = file_sha256(path)
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict) or metadata.get('status') != 'complete' or metadata.get('command') != 'features' or not metadata.get('alignment'):
        raise LearningError('Only completed aligned feature-export manifests can be trained')
    entries = metadata.get('examples')
    if not isinstance(entries, list) or not entries:
        raise LearningError('Feature manifest must contain completed example records')
    if not isinstance(split, str) or not split.strip():
        raise LearningError('Feature-cache split must be a nonempty string')
    selected, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('path'), str) or not isinstance(entry.get('split'), str) or not isinstance(entry.get('sha256'), str) or not re.fullmatch('[a-f0-9]{64}', entry['sha256']):
            raise LearningError('Feature manifest example lacks path, split or SHA256')
        example = Path(entry['path'])
        if not example.is_absolute():
            example = path.parent / example
        resolved = example.resolve(strict=True)
        if not resolved.is_relative_to(path.parent) or resolved.suffix != '.pt' or not resolved.is_file() or resolved in seen:
            raise LearningError('Feature manifest paths must be unique tensor files inside their export directory')
        seen.add(resolved)
        if file_sha256(resolved) != entry['sha256']:
            raise LearningError('Feature example SHA256 differs from its completed export manifest')
        if entry['split'] == split:
            selected.append({**entry, 'path': str(resolved)})
    if not selected:
        raise LearningError('Completed feature export contains no examples for split ' + split)
    if file_sha256(path) != expected:
        raise LearningError('Feature manifest changed while being verified')
    details = {key: value for key, value in metadata.items() if key != 'examples'}
    canonical(details)
    return {'path': str(path), 'sha256': expected, 'split': split, 'metadata': details, 'examples': selected, 'exported_examples': len(entries)}


def load_examples(root, config):
    if config['data'].get('synthetic'):
        return _synthetic(_torch(), config)
    path = Path(config['data']['path'])
    if not path.is_absolute():
        path = Path(root) / path
    if path.is_dir() or path.suffix == '.json':
        manifest = inspect_feature_manifest(root, path, config['data'].get('split', 'train'))
        torch = _torch()
        values = []
        for entry in manifest['examples']:
            raw = torch.load(entry['path'], map_location='cpu', weights_only=True)
            if file_sha256(entry['path']) != entry['sha256']:
                raise LearningError('Feature example changed while being loaded')
            value = validate_example(raw, config['model'], config['loss'], config['scope'])
            if ('tokens' in entry and entry['tokens'] != len(value['input_ids'])) or ('loss_tokens' in entry and entry['loss_tokens'] != int(value['loss_mask'].sum().item())):
                raise LearningError('Feature tensor token counts differ from the completed export manifest')
            values.append(value)
        if file_sha256(manifest['path']) != manifest['sha256']:
            raise LearningError('Feature manifest changed while examples were loaded')
        return values, {**manifest, 'examples': len(values), 'example_files': manifest['examples'], 'synthetic': False}
    torch = _torch()
    expected = file_sha256(path)
    data = torch.load(path, map_location='cpu', weights_only=True)
    if file_sha256(path) != expected:
        raise LearningError('Dataset changed while being loaded')
    if isinstance(data, dict) and 'examples' in data:
        metadata = data.get('metadata', {})
        data = data['examples']
    else:
        metadata = {}
        data = [data]
    if not isinstance(data, (list, tuple)) or not data:
        raise LearningError('Dataset must contain nonempty example records')
    if not isinstance(metadata, dict):
        raise LearningError('Dataset metadata must be a dictionary')
    canonical(metadata)
    if config['scope'] != 'engineering' and not metadata.get('alignment'):
        raise LearningError('Research feature caches must record their alignment metadata')
    values = [validate_example(value, config['model'], config['loss'], config['scope']) for value in data]
    return values, {'path': str(path.resolve()), 'sha256': expected, 'synthetic': False, 'metadata': metadata, 'examples': len(values)}


def _counters_valid(counters, config):
    for name in ('optimizer_steps', 'micro_batches', 'loss_tokens', 'nonpadding_tokens', 'input_slots', 'examples_consumed'):
        _integer(counters.get(name), name, 0)
    if counters['optimizer_steps'] > config['training']['optimizer_steps'] or counters['micro_batches'] != counters['optimizer_steps'] * config['training']['gradient_accumulation_steps'] or counters['examples_consumed'] != counters['micro_batches'] * config['training']['batch_size'] or not counters['loss_tokens'] <= counters['nonpadding_tokens'] <= counters['input_slots']:
        raise LearningError('Checkpoint counters violate the declared training schedule')


def verify_checkpoint(path, run_dir, expected):
    path, run_dir = Path(path), Path(run_dir).resolve()
    if path.is_symlink() or path.with_suffix('.json').is_symlink():
        raise LearningError('Checkpoint or committed receipt cannot be a symlink')
    path = path.resolve()
    if path.parent != run_dir or path.suffix != '.pt':
        raise LearningError('Resume checkpoint must be inside this run trained-model directory')
    receipt = json.loads(path.with_suffix('.json').read_text())
    if receipt.get('status') != 'committed' or file_sha256(path) != receipt.get('checkpoint_sha256'):
        raise LearningError('Checkpoint file SHA256 or commit marker differs')
    metadata = receipt['metadata']
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise LearningError('Immutable checkpoint identity differs: ' + name)
    return receipt


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def recover_uncommitted_checkpoints(directory, expected, lock):
    directory = Path(directory).resolve()
    lock_path = directory / 'training.lock'
    actual, required = os.fstat(lock.fileno()), lock_path.stat(follow_symlinks=False)
    if (actual.st_dev, actual.st_ino) != (required.st_dev, required.st_ino) or not stat.S_ISREG(required.st_mode) or required.st_uid != os.getuid():
        raise LearningError('Checkpoint recovery requires this run training lock')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = directory / 'run.json'
    if manifest_path.is_symlink() or json.loads(manifest_path.read_text()).get('identity') != expected:
        raise LearningError('Checkpoint recovery requires an existing matching run identity')
    recovered = []
    for path in sorted(directory.glob('checkpoint-step-*.pt')):
        if not re.fullmatch('checkpoint-step-[0-9]{6,}\\.pt', path.name):
            continue
        receipt = path.with_suffix('.json')
        if receipt.exists() or receipt.is_symlink():
            continue
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
            raise LearningError('Uncommitted checkpoint is not an owned regular run file')
        checksum = file_sha256(path)
        recovery_root = directory / 'recovery'
        recovery_root.mkdir(exist_ok=True)
        if recovery_root.resolve() != recovery_root:
            raise LearningError('Checkpoint recovery cannot use a symlink directory')
        recovery = Path(tempfile.mkdtemp(prefix='uncommitted-checkpoint-', dir=recovery_root))
        destination = recovery / path.name
        record = {'status': 'planned', 'reason': 'checkpoint_file_without_committed_receipt', 'original_path': str(path), 'quarantined_path': str(destination), 'sha256': checksum, 'bytes': before.st_size, 'identity': expected}
        _atomic_json(recovery / 'receipt.json', record)
        _fsync_directory(recovery_root)
        after = path.stat(follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or receipt.exists() or receipt.is_symlink():
            raise LearningError('Uncommitted checkpoint changed before quarantine')
        path.rename(destination)
        _fsync_directory(directory)
        _fsync_directory(recovery)
        if file_sha256(destination) != checksum:
            raise LearningError('Quarantined checkpoint differs from its recorded SHA256')
        record['status'] = 'quarantined'
        _atomic_json(recovery / 'receipt.json', record)
        recovered.append(record)
    return recovered


def save_checkpoint(model, optimizer, directory, metadata):
    torch = _torch()
    step = metadata['counters']['optimizer_steps']
    path = directory / ('checkpoint-step-' + str(step).zfill(6) + '.pt')
    if path.exists() or path.is_symlink() or path.with_suffix('.json').exists() or path.with_suffix('.json').is_symlink():
        raise LearningError('Refusing to overwrite an existing checkpoint or input weights')
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'metadata': metadata, 'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.checkpoint-', suffix='.pt', dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        checksum = file_sha256(temporary)
        os.replace(temporary, path)
        _fsync_directory(directory)
        receipt = {'status': 'committed', 'checkpoint': str(path), 'checkpoint_sha256': checksum, 'metadata': metadata}
        _atomic_json(path.with_suffix('.json'), receipt)
        _atomic_json(directory / 'latest.json', {'checkpoint': str(path), 'checkpoint_sha256': checksum, 'optimizer_steps': step})
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _device(value, allow_cpu):
    selected = value or 'cuda:0'
    if selected == 'cpu' and allow_cpu:
        return selected
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not os.environ.get('SLURM_JOB_ID') or visible is None or len(visible.split(',')) != 1 or not visible.strip() or selected != 'cuda:0':
        raise LearningError('Actual training requires Slurm and exactly one visible GPU; CPU is only an injected unit-test option')
    return selected


def _history(path, step):
    path = Path(path)
    if not path.exists():
        return
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise LearningError('Training history recovery requires an owned regular run log')
    path = path.resolve()
    contents = path.read_bytes()
    boundary = contents.rfind(b'\n') + 1
    prefix, tail = contents[:boundary], contents[boundary:]
    kept, removed, previous = [], False, 0
    for raw in prefix.splitlines(keepends=True):
        if not raw.strip():
            if not removed:
                kept.append(raw)
            continue
        try:
            row = json.loads(raw)
            row_step = row['counters']['optimizer_steps']
            if type(row_step) is not int or row_step <= previous:
                raise ValueError('nonincreasing optimizer step')
        except (ValueError, TypeError, KeyError) as error:
            raise LearningError('Malformed complete training-history row; original bytes preserved') from error
        previous = row_step
        if row_step <= step:
            kept.append(raw)
        else:
            removed = True
    if tail or removed:
        recovery_root = path.parent / 'recovery'
        recovery_root.mkdir(exist_ok=True)
        if recovery_root.resolve() != recovery_root:
            raise LearningError('History recovery cannot use a symlink directory')
        recovery = Path(tempfile.mkdtemp(prefix='training-history-', dir=recovery_root))
        for name, data in (('original.jsonl', contents), ('trailing-tail.bin', tail)):
            with (recovery / name).open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        _atomic_json(recovery / 'receipt.json', {'status': 'preserved', 'original_path': str(path), 'original_sha256': hashlib.sha256(contents).hexdigest(), 'trailing_tail_sha256': hashlib.sha256(tail).hexdigest(), 'trailing_tail_bytes': len(tail), 'committed_optimizer_steps': step, 'removed_complete_uncommitted_rows': removed})
        _fsync_directory(recovery_root)
        temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
        with temporary.open('wb') as stream:
            stream.write(b''.join(kept))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return str(recovery)


def train(value, root, *, device=None, allow_cpu=False, resume=None, stop_after_steps=None):
    config = normalize_config(value)
    selected_device = _device(device, allow_cpu)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch = _torch()
    if selected_device.startswith('cuda') and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise LearningError('The worker must see exactly one usable CUDA GPU')
    root = Path(root).resolve()
    directory = trained_directory(root, config)
    directory.mkdir(parents=True, exist_ok=True)
    output = root / 'outputs/learning' / config['experiment'] / config['run_id']
    output.mkdir(parents=True, exist_ok=True)
    config_hash = identity(config)
    source_hash = file_sha256(__file__)
    examples, data = load_examples(root, config)
    data_hash = identity(data)
    frozen = {'implementation': IMPLEMENTATION, 'scope': config['scope'], 'experiment': config['experiment'], 'run_id': config['run_id'], 'config_sha256': config_hash, 'data_identity_sha256': data_hash, 'source_sha256': source_hash}
    with (directory / 'training.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LearningError('Another trainer holds this run lock') from error
        manifest = directory / 'run.json'
        if manifest.exists():
            if manifest.is_symlink() or manifest.stat().st_uid != os.getuid() or json.loads(manifest.read_text())['identity'] != frozen:
                raise LearningError('Existing training run has incompatible immutable identities')
        else:
            if any(directory.glob('checkpoint-step-*')) or manifest.is_symlink():
                raise LearningError('Existing checkpoint inputs without a matching run identity will not be changed')
            _atomic_json(manifest, {'identity': frozen, 'config': config, 'data': data, 'qualifier': 'Shared-body expert prototype, not the native official EAGLE-3 recipe'})
        checkpoints = sorted(directory.glob('checkpoint-step-*.json'))
        if (checkpoints or any(directory.glob('checkpoint-step-*.pt'))) and resume is None:
            raise LearningError('Existing checkpoints require explicit resume; input weights will not be overwritten')
        checkpoint_recoveries = recover_uncommitted_checkpoints(directory, frozen, lock) if resume is not None else []
        if str(resume) == 'latest' and not checkpoints and not checkpoint_recoveries:
            raise LearningError('No committed checkpoint or recoverable interrupted first checkpoint is available to resume')
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.use_deterministic_algorithms(True)
        if selected_device.startswith('cuda'):
            torch.cuda.manual_seed_all(config['seed'])
            torch.backends.cudnn.benchmark = False
        model = build_model(config['model']).to(selected_device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'], foreach=False)
        counters = dict.fromkeys(('optimizer_steps', 'micro_batches', 'loss_tokens', 'nonpadding_tokens', 'input_slots', 'examples_consumed'), 0)
        checkpoint = None
        if resume is not None and (checkpoints or str(resume) != 'latest'):
            if str(resume) == 'latest':
                if not checkpoints:
                    raise LearningError('No committed checkpoint is available to resume')
                path = checkpoints[-1].with_suffix('.pt')
            else:
                path = Path(resume)
                if not path.is_absolute():
                    path = root / path
                if checkpoints and path.resolve() != checkpoints[-1].with_suffix('.pt').resolve():
                    raise LearningError('Rewinding an existing run requires a new run identity')
            checkpoint = verify_checkpoint(path, directory, frozen)
            packed = torch.load(path, map_location='cpu', weights_only=True)
            if packed['metadata'] != checkpoint['metadata']:
                raise LearningError('Checkpoint metadata and committed receipt differ')
            counters = dict(packed['metadata']['counters'])
            _counters_valid(counters, config)
            if packed['metadata']['data_position'] != counters['examples_consumed']:
                raise LearningError('Checkpoint data cursor differs from counters')
            model.load_state_dict(packed['model'], strict=True)
            optimizer.load_state_dict(packed['optimizer'])
            torch.set_rng_state(packed['torch_rng_state'])
            if selected_device.startswith('cuda') and packed['cuda_rng_state']:
                torch.cuda.set_rng_state_all(packed['cuda_rng_state'])
        history = output / 'training.jsonl'
        history_recovery = _history(history, counters['optimizer_steps'])
        ceiling = config['training']['optimizer_steps']
        if stop_after_steps is not None:
            _integer(stop_after_steps, 'stop_after_steps', 0)
            ceiling = min(ceiling, stop_after_steps)
        batch_size = config['training']['batch_size']
        accumulation = config['training']['gradient_accumulation_steps']
        position = counters['examples_consumed']
        orders = {}

        def next_batch():
            nonlocal position
            selected = []
            for offset in range(batch_size):
                epoch, cursor = divmod(position + offset, len(examples))
                if epoch not in orders:
                    order = list(range(len(examples)))
                    if config['training']['shuffle']:
                        random.Random(config['seed'] + epoch).shuffle(order)
                    orders.clear()
                    orders[epoch] = order
                selected.append(examples[orders[epoch][cursor]])
            position += batch_size
            return collate_examples(selected, config['model'], config['loss'], config['scope'])

        try:
            while counters['optimizer_steps'] < ceiling:
                micro_batches = [next_batch() for _ in range(accumulation)]
                step_tokens = sum(int(batch['loss_mask'].sum().item()) for batch in micro_batches)
                optimizer.zero_grad(set_to_none=True)
                losses = dict.fromkeys(('loss', 'token_loss', 'feature_mse', 'gate_loss'), 0.0)
                for cpu_batch in micro_batches:
                    batch = {name: tensor.to(selected_device) for name, tensor in cpu_batch.items()}
                    losses_now = compute_loss(model(batch['input_ids'], batch['features'], batch['attention_mask']), batch, config['loss'])
                    weight = losses_now['loss_tokens'] / step_tokens
                    (losses_now['loss'] * weight).backward()
                    for name in losses:
                        losses[name] += float(losses_now[name].detach().item()) * weight
                    for name in ('loss_tokens', 'nonpadding_tokens', 'input_slots'):
                        counters[name] += losses_now[name]
                    counters['micro_batches'] += 1
                    counters['examples_consumed'] += batch_size
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                counters['optimizer_steps'] += 1
                _counters_valid(counters, config)
                row = {'implementation': IMPLEMENTATION, 'scope': config['scope'], 'losses': losses, 'gradient_norm': float(norm.item()), 'counters': dict(counters), 'config_sha256': config_hash, 'data_identity_sha256': data_hash}
                with history.open('ab') as stream:
                    stream.write(canonical(row) + b'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                if counters['optimizer_steps'] % config['training']['checkpoint_every_steps'] == 0 or counters['optimizer_steps'] == ceiling:
                    metadata = {**frozen, 'config': config, 'data': data, 'counters': dict(counters), 'data_position': position, 'torch_version': str(torch.__version__), 'python_version': platform.python_version(), 'label_alignment': 'Already aligned; no internal token or feature shift', 'last_step_metrics': row}
                    checkpoint = save_checkpoint(model, optimizer, directory, metadata)
            status = 'complete' if counters['optimizer_steps'] == config['training']['optimizer_steps'] else 'partial'
            result = {'status': status, **frozen, 'qualifier': 'Reference shared-body expert prototype, not native official EAGLE-3 retraining or a scientific benchmark result', 'capabilities': capabilities(), 'device': selected_device, 'data': data, 'counters': counters, 'declared_budget': config['budget'], 'budget_eligibility': budget_eligibility(counters, config['budget']), 'checkpoint': None if checkpoint is None else checkpoint['checkpoint'], 'checkpoint_sha256': None if checkpoint is None else checkpoint['checkpoint_sha256'], 'checkpoint_recoveries': checkpoint_recoveries, 'history_recovery': history_recovery, 'body_forward_calls_this_attempt': model.body.forward_calls, 'parameter_count': sum(parameter.numel() for parameter in model.parameters()), 'benchmark_scores_produced': False}
            _atomic_json(output / 'report.json', result)
            return result
        except BaseException as error:
            _atomic_json(output / 'failure.json', {**frozen, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error), 'counters': counters, 'latest_committed_checkpoint': None if checkpoint is None else checkpoint['checkpoint']})
            raise


run_training = train


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/engineering-training.json'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', nargs='?', const='latest')
    parser.add_argument('--capabilities', action='store_true')
    args = parser.parse_args(argv)
    if args.capabilities:
        print(json.dumps(capabilities(), sort_keys=True), flush=True)
        return
    path = args.config if args.config.is_absolute() else args.root / args.config
    config = json.loads(path.read_text())
    if args.smoke:
        config = copy.deepcopy(config)
        config['scope'] = 'engineering'
        config['run_id'] += '-smoke'
        config['data'] = {'synthetic': True, 'examples': 4, 'sequence_length': min(8, config['model'].get('max_sequence_length', 128)), 'include_teacher_logits': True}
        config.setdefault('training', {}).update(optimizer_steps=2, batch_size=2, gradient_accumulation_steps=1, checkpoint_every_steps=1)
        config['budget'] = {'optimizer_steps': 2, 'loss_tokens': 2 * 2 * config['data']['sequence_length']}
    try:
        result = train(config, args.root, resume=args.resume)
    except (OSError, ValueError) as error:
        print(json.dumps({'status': 'failed', 'implementation': IMPLEMENTATION, 'error_type': type(error).__name__, 'error': str(error)}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({'status': result['status'], 'implementation': IMPLEMENTATION, 'scope': result['scope'], 'counters': result['counters'], 'budget_eligible': result['budget_eligibility']['eligible'], 'checkpoint': result['checkpoint']}), flush=True)


if __name__ == '__main__':
    main()
