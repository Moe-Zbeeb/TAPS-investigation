import functools
import importlib
from pathlib import Path

from . import pool
from .pilot_cache import require


def checkpoint_adapter(original, mode):
    require(mode in ('non_reentrant', 'off_reference'), 'Unknown corrected checkpoint policy')

    @functools.wraps(original)
    def checkpoint(function, *args, **options):
        require(options.get('use_reentrant') in (None, False), 'Reentrant checkpointing is inadmissible for nested native outputs')
        options.pop('use_reentrant', None)
        if mode == 'off_reference':
            require(not options, 'Unexpected native checkpoint reference options')
            return function(*args)
        return original(function, *args, use_reentrant=False, **options)

    checkpoint._taps_checkpoint_policy = mode
    return checkpoint


def install_checkpoint_policy(mode):
    module = importlib.import_module('torch.utils.checkpoint')
    original = module.checkpoint
    require(not hasattr(original, '_taps_checkpoint_policy'), 'Checkpoint policy was already installed')
    source = Path(module.__file__).resolve()
    module.checkpoint = checkpoint_adapter(original, mode)
    return {'mode': mode, 'torch_checkpoint_path': str(source), 'torch_checkpoint_sha256': pool.digest(source), 'adapter_path': str(Path(__file__).resolve()), 'adapter_sha256': pool.digest(__file__), 'nested_tensor_outputs_supported': True}


def prime_rotary_cache(model, torch, length=2048, heads=7):
    rotary = model.midlayer.self_attn.rotary_emb
    require(rotary.inv_freq.device.type == 'cpu' and rotary.inv_freq.dtype == torch.float32, 'Prime the native rotary cache before the FP16/device conversion')
    require(model.length == heads and length == 2048, 'Corrected native rotary geometry changed')
    previous = int(rotary.max_seq_len_cached)
    before = [rotary.cos_cached.clone(), rotary.sin_cached.clone()]
    capacity = max(previous, length + heads - 1)
    rotary._set_cos_sin_cache(seq_len=capacity, device=rotary.inv_freq.device, dtype=torch.float32)
    require(all(torch.equal(old, current[..., :previous, :]) for old, current in zip(before, (rotary.cos_cached, rotary.sin_cached))), 'Rotary priming changed the existing FP32 prefix')
    def forbid_extension(*args, **options):
        raise ValueError('The qualified native rotary cache cannot be extended or rebuilt')
    rotary._set_cos_sin_cache = forbid_extension
    return {'previous_capacity': previous, 'primed_capacity': capacity, 'construction_device': 'cpu', 'construction_dtype': 'torch.float32', 'existing_prefix_unchanged': True, 'frozen_after_priming': True, 'purpose': 'avoid stateful cache extension during checkpoint recomputation'}


class ParameterCoverage:
    def __init__(self, model, torch):
        self.torch = torch
        self.expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        require(len(self.expected) == 13 and 'fc.weight' in self.expected and 'lm_head.weight' in self.expected and 'norm.weight' in self.expected and sum(name.startswith('midlayer.') for name in self.expected) == 10, 'The intended thirteen-trainable native architecture changed')
        self.values = {}
        self.handles = []
        for name, parameter in model.named_parameters():
            if name in self.expected:
                self.handles.append(parameter.register_hook(self.observer(name)))

    def observer(self, name):
        def observe(gradient):
            finite = bool(self.torch.isfinite(gradient).all())
            count = int(self.torch.count_nonzero(gradient))
            value = self.values.setdefault(name, {'callbacks': 0, 'finite': True, 'nonzero': False, 'maximum_callback_nonzero_elements': 0, 'sum_finite_callback_l2': 0.0})
            value['callbacks'] += 1
            value['finite'] = value['finite'] and finite
            value['nonzero'] = value['nonzero'] or count > 0
            value['maximum_callback_nonzero_elements'] = max(value['maximum_callback_nonzero_elements'], count)
            if finite:
                value['sum_finite_callback_l2'] += float(gradient.float().norm())
            return gradient
        return observe

    def begin(self):
        self.values = {}

    def finish(self):
        require(set(self.values) == self.expected, 'At least one intended native trainable received no backward callback')
        require(all(value['finite'] and value['nonzero'] for value in self.values.values()), 'At least one intended native trainable has nonfinite or entirely zero callback gradients')
        return {name: dict(value) for name, value in sorted(self.values.items())}

    def close(self):
        for handle in self.handles:
            handle.remove()
