import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import typing
import warnings


SOURCE_SHA256 = '71295b56d8bd5648a1c630af0995f6a23a13671c8e60fe92636fffd9ee786a9c'


def require(value, code):
    if not value:
        raise ValueError(code)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_identity(model, torch):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        result.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'nested_gradient_probe_requires_cpu_slurm')
    require(not args.output.exists() and digest(args.source) == SOURCE_SHA256, 'fresh_output_and_pinned_source_required')
    import torch
    import deepspeed
    import torch.utils.checkpoint as checkpoint

    torch.set_num_threads(1)
    tree = ast.parse(args.source.read_bytes())
    decoder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LlamaDecoderLayeremb')
    forward = next(node for node in decoder.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    namespace = {'torch': torch, 'Optional': typing.Optional, 'Tuple': typing.Tuple, 'List': typing.List}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(args.source), 'exec'), namespace)
    actual_forward = namespace['forward']
    model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Model')
    checkpoint_calls = [node for node in ast.walk(model_class) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'checkpoint']
    require(len(checkpoint_calls) == 1 and not any(keyword.arg == 'use_reentrant' for keyword in checkpoint_calls[0].keywords), 'native_checkpoint_mode_was_not_unspecified')

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(8, 4, bias=False)

        def forward(self, hidden_states, cache_hidden, **options):
            value = self.projection(hidden_states)
            return value, [[value], [value]]

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_norm = torch.nn.LayerNorm(4)
            self.input_layernorm = torch.nn.LayerNorm(4)
            self.self_attn = Attention()
            self.post_attention_layernorm = torch.nn.LayerNorm(4)
            self.mlp = torch.nn.Linear(4, 4, bias=False)

        forward = actual_forward

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(12, 4, bias=False)
            self.midlayer = Decoder()
            self.norm = torch.nn.LayerNorm(4)
            self.lm_head = torch.nn.Linear(4, 7, bias=False)

    results = []
    for mode in ('off', 'default_unspecified', 'explicit_reentrant', 'explicit_nonreentrant'):
        torch.manual_seed(20261112)
        model = Model().train()
        features = torch.randn(1, 13, 12)
        hidden = model.fc(features)
        embedding = torch.randn(1, 13, 4, requires_grad=True)
        attention, positions = torch.ones((1, 13)), torch.arange(13)[None]
        cache = [[], []]
        initial = model_identity(model, torch)
        captured = []
        loss = 0
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            for head in range(7):
                def run(*inputs):
                    return model.midlayer(*inputs, None, False)
                inputs = embedding, hidden, cache, attention, positions
                if mode == 'off':
                    outputs, cache = run(*inputs)
                elif mode == 'default_unspecified':
                    outputs, cache = checkpoint.checkpoint(run, *inputs)
                else:
                    outputs, cache = checkpoint.checkpoint(run, *inputs, use_reentrant=mode == 'explicit_reentrant')
                hidden = outputs[0]
                captured.append({'head': head, 'hidden_requires_grad': hidden.requires_grad, 'hidden_grad_fn': type(hidden.grad_fn).__name__ if hidden.grad_fn else None})
                logits = model.lm_head(model.norm(hidden))
                labels = torch.full((13,), head % 7, dtype=torch.long)
                loss = loss + (0.8 ** head) * torch.nn.functional.cross_entropy(logits.reshape(-1, 7), labels)
            loss.backward()
        gradients = {name: {'gradient_is_none': parameter.grad is None, 'gradient_norm': None if parameter.grad is None else float(parameter.grad.double().norm()), 'gradient_nonzero_elements': None if parameter.grad is None else int(torch.count_nonzero(parameter.grad))} for name, parameter in model.named_parameters()}
        results.append({'mode': mode, 'initial_model_sha256': initial, 'loss': float(loss.detach()), 'nested_outputs': captured, 'gradients': gradients, 'warnings': [{'category': type(item.message).__name__, 'message': str(item.message)[:500]} for item in observed]})
    require(len({row['initial_model_sha256'] for row in results}) == 1, 'probe_modes_have_different_initial_weights')
    by_mode = {row['mode']: row for row in results}
    cut = {mode: all(value['gradient_is_none'] for name, value in by_mode[mode]['gradients'].items() if name.startswith(('fc.', 'midlayer.'))) and all(not value['gradient_is_none'] for name, value in by_mode[mode]['gradients'].items() if name.startswith(('norm.', 'lm_head.'))) for mode in ('default_unspecified', 'explicit_reentrant')}
    connected = {mode: all(not value['gradient_is_none'] for value in by_mode[mode]['gradients'].values()) for mode in ('off', 'explicit_nonreentrant')}
    reproduced = all(cut.values()) and all(connected.values())
    checkpoint_path = Path(inspect.getfile(checkpoint))
    report = {'status': 'reproduced_nested_reentrant_gradient_cut' if reproduced else 'nested_cut_hypothesis_not_reproduced', 'scope': 'CPU mechanism diagnostic using exact pinned decoder forward AST with tiny substitute modules, not an executed full8B training gradient test', 'torch_version': torch.__version__, 'deepspeed_version': deepspeed.__version__, 'decoder_source_sha256': SOURCE_SHA256, 'decoder_forward_ast_sha256': hashlib.sha256(ast.dump(forward).encode()).hexdigest(), 'torch_checkpoint_source': {'path': str(checkpoint_path), 'sha256': digest(checkpoint_path)}, 'source_checkpoint_use_reentrant_keyword_absent': True, 'observed_cut_by_mode': cut, 'observed_full_connectivity_by_mode': connected, 'results': results, 'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'probe_source_sha256': digest(Path(__file__)), 'gpu_work_performed': False, 'scientific_training_admitted': False, 'jobs_or_active_sources_modified': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': digest(args.output), 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
