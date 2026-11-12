import argparse
import json
import os
from pathlib import Path
import sys
import time

from . import native_full_body as corrected, native_training as native, pool
from .pilot_cache import read, require


def prepare(plan_path, output):
    native.require_slurm()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Native export inventory requires CPU Slurm')
    corrected.install()
    plan = native.load_plan(plan_path)
    directory = Path(plan['trained_models']['resolved_path']) / 'budget_step_4'
    require(plan['config']['pipeline_stage'] == 'corrected-qualification-v3' and plan['config']['budget'] == {'optimizer_steps': 4} and read(Path(plan['run_dir']) / 'launch.json')['status'] == 'complete', 'Native export requires a completed corrected four-update run')
    import torch
    state = torch.load(directory / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    require(not any(name.startswith(('target_model.', 'embed_tokens.')) for name in state), 'Native export contains frozen verifier/embedding weights')
    cache = torch.load(Path(plan['workdir']) / 'cache.pt', map_location='cpu', weights_only=False)
    native.validate_vocabulary(torch, state['d2t'], state['t2d'], 128256, 32000)
    require(torch.equal(state['d2t'], cache['d2t']) and torch.equal(state['t2d'], cache['t2d']), 'Native export vocabulary differs from the train-only cache')
    source = {name: value for name, value in state.items() if name not in ('d2t', 't2d')}
    require(len(source) == 13 and 'fc.weight' in source and 'lm_head.weight' in source and 'norm.weight' in source and all(bool(torch.isfinite(value).all()) for value in source.values()), 'Native export lacks the complete finite13 trainables')
    latest = read(Path(plan['trained_models']['resolved_path']) / 'recovery/latest.json')
    checkpoint = Path(latest['path'])
    require(native.digest(checkpoint / 'manifest.json') == latest['manifest_sha256'], 'Native export recovery manifest changed')
    manifest = read(checkpoint / 'manifest.json')
    candidates = [name for name in manifest['files'] if name.endswith('mp_rank_00_model_states.pt')]
    require(len(candidates) == 1 and manifest['plan_identity_sha256'] == plan['plan_identity_sha256'], 'Native export raw model provenance differs')
    raw_path = checkpoint / candidates[0]
    require(native.digest(raw_path) == manifest['files'][candidates[0]], 'Native export raw source bytes changed')
    raw = torch.load(raw_path, map_location='cpu', weights_only=False)
    require(raw['global_steps'] - raw['skipped_steps'] == 4 and set(state) <= set(raw['module']), 'Native export raw checkpoint coverage differs')
    require(native.tensor_state_identity(state, torch) == native.tensor_state_identity({name: raw['module'][name].to(value.dtype) for name, value in state.items()}, torch), 'Native export does not byte-exactly match its complete raw checkpoint')
    require(native.digest(directory / 'config.json') == native.UPSTREAM['config.json'], 'Native export configuration differs')
    files = {name: {'sha256': native.digest(directory / name), 'size_bytes': (directory / name).stat().st_size} for name in ('config.json', 'pytorch_model.bin')}
    receipt = {'status': 'complete_corrected_native_export_inventory_requires_stock_inference', 'plan': {'path': str(Path(plan_path).resolve()), 'sha256': native.digest(plan_path)}, 'plan_identity_sha256': plan['plan_identity_sha256'], 'source_sha256': corrected.sources(), 'helper': {'path': str(Path(__file__).resolve()), 'sha256': native.digest(__file__)}, 'export_directory': str(directory), 'files': files, 'trainable_state_identity_sha256': native.tensor_state_identity(source, torch), 'vocabulary_cache_sha256': plan['config']['full_body_v3']['vocabulary_cache']['sha256'], 'trainable_names': sorted(source), 'raw_model_source': {'path': str(raw_path), 'sha256': native.digest(raw_path)}, 'raw_checkpoint_manifest': {'path': str(checkpoint / 'manifest.json'), 'sha256': latest['manifest_sha256']}, 'scientific_training_admitted': False}
    require(not Path(output).exists(), 'Native export receipt already exists')
    pool.atomic_json(output, receipt)
    return receipt


def stock(plan_path, receipt_path, output):
    require(os.environ.get('SLURM_JOB_ID') and int(os.environ['WORLD_SIZE']) == 8, 'Native stock export requires eight-GPU Slurm')
    corrected.install()
    plan = native.load_plan(plan_path)
    receipt = read(receipt_path)
    require(receipt['status'] == 'complete_corrected_native_export_inventory_requires_stock_inference' and receipt['plan_identity_sha256'] == plan['plan_identity_sha256'] and receipt['plan']['sha256'] == native.digest(plan_path) and receipt['source_sha256'] == corrected.sources() and receipt['helper'] == {'path': str(Path(__file__).resolve()), 'sha256': native.digest(__file__)}, 'Native export receipt belongs to other source/plan/helper')
    export = Path(receipt['export_directory'])
    require(all(native.digest(export / name) == item['sha256'] for name, item in receipt['files'].items()), 'Native export bytes changed')
    import torch
    from transformers import AutoTokenizer
    rank = int(os.environ['LOCAL_RANK'])
    require(int(os.environ['RANK']) == rank and not (Path(output) / ('rank-' + str(rank) + '.json')).exists(), 'Native stock evidence rank is mismatched or already recorded')
    require(torch.cuda.device_count() == 8 and torch.cuda.get_device_capability(rank) == (8, 0), 'Native stock export requires all eight A100 GPUs')
    torch.cuda.set_device(rank)
    checkout = Path(plan['project_root']) / 'repos/EAGLE'
    sources = {str(path.relative_to(checkout)): pool.digest(path) for path in sorted((checkout / 'eagle/model').rglob('*.py'))}
    sys.path.insert(0, str(checkout))
    from eagle.model.ea_model import EaModel
    model = EaModel.from_pretrained(base_model_path=plan['target']['path'], ea_model_path=str(export), torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={'': rank}, use_eagle3=True, total_token=60, depth=7, top_k=10).eval()
    tokenizer = AutoTokenizer.from_pretrained(plan['target']['path'], local_files_only=True)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Write the next integer after one.'}], tokenize=True, add_generation_prompt=True, return_tensors='pt').to(rank)
    started = time.monotonic()
    with torch.no_grad():
        speculative = model.eagenerate(prompt.clone(), temperature=0, max_new_tokens=16, max_length=2048, is_llama3=True)
        autoregressive = model.naivegenerate(prompt.clone(), temperature=0, max_new_tokens=16, max_length=2048, is_llama3=True)
    torch.cuda.synchronize(rank)
    require(speculative.ndim == autoregressive.ndim == 2 and speculative.shape[0] == autoregressive.shape[0] == 1 and min(speculative.shape[1], autoregressive.shape[1]) > prompt.shape[1] and torch.equal(speculative[:, :prompt.shape[1]], prompt) and torch.equal(autoregressive[:, :prompt.shape[1]], prompt), 'Native stock export did not produce genuine continuations')
    result = {'status': 'passed_stock_corrected_native_export_generation', 'scope': 'one synthetic-prompt compatibility test; no quality/speed/losslessness claim', 'rank': rank, 'plan_identity_sha256': plan['plan_identity_sha256'], 'source_sha256': corrected.sources(), 'helper': {'path': str(Path(__file__).resolve()), 'sha256': native.digest(__file__)}, 'export_receipt': {'path': str(Path(receipt_path).resolve()), 'sha256': pool.digest(receipt_path)}, 'stock_source_files': sources, 'prompt_ids': prompt[0].tolist(), 'speculative_ids': speculative[0].tolist(), 'autoregressive_ids': autoregressive[0].tolist(), 'tokens_equal': torch.equal(speculative, autoregressive), 'generation_wall_seconds': time.monotonic() - started}
    pool.atomic_json(Path(output) / ('rank-' + str(rank) + '.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'stock'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    result = prepare(args.plan, args.output) if args.action == 'prepare' else stock(args.plan, args.receipt, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
