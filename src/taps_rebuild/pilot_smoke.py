import argparse
import json
import os
from pathlib import Path
import sys
import time

from . import pool
from .pilot_cache import read, require
from .pilot_gate import backend, devices
from .pilot_train import load_plan


def run(plan_path, export, output):
    require(os.environ.get('SLURM_JOB_ID') and int(os.environ['WORLD_SIZE']) == 8, 'Export smoke requires eight-GPU Slurm')
    import torch
    from transformers import AutoTokenizer

    plan = load_plan(plan_path, qualification=True)
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    require(backend(torch) == plan['scan_runtime'] and devices() == plan['hardware'], 'Export smoke hardware/runtime differs')
    receipt = read(export / 'export.json')
    require(receipt['plan_identity_sha256'] == plan['identity_sha256'] and all(pool.digest(export / name) == item['sha256'] for name, item in receipt['files'].items()), 'Trained export bytes changed')
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
    require(speculative.ndim == autoregressive.ndim == 2 and speculative.shape[0] == autoregressive.shape[0] == 1 and min(speculative.shape[1], autoregressive.shape[1]) > prompt.shape[1] and torch.equal(speculative[:, :prompt.shape[1]], prompt) and torch.equal(autoregressive[:, :prompt.shape[1]], prompt), 'Stock inference did not produce genuine continuations')
    output.mkdir(parents=True, exist_ok=True)
    result = {'status': 'passed_stock_export_generation', 'scope': 'trained export compatibility on one synthetic prompt; no quality/speed/losslessness result', 'rank': rank, 'plan_identity_sha256': plan['identity_sha256'], 'export_path': str(export), 'export_receipt_sha256': pool.digest(export / 'export.json'), 'export_state_identity_sha256': receipt['state_identity_sha256'], 'stock_source_files': sources, 'prompt_ids': prompt[0].tolist(), 'speculative_ids': speculative[0].tolist(), 'autoregressive_ids': autoregressive[0].tolist(), 'tokens_equal': torch.equal(speculative, autoregressive), 'generation_wall_seconds': time.monotonic() - started}
    pool.atomic_json(output / ('rank-' + str(rank) + '.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'export', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.export, args.output), indent=2))


if __name__ == '__main__':
    main()
