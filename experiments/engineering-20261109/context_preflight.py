import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer


root = Path(os.environ['TAPS_PROJECT_ROOT'])
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('Context preflight requires a CPU Slurm step with CUDA hidden')
config = json.loads((root / 'configs/reproduction.json').read_bytes())
assets = json.loads((root / 'outputs/preparation/assets.json').read_bytes())
sources = json.loads((root / 'outputs/preparation/sources.json').read_bytes())
results = []
for method, selected in config['methods'].items():
    tokenizer = AutoTokenizer.from_pretrained(assets['models'][selected['target']]['path'], local_files_only=True)
    system = sources['sources'][selected['source']]['system_prompt']['text']
    for benchmark in selected['benchmarks']:
        entry = sources['benchmarks'][benchmark]
        lengths = []
        for line in Path(entry['path']).read_text().splitlines():
            row = json.loads(line)
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': row['turns'][0]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            size = len(tokenizer.encode(text, add_special_tokens=False))
            lengths.append({'question_id': row['question_id'], 'tokens': size})
        effective = config['max_length'] - config['total_token'] + 1 - 10
        results.append({'method': method, 'benchmark': benchmark, 'questions': len(lengths), 'max_prompt_tokens': max(item['tokens'] for item in lengths), 'effective_context_limit': effective, 'out_of_context': [item for item in lengths if item['tokens'] >= effective], 'generation_budget_limited': [item for item in lengths if item['tokens'] + config['max_new_tokens'] > effective]})
value = {'status': 'passed' if not any(item['out_of_context'] for item in results) else 'failed', 'scope': 'first_turn_tokenizer_context_preflight', 'subsequent_turns': 'checked by the GPU runner before generation', 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'config_sha256': hashlib.sha256((root / 'configs/reproduction.json').read_bytes()).hexdigest(), 'results': results}
destination = root / 'outputs/engineering/context-preflight-20261109.json'
with destination.open('x') as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps(value, sort_keys=True))
if value['status'] != 'passed':
    raise SystemExit(1)
