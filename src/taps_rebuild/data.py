import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as handle:
        handle.write(canonical(value) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def records(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        value = json.loads(path.read_text())
        if not isinstance(value, list):
            raise ValueError('JSON inputs must contain an array of examples')
        yield from value


def messages_from(row):
    messages = row.get('messages', row.get('conversations'))
    if messages is None and 'instruction' in row and 'output' in row:
        prompt = row['instruction'] + ('\n' + row['input'] if row.get('input') else '')
        messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': row['output']}]
    if messages is None and 'prompt' in row and 'response' in row:
        messages = [{'role': 'user', 'content': row['prompt']}, {'role': 'assistant', 'content': row['response']}]
    if not isinstance(messages, list) or not messages:
        raise ValueError('Example requires a supported text conversation')
    roles = {'human': 'user', 'gpt': 'assistant', 'user': 'user', 'assistant': 'assistant', 'system': 'system'}
    result = []
    for message in messages:
        role = roles.get(message.get('role', message.get('from')))
        content = message.get('content', message.get('value'))
        if role is None or not isinstance(content, str) or not content.strip():
            raise ValueError('Conversation roles and nonempty text must be explicit')
        result.append({'role': role, 'content': content})
    start = 1 if result[0]['role'] == 'system' else 0
    if any(item['role'] != ('user' if (index - start) % 2 == 0 else 'assistant') for index, item in enumerate(result) if index >= start):
        raise ValueError('Conversation must alternate user and assistant turns')
    if result[-1]['role'] != 'assistant' or start == len(result):
        raise ValueError('Training conversation must end with an assistant response')
    return result


def prompt_identity(messages):
    value = [{'role': item['role'], 'content': unicodedata.normalize('NFKC', item['content']).replace('\r\n', '\n').strip()} for item in messages if item['role'] != 'assistant']
    return identity(value)


def split_for(prompt_id, seed, train_fraction, dev_fraction):
    if not 0 < train_fraction < 1 or not 0 < dev_fraction < 1 or train_fraction + dev_fraction >= 1:
        raise ValueError('Train/dev fractions must leave a nonempty test fraction')
    value = int(identity([prompt_id, seed])[:16], 16) / 2**64
    return 'train' if value < train_fraction else 'dev' if value < train_fraction + dev_fraction else 'test'


def normalize_rows(rows, *, domain, source_id, source_revision, seed, train_fraction, dev_fraction, excluded_prompt_ids=()):
    if not re.fullmatch('[a-z0-9_-]+', domain) or not source_id or not source_revision:
        raise ValueError('Domain and immutable source identity are required')
    seen = set(excluded_prompt_ids)
    normalized, excluded = [], []
    for index, row in enumerate(rows):
        messages = messages_from(row)
        prompt_id = prompt_identity(messages)
        example_id = identity([source_id, source_revision, index, prompt_id])
        if prompt_id in seen:
            excluded.append({'source_row': index, 'example_id': example_id, 'prompt_id': prompt_id, 'reason': 'duplicate_or_explicit_holdout'})
            continue
        seen.add(prompt_id)
        normalized.append({'example_id': example_id, 'prompt_id': prompt_id, 'domain': domain, 'split': split_for(prompt_id, seed, train_fraction, dev_fraction), 'messages': messages, 'response_regime': 'original', 'source': {'id': source_id, 'revision': source_revision, 'row': index}})
    return normalized, excluded


def assistant_token_mask(tokenizer, messages):
    complete = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    mask = [0] * len(complete)
    for index, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        before = tokenizer.apply_chat_template(messages[:index], tokenize=True, add_generation_prompt=True)
        through = tokenizer.apply_chat_template(messages[:index + 1], tokenize=True, add_generation_prompt=False)
        if through[:len(before)] != before or complete[:len(through)] != through:
            raise ValueError('Tokenizer template does not support an exact assistant span')
        mask[len(before):len(through)] = [1] * (len(through) - len(before))
    if not any(mask):
        raise ValueError('Conversation has no supervised assistant tokens')
    return complete, mask


def require_slurm(gpu=False):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Data processing requires Slurm')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpu and (not visible or len(visible.split(',')) != 1):
        raise ValueError('Expose exactly one GPU per data worker')
    if not gpu and visible:
        raise ValueError('CPU preparation requires empty CUDA_VISIBLE_DEVICES')


def output_path(root, path):
    root, path = Path(root).resolve(), Path(path)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root / 'data'):
        raise ValueError('Dataset outputs must remain under project data/')
    return path


def prepared_target(root, target_id):
    receipt_path = Path(root) / 'outputs/preparation/assets.json'
    receipt = json.loads(receipt_path.read_text())
    model = receipt['models'][target_id]
    if model['status'] != 'verified':
        raise ValueError('Target checkpoint must have verified local files')
    return model, file_digest(receipt_path)


def load_teacher(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Data worker requires one usable CUDA device')
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16, device_map={'': 0}, attn_implementation='sdpa').eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return torch, model, tokenizer


def feature_tensors(torch, tokenizer, model, messages, domain_id, feature_layers, max_tokens, include_teacher_logits):
    tokens, mask = assistant_token_mask(tokenizer, messages)
    if len(tokens) > max_tokens or len(tokens) < 3:
        raise ValueError('Example exceeds the declared feature context or is too short')
    inputs = torch.tensor([tokens], device='cuda:0', dtype=torch.long)
    with torch.inference_mode():
        result = model(inputs, output_hidden_states=True, use_cache=False)
        if any(not 0 <= layer < len(result.hidden_states) for layer in feature_layers):
            raise ValueError('Teacher feature layer index is invalid')
        selected = torch.cat([result.hidden_states[layer] for layer in feature_layers], dim=-1)[0]
        value = {'input_ids': inputs[0, 1:-1].cpu(), 'features': selected[:-2].cpu(), 'feature_targets': selected[1:-1].cpu(), 'labels': inputs[0, 2:].cpu(), 'loss_mask': torch.tensor(mask[2:], dtype=torch.bool), 'attention_mask': torch.ones(len(tokens) - 2, dtype=torch.bool), 'domain_id': int(domain_id)}
        if include_teacher_logits:
            value['teacher_logits'] = result.logits[0, 1:-1].to(dtype=torch.float16).cpu()
    if not value['loss_mask'].any().item():
        raise ValueError('Aligned example has no supervised next-token labels')
    return value


def regenerate(torch, model, tokenizer, example, *, seed, max_new_tokens, max_context, temperature):
    messages = []
    turns = []
    for index, item in enumerate(example['messages']):
        if item['role'] != 'assistant':
            messages.append(dict(item))
            continue
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to('cuda:0')
        if encoded.shape[1] + max_new_tokens > max_context:
            raise ValueError('Regeneration exceeds the declared common context budget')
        torch.manual_seed(seed + index)
        torch.cuda.manual_seed_all(seed + index)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = model.generate(encoded, max_new_tokens=max_new_tokens, do_sample=temperature > 0, **({'temperature': temperature, 'top_p': 1.0, 'top_k': 0} if temperature > 0 else {}))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        new = result[0, encoded.shape[1]:].cpu().tolist()
        terminators = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')}
        configured_eos = model.generation_config.eos_token_id
        terminators.update(configured_eos if isinstance(configured_eos, list) else [configured_eos])
        stop = next((i for i, token in enumerate(new) if token in terminators), len(new))
        messages.append({'role': 'assistant', 'content': tokenizer.decode(new[:stop], skip_special_tokens=True)})
        turns.append({'message_index': index, 'seed': seed + index, 'prompt_tokens': int(encoded.shape[1]), 'generated_tokens': len(new), 'response_tokens_before_stop': stop, 'stop_token_id': new[stop] if stop < len(new) else None, 'elapsed_seconds': elapsed})
    value = dict(example)
    value['messages'] = messages
    value['response_regime'] = 'target_regenerated_all_assistant_turns'
    value['original_messages_sha256'] = identity(example['messages'])
    value['generation'] = {'seed': seed, 'temperature': temperature, 'max_new_tokens': max_new_tokens, 'max_context': max_context, 'sampling_top_p': 1.0, 'sampling_top_k': 0, 'turns': turns}
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    normalize = commands.add_parser('normalize')
    normalize.add_argument('--input', type=Path, required=True)
    normalize.add_argument('--output', type=Path, required=True)
    normalize.add_argument('--domain', required=True)
    normalize.add_argument('--source-id', required=True)
    normalize.add_argument('--source-revision', required=True)
    normalize.add_argument('--seed', type=int, default=20261109)
    normalize.add_argument('--train-fraction', type=float, required=True)
    normalize.add_argument('--dev-fraction', type=float, required=True)
    normalize.add_argument('--exclude', type=Path, action='append', default=[])
    for name in ['regenerate', 'features']:
        command = commands.add_parser(name)
        command.add_argument('--input', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--target-id', required=True)
        command.add_argument('--limit', type=int, required=True)
        command.add_argument('--max-context', type=int, default=2048)
    regen = commands.choices['regenerate']
    regen.add_argument('--seed', type=int, default=20261109)
    regen.add_argument('--temperature', type=float, default=0)
    regen.add_argument('--max-new-tokens', type=int, required=True)
    features = commands.choices['features']
    features.add_argument('--domain-map', type=Path, required=True)
    features.add_argument('--feature-layers', required=True)
    features.add_argument('--include-teacher-logits', action='store_true')
    args = parser.parse_args()
    require_slurm(gpu=args.command != 'normalize')
    root = args.root.resolve()
    destination = output_path(root, args.output)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Choose a fresh dataset output; existing inputs are never overwritten')
    if args.command == 'normalize':
        if args.input.resolve().is_relative_to(root / 'data/reference'):
            raise ValueError('Published evaluation prompts cannot become research training data')
        excluded = [row['prompt_id'] for path in args.exclude for row in records(path)]
        values, rejected = normalize_rows(records(args.input), domain=args.domain, source_id=args.source_id, source_revision=args.source_revision, seed=args.seed, train_fraction=args.train_fraction, dev_fraction=args.dev_fraction, excluded_prompt_ids=excluded)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as handle:
            for value in values:
                handle.write(canonical(value) + '\n')
        atomic_json(destination.with_suffix('.manifest.json'), {'status': 'complete', 'input_sha256': file_digest(args.input), 'output_sha256': file_digest(destination), 'examples': len(values), 'excluded': rejected, 'source_id': args.source_id, 'source_revision': args.source_revision, 'seed': args.seed, 'split_fractions': {'train': args.train_fraction, 'dev': args.dev_fraction, 'test': 1 - args.train_fraction - args.dev_fraction}})
        return
    if not 1 <= args.limit <= 1_000_000 or args.max_context < 4:
        raise ValueError('Explicit bounded example count and context are required')
    target, asset_receipt_sha = prepared_target(root, args.target_id)
    torch, model, tokenizer = load_teacher(target['path'])
    destination.mkdir(parents=True, exist_ok=False)
    index, started = [], time.time()
    domain_map = json.loads(args.domain_map.read_text()) if args.command == 'features' else None
    layers = [int(value) for value in args.feature_layers.split(',')] if args.command == 'features' else None
    output_file = (destination / 'records.jsonl').open('x')
    try:
        for ordinal, example in enumerate(records(args.input)):
            if ordinal >= args.limit:
                break
            if args.command == 'regenerate':
                generation_seed = int(identity([args.seed, example['prompt_id'], args.target_id])[:8], 16)
                value = regenerate(torch, model, tokenizer, example, seed=generation_seed, max_new_tokens=args.max_new_tokens, max_context=args.max_context, temperature=args.temperature)
                value['target_model_identity_sha256'] = target['model_identity_sha256']
                output_file.write(canonical(value) + '\n')
                output_file.flush()
                os.fsync(output_file.fileno())
                index.append({'example_id': value['example_id'], 'prompt_id': value['prompt_id'], 'messages_sha256': identity(value['messages'])})
            else:
                value = feature_tensors(torch, tokenizer, model, example['messages'], domain_map[example['domain']], layers, args.max_context, args.include_teacher_logits)
                path = destination / f'{ordinal:08d}.pt'
                temporary = path.with_suffix('.pt.tmp')
                with temporary.open('wb') as handle:
                    torch.save(value, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(path)
                item = {'path': str(path), 'sha256': file_digest(path), 'example_id': example['example_id'], 'prompt_id': example['prompt_id'], 'domain': example['domain'], 'split': example['split'], 'response_regime': example['response_regime'], 'tokens': len(value['input_ids']), 'loss_tokens': int(value['loss_mask'].sum().item()), 'messages_sha256': identity(example['messages'])}
                index.append(item)
                output_file.write(canonical(item) + '\n')
                output_file.flush()
                os.fsync(output_file.fileno())
        atomic_json(destination / 'manifest.json', {'status': 'complete', 'command': args.command, 'scope': 'prepared_data_not_scientific_results', 'processing_config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'source_sha256': file_digest(__file__), 'input_sha256': file_digest(args.input), 'target_model_id': args.target_id, 'target_model_identity_sha256': target['model_identity_sha256'], 'asset_receipt_sha256': asset_receipt_sha, 'feature_layers': layers, 'feature_layer_indexing': 'Hugging Face output_hidden_states indices, before corresponding decoder layer', 'alignment': 'features[t]=teacher_features[t], input_ids[t]=original[t+1], feature_targets[t]=teacher_features[t+1], labels[t]=original[t+2], teacher_logits[t]=teacher_logits[t+1]', 'include_teacher_logits': bool(getattr(args, 'include_teacher_logits', False)), 'examples': index, 'elapsed_seconds': time.time() - started, 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'completed_at_utc': datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        atomic_json(destination / 'manifest.json', {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error), 'completed_examples': index, 'input_sha256': file_digest(args.input), 'target_model_identity_sha256': target['model_identity_sha256']})
        raise
    finally:
        output_file.close()


if __name__ == '__main__':
    main()
