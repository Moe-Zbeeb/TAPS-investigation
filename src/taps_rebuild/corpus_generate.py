import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import heapq
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import native_training, pool


class GenerationError(ValueError):
    pass


def check(condition, message):
    if not condition:
        raise GenerationError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def prepare(root, config_path, corpus, run_id, limit=None):
    check(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Generation preparation requires a CPU Slurm task')
    root, corpus = Path(root).resolve(), Path(corpus).resolve()
    config, source = read(config_path), read(corpus / 'completion.json')
    check(source['status'] == 'complete' and source['config_sha256'] == pool.digest(config_path), 'Corpus configuration or completion differs')
    directory = root / 'outputs/teacher-generation' / run_id
    check(directory.resolve() == directory and not directory.exists(), 'Choose a fresh teacher-generation run')
    assets_path = root / 'outputs/preparation/assets.json'
    target = native_training.model_record(read(assets_path), config['target_id'], verify_content=True)
    directory.mkdir(parents=True)
    inputs = directory / 'inputs'
    inputs.mkdir()
    scripts = {}
    for name in ('corpus_generate.py', 'native_training.py', 'pool.py'):
        path = inputs / name
        path.write_bytes(Path(__file__).with_name(name).read_bytes())
        scripts[name] = pool.digest(path)
    main = root / 'repos/EAGLE/eagle/traineagle3/main.py'
    check(pool.digest(main) == native_training.UPSTREAM['main.py'], 'Native system-prompt source changed')
    counts = [0] * 8
    handles = [(inputs / ('shard-' + str(index) + '.jsonl')).open('x') for index in range(8)]
    selected = 0
    input_proofs = []
    try:
        for split in ('train', 'dev', 'test'):
            path = corpus / (split + '.jsonl')
            check(pool.digest(path) == source['files'][path.name]['sha256'], 'Canonical corpus changed')
            input_proofs.append({'split': split, 'sha256': pool.digest(path)})
            for row in rows(path):
                if limit is not None and selected >= limit:
                    break
                rank = selected % 8
                handles[rank].write(native_training.canonical({'ordinal': selected, 'record': row}).decode() + '\n')
                counts[rank] += 1
                selected += 1
    finally:
        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
    check(selected >= 8 and all(counts), 'Teacher preparation must assign work to all eight GPUs')
    prompt = native_training.system_prompt(main)
    plan = {'status': 'prepared', 'scope': 'teacher_generation_smoke' if limit is not None else 'full_training_data_preparation', 'root': str(root), 'directory': str(directory), 'config': config, 'config_sha256': pool.digest(config_path), 'corpus': str(corpus), 'corpus_completion_sha256': pool.digest(corpus / 'completion.json'), 'target_id': config['target_id'], 'target': target, 'assets_sha256': pool.digest(assets_path), 'system_prompt': prompt, 'system_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'native_main_sha256': pool.digest(main), 'scripts': scripts, 'input_proofs': input_proofs, 'shards': [{'path': str(inputs / ('shard-' + str(index) + '.jsonl')), 'sha256': pool.digest(inputs / ('shard-' + str(index) + '.jsonl')), 'examples': count} for index, count in enumerate(counts)], 'world_size': 8, 'examples': selected, 'teacher_features': 'online native training', 'training_started': False}
    plan['identity_sha256'] = pool.identity(plan)
    pool.atomic_json(directory / 'plan.json', plan)
    return plan


def load_plan(path):
    plan = read(path)
    identity = plan.pop('identity_sha256')
    check(pool.identity(plan) == identity, 'Teacher plan changed')
    plan['identity_sha256'] = identity
    directory = Path(plan['directory'])
    check(Path(path).resolve() == directory / 'plan.json', 'Teacher plan was relocated')
    for name, expected in plan['scripts'].items():
        check(pool.digest(directory / 'inputs' / name) == expected, 'Frozen teacher source changed')
        check(pool.digest(Path(__file__).with_name(name)) == expected, 'Executing teacher source differs from the frozen plan')
    for name, item in plan['target']['files'].items():
        file = Path(plan['target']['path']) / name
        check(file.is_file() and file.stat().st_size == item['size_bytes'], 'Teacher model inventory changed')
    return plan


def generate_batch(llm, tokenizer, records, settings, system_prompt, sampling_factory):
    state = []
    stops = sorted({token for token in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')) if isinstance(token, int) and token >= 0})
    for item in records:
        state.append({'ordinal': item['ordinal'], 'source': item['record'], 'users': [message['content'] for message in item['record']['messages'] if message['role'] == 'user'], 'messages': [], 'turns': [], 'reason': None})
    started = time.perf_counter()
    for turn in range(max(len(item['users']) for item in state)):
        active, prompts, parameters = [], [], []
        for item in state:
            if item['reason'] is not None or turn >= len(item['users']):
                continue
            item['messages'].append({'role': 'user', 'content': item['users'][turn]})
            messages = [{'role': 'system', 'content': system_prompt}, *item['messages']]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            remaining = settings['max_context'] - len(ids)
            if remaining < 1:
                item['reason'] = 'prompt_exceeds_generation_context'
                continue
            seed = int(pool.identity([settings['seed'], item['source']['example_id'], turn])[:8], 16)
            parameters.append(sampling_factory(temperature=settings['temperature'], top_p=settings['top_p'], max_tokens=min(settings['max_new_tokens'], remaining), seed=seed, stop_token_ids=stops))
            prompts.append({'prompt_token_ids': ids})
            active.append(item)
        if not active:
            continue
        outputs = llm.generate(prompts, parameters, use_tqdm=False)
        check(len(outputs) == len(active), 'Teacher batch output count differs')
        for item, prompt, response in zip(active, prompts, outputs):
            check(len(response.outputs) == 1, 'Expected one target continuation')
            if response.prompt_token_ids is not None:
                check(list(response.prompt_token_ids) == prompt['prompt_token_ids'], 'Teacher output was paired with another prompt')
            output = response.outputs[0]
            item['turns'].append({'turn': turn, 'prompt_sha256': pool.identity(prompt['prompt_token_ids']), 'prompt_tokens': len(prompt['prompt_token_ids']), 'output_tokens': len(output.token_ids), 'finish_reason': output.finish_reason, 'stop_reason': output.stop_reason})
            if output.finish_reason != 'stop':
                item['reason'] = 'response_censored_at_generation_limit'
            elif not output.text.strip() or any(token in output.text for token in native_training.SPECIAL):
                item['reason'] = 'empty_or_template_control_response'
            else:
                item['messages'].append({'role': 'assistant', 'content': output.text})
    outcomes = []
    for item in state:
        record = None
        if item['reason'] is None:
            record = {**item['source'], 'messages': item['messages'], 'response_regime': 'target_generated', 'original_response_complete': True}
            native_training.native_row(record, canonical_messages=True, split=record['split'])
        outcomes.append({'ordinal': item['ordinal'], 'example_id': item['source']['example_id'], 'status': 'complete' if record else 'excluded', 'reason': item['reason'], 'record': record, 'generation_turns': item['turns']})
    return {'outcomes': outcomes, 'wall_seconds': time.perf_counter() - started, 'generated_tokens': sum(turn['output_tokens'] for item in state for turn in item['turns'])}


def committed(path, identity):
    completed = {}
    if not path.exists():
        return completed
    with path.open('r+b') as stream:
        boundary = 0
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                recovery = path.with_name(path.name + '.uncommitted-' + str(time.time_ns()))
                recovery.write_bytes(line)
                stream.truncate(boundary)
                break
            row = json.loads(line)
            check(row['plan_identity_sha256'] == identity, 'Existing teacher chunk belongs to another plan')
            for item in row['outcomes']:
                check(item['ordinal'] not in completed, 'Duplicate teacher completion')
                completed[item['ordinal']] = item['example_id']
            boundary = stream.tell()
    return completed


def worker(plan_path, rank):
    plan = load_plan(plan_path)
    check(os.environ.get('SLURM_JOB_ID') and len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) == 1, 'Teacher worker needs one allocated GPU')
    from vllm import LLM, SamplingParams
    import torch
    check(torch.cuda.device_count() == 1, 'Teacher worker GPU binding differs')
    check(importlib.metadata.version('vllm') == '0.8.5.post1', 'Use the pinned serving-vllm runtime')
    shard = plan['shards'][rank]
    check(pool.digest(shard['path']) == shard['sha256'], 'Teacher shard changed')
    directory = Path(plan['directory'])
    output = directory / ('worker-' + str(rank) + '.jsonl')
    completed = committed(output, plan['identity_sha256'])
    if len(completed) == shard['examples']:
        pool.atomic_json(directory / ('worker-' + str(rank) + '-receipt.json'), {'status': 'complete', 'rank': rank, 'plan_identity_sha256': plan['identity_sha256'], 'examples': len(completed), 'raw_sha256': pool.digest(output), 'slurm_job_id': os.environ['SLURM_JOB_ID']})
        return
    llm = LLM(model=plan['target']['path'], tokenizer=plan['target']['path'], tensor_parallel_size=1, dtype='float16', seed=plan['config']['target_generation']['seed'], gpu_memory_utilization=0.85, max_model_len=plan['config']['target_generation']['max_context'], max_num_seqs=64, enable_prefix_caching=True, enforce_eager=True, disable_log_stats=True, hf_token=False)
    tokenizer = llm.get_tokenizer()
    settings = plan['config']['target_generation']
    batch_size = settings['batch_conversations_per_worker']
    started, generated = time.perf_counter(), 0
    try:
        with output.open('ab') as stream:
            batch = []
            for row in rows(shard['path']):
                if row['ordinal'] in completed:
                    check(completed[row['ordinal']] == row['record']['example_id'], 'Resumed teacher input differs')
                    continue
                batch.append(row)
                if len(batch) < batch_size:
                    continue
                result = generate_batch(llm, tokenizer, batch, settings, plan['system_prompt'], SamplingParams)
                result.update(plan_identity_sha256=plan['identity_sha256'], rank=rank, slurm_job_id=os.environ['SLURM_JOB_ID'], cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
                stream.write(native_training.canonical(result) + b'\n')
                stream.flush()
                os.fsync(stream.fileno())
                generated += result['generated_tokens']
                completed.update({item['ordinal']: item['example_id'] for item in result['outcomes']})
                pool.atomic_json(directory / ('worker-' + str(rank) + '-progress.json'), {'completed_examples': len(completed), 'expected_examples': shard['examples'], 'generated_tokens_this_attempt': generated, 'elapsed_seconds_this_attempt': time.perf_counter() - started})
                batch = []
            if batch:
                result = generate_batch(llm, tokenizer, batch, settings, plan['system_prompt'], SamplingParams)
                result.update(plan_identity_sha256=plan['identity_sha256'], rank=rank, slurm_job_id=os.environ['SLURM_JOB_ID'], cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
                stream.write(native_training.canonical(result) + b'\n')
                stream.flush()
                os.fsync(stream.fileno())
                completed.update({item['ordinal']: item['example_id'] for item in result['outcomes']})
        check(len(completed) == shard['examples'], 'Incomplete teacher shard')
        pool.atomic_json(directory / ('worker-' + str(rank) + '-receipt.json'), {'status': 'complete', 'rank': rank, 'plan_identity_sha256': plan['identity_sha256'], 'examples': len(completed), 'raw_sha256': pool.digest(output), 'slurm_job_id': os.environ['SLURM_JOB_ID']})
    finally:
        del llm


def outcomes(path):
    for batch in rows(path):
        yield from batch['outcomes']


def merge(plan):
    directory = Path(plan['directory'])
    output = Path(plan['root']) / 'data/corpora' / (directory.name + '-target-generated')
    if (output / 'completion.json').exists():
        receipt = read(output / 'completion.json')
        check(receipt['plan_identity_sha256'] == plan['identity_sha256'] and all(pool.digest(output / name) == item['sha256'] for name, item in receipt['files'].items()), 'Existing merged target corpus differs')
        pool.atomic_json(directory / 'completion.json', {**receipt, 'corpus_output': str(output)})
        return receipt
    if output.exists():
        output.rename(output.with_name(output.name + '.interrupted-' + str(time.time_ns())))
    output.mkdir()
    paths = []
    for index in range(8):
        path = directory / ('worker-' + str(index) + '.jsonl')
        receipt = read(directory / ('worker-' + str(index) + '-receipt.json'))
        check(receipt['status'] == 'complete' and receipt['plan_identity_sha256'] == plan['identity_sha256'] and receipt['raw_sha256'] == pool.digest(path), 'Teacher worker completion proof differs')
        paths.append(path)
    counts = Counter()
    handles = {split: ((output / (split + '.jsonl')).open('x'), (output / ('native-' + split + '.jsonl')).open('x')) for split in ('train', 'dev', 'test')}
    try:
        for expected, row in enumerate(heapq.merge(*(outcomes(path) for path in paths), key=lambda x: x['ordinal'])):
            check(row['ordinal'] == expected, 'Teacher coverage has a duplicate or gap')
            counts['processed'] += 1
            if row['status'] == 'excluded':
                counts['excluded:' + row['reason']] += 1
                continue
            value = row['record']
            split = value['split']
            handles[split][0].write(native_training.canonical(value).decode() + '\n')
            handles[split][1].write(native_training.canonical(native_training.native_row(value, canonical_messages=True, split=split)).decode() + '\n')
            counts['accepted:' + split] += 1
        check(counts['processed'] == plan['examples'], 'Teacher population is incomplete')
    finally:
        for pair in handles.values():
            for stream in pair:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
    files = {p.name: {'sha256': pool.digest(p), 'size_bytes': p.stat().st_size} for p in output.glob('*.jsonl')}
    receipt = {'status': 'complete', 'scope': plan['scope'], 'plan_identity_sha256': plan['identity_sha256'], 'plan_path': str(directory / 'plan.json'), 'target_id': plan['target_id'], 'target_identity_sha256': plan['target']['model_identity_sha256'], 'response_regime': 'target_generated', 'counts': dict(counts), 'files': files, 'teacher_features': 'online native training', 'native_tokenization_filter_performed': False, 'training_started': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    pool.atomic_json(output / 'completion.json', receipt)
    pool.atomic_json(directory / 'completion.json', {**receipt, 'corpus_output': str(output)})
    return receipt


def run(plan_path):
    plan = load_plan(plan_path)
    hardware = pool.allocation(8)
    directory = Path(plan['directory'])
    lock = (directory / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pool.atomic_json(directory / ('attempt-' + os.environ['SLURM_JOB_ID'] + '.json'), {'hardware': hardware, 'plan_identity_sha256': plan['identity_sha256'], 'started_at_utc': datetime.now(timezone.utc).isoformat()})
    python = Path(plan['root']) / plan['config']['target_generation']['runtime_python']
    package_root = Path(__file__).parents[1]
    children = []
    try:
        for rank, binding in enumerate(hardware['device_bindings']):
            environment = pool.worker_environment(binding['token'], package_root.parent, Path(plan['root']), python)
            environment['PYTHONPATH'] = str(package_root)
            short = Path('/tmp') / ('taps-g-' + str(os.getuid()) + '-' + os.environ['SLURM_JOB_ID'] + '-' + str(rank))
            short.mkdir(exist_ok=True)
            check(short.stat().st_uid == os.getuid() and not short.is_symlink(), 'Unsafe teacher IPC path')
            environment.update(VLLM_USE_V1='1', VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_RPC_BASE_PATH=str(short), TMPDIR=str(short), VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')
            log = (directory / ('worker-' + str(rank) + '.log')).open('ab')
            command = [str(python), '-B', '-m', 'taps_rebuild.corpus_generate', 'worker', '--plan', str(plan_path), '--rank', str(rank)]
            process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),), start_new_session=True)
            children.append((process, log))
        while any(process.poll() is None for process, _ in children):
            check(not any(process.poll() not in (None, 0) for process, _ in children), 'Teacher worker failed')
            time.sleep(2)
        check(all(process.returncode == 0 for process, _ in children), 'Teacher worker failed')
        return merge(plan)
    finally:
        for process, log in children:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process, log in children:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            log.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    preparation = subparsers.add_parser('prepare')
    preparation.add_argument('--root', type=Path, required=True)
    preparation.add_argument('--config', type=Path, required=True)
    preparation.add_argument('--corpus', type=Path, required=True)
    preparation.add_argument('--run-id', required=True)
    preparation.add_argument('--limit', type=int)
    for action in ('run', 'worker'):
        command = subparsers.add_parser(action)
        command.add_argument('--plan', type=Path, required=True)
        if action == 'worker':
            command.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        value = prepare(args.root, args.config, args.corpus, args.run_id, args.limit)
    elif args.command == 'run':
        value = run(args.plan)
    else:
        worker(args.plan, args.rank)
        return
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
