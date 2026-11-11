import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def worker(args):
    import numpy
    import torch
    from taps_rebuild.backends.official import OfficialBackend
    from taps_rebuild.runner import seed_runtime, verify_provenance

    selected = read(args.prepared / 'cases.json')['eagle3']
    provenance = verify_provenance(args.root, selected['run_config'], selected['run_manifest'], 'eagle3')[-1]
    case = selected['cases'][args.index]
    pair, probe = case['pair'], case['divergence']
    prompt = pair['target_only']['input_token_ids']
    expected = prompt + pair['target_only']['output_token_ids'][:probe]
    backend = OfficialBackend('eagle3', selected['source_path'], selected['target_path'], selected['draft_path'], total_token=selected['config']['total_token'], depth=selected['definition']['depth'], top_k=selected['config']['top_k'])
    backend.close()
    backend.model = backend.module.EaModel.from_pretrained(base_model_path=selected['target_path'], ea_model_path=selected['draft_path'], total_token=selected['config']['total_token'], depth=selected['definition']['depth'], top_k=selected['config']['top_k'], torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map={'': 0}, local_files_only=True, use_eagle3=True).eval()
    captures, outputs = {}, {}
    def cpu(value):
        return value.detach().cpu().clone()
    def arena(data, lengths):
        return {'pointers': [int(x.data_ptr()) for x in data], 'shapes': [list(x.shape) for x in data], 'dtypes': [str(x.dtype) for x in data], 'logical_lengths': cpu(lengths)}
    try:
        for mode in ('target_only', 'speculative'):
            outputs[mode] = []
            for repeat in range(2):
                original_forward = backend.model.base_model.forward
                original_tree = backend.module.tree_decoding
                original_posterior = backend.module.evaluate_posterior
                state = {'prefix': [], 'tree': None}
                captured = []
                def forward(*values, **kwargs):
                    ids = values[0] if values else kwargs['input_ids']
                    state['prefix'].extend(ids[0].detach().cpu().tolist())
                    before = arena(backend.model.past_key_values_data, backend.model.current_length_data)
                    mask = getattr(backend.model.base_model.model, 'tree_mask', None)
                    result = original_forward(*values, **kwargs)
                    if len(state['prefix']) == len(expected):
                        captured.append({'logits': cpu(result.logits[0, -1]), 'prefix': torch.tensor(state['prefix']), 'cache_before': before, 'tree_mask': cpu(mask) if mask is not None else None, 'position_ids': cpu(kwargs['position_ids']) if kwargs.get('position_ids') is not None else None})
                    return result
                def tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices):
                    state['tree'] = {'tree_prefix': cpu(input_ids[0]), 'tree_candidates': cpu(tree_candidates), 'retrieve_indices': cpu(retrieve_indices), 'position_ids': cpu(tree_position_ids + input_ids.shape[1]), 'tree_mask': cpu(model.base_model.model.tree_mask), 'cache_before': arena(model.past_key_values_data, model.current_length_data)}
                    return original_tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices)
                def posterior(logits, candidates, logits_processor):
                    result = original_posterior(logits, candidates, logits_processor)
                    best, accepted = int(result[0]), int(result[1])
                    record = state['tree']
                    committed = len(record['tree_prefix']) - len(prompt)
                    if committed < probe <= committed + accepted + 1:
                        offset = probe - committed
                        captured.append({**record, 'prefix': torch.cat((record['tree_prefix'], cpu(candidates[best, :offset]))), 'logits': cpu(logits[best, offset - 1]), 'candidates': cpu(candidates), 'best_row': best, 'accepted': accepted, 'offset': offset, 'root_token_id': int(candidates[0, 0])})
                    return result
                if mode == 'target_only':
                    backend.model.base_model.forward = forward
                else:
                    backend.module.tree_decoding = tree
                    backend.module.evaluate_posterior = posterior
                try:
                    seed_runtime(pair[mode]['seed'], torch, numpy)
                    result = backend.generate(torch.tensor([prompt]), temperature=0, max_new_tokens=probe + 4, max_length=selected['config']['max_length'], generation_mode=mode)
                    tokens = result['output_ids'][0, len(prompt):].cpu().tolist()
                    require(len(captured) == 1, 'Probe capture is missing or duplicated')
                    captured[0]['emitted_token_id'] = tokens[probe]
                    captures[mode + '-' + str(repeat)] = captured[0]
                    outputs[mode].append(tokens)
                finally:
                    backend.model.base_model.forward = original_forward
                    backend.module.tree_decoding = original_tree
                    backend.module.evaluate_posterior = original_posterior
        old_data = backend.model.past_key_values_data
        for repeat in range(2):
            backend.module.reset_tree_mode(backend.model)
            past, data, lengths = backend.module.initialize_past_key_values(backend.model.base_model, max_length=selected['config']['max_length'])
            before = arena(data, lengths)
            require(not set(before['pointers']) & {int(x.data_ptr()) for x in old_data}, 'Clean cache aliases inherited cache')
            with torch.inference_mode():
                result = backend.model.base_model(torch.tensor([expected], device=backend.model.base_model.device), past_key_values=past, use_cache=True)
            captures['clean-' + str(repeat)] = {'logits': cpu(result.logits[0, -1]), 'prefix': torch.tensor(expected), 'cache_before': before, 'cache_after_lengths': cpu(lengths), 'inherited_pointers': [int(x.data_ptr()) for x in old_data], 'tree_mask': None, 'position_ids': None}
            del result, past, data, lengths
        tensor_path = args.output / ('vectors-' + str(args.index) + '.pt')
        torch.save(captures, tensor_path)
        write(args.output / ('worker-' + str(args.index) + '.json'), {'status': 'complete', 'worker': args.index, 'cases_sha256': digest(args.prepared / 'cases.json'), 'script_sha256': digest(__file__), 'protocol_sha256': digest(Path(__file__).with_name('PREREGISTRATION.md')), 'case_index': args.index, 'probe': probe, 'outputs': outputs, 'capture': tensor_path.name, 'capture_sha256': digest(tensor_path), 'vocabulary_size': backend.model.base_model.config.vocab_size, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'gpu': torch.cuda.get_device_name(0), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'dtype': 'torch.float32', 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}, 'provenance': provenance})
    finally:
        backend.close()


def controller(args):
    from taps_rebuild.pool import allocation, worker_environment

    hardware = allocation(8)
    require(all(gpu['memory_total_mib'] >= 75000 for gpu in hardware['gpu_inventory']), 'FP32 confirmation requires eight 80GB GPUs')
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('confirmation_context', bundle / 'context.py')
    context = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context)
    policy = read(bundle / 'stale-context.json')
    require(policy['node'] == hardware['node'], 'Context policy belongs to another node')
    def query():
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=15)
        apps = [{'gpu_uuid': row[0].strip(), 'pid': int(row[1]), 'process_name': row[2].strip(), 'used_memory_mib': int(row[3])} for row in csv.reader(io.StringIO(text)) if row]
        text = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15)
        free = {row[0].strip(): int(row[1]) for row in csv.reader(io.StringIO(text)) if row}
        blocked = context.blocking_contexts(apps, free, policy, lambda pid: Path('/proc', str(pid)).exists(), context.host_proc_visible())
        (args.output / 'gpu-context.json').write_text(json.dumps({'applications': apps, 'blocked': blocked, 'free_memory_mib': free, 'policy': policy}, indent=2) + '\n')
        return str(blocked) if blocked else ''
    context.wait_for_idle(query, lambda _: None)
    write(args.output / 'manifest.json', {'hardware': hardware, 'prepared_manifest_sha256': digest(args.prepared / 'manifest.json'), 'cases_sha256': digest(args.prepared / 'cases.json'), 'files': {name: digest(bundle / name) for name in ('driver.py', 'PREREGISTRATION.md', 'context.py', 'stale-context.json')}, 'epsilon': 0.0001, 'retrospective_case_selection': True, 'training_admission': False})
    selected = read(args.prepared / 'cases.json')['eagle3']
    python = Path(selected['config']['legacy_runtime_root']) / 'envs/taps-eagle/bin/python'
    processes = []
    try:
        for binding in hardware['device_bindings']:
            index = binding['shard_index']
            env = worker_environment(binding['token'], Path(selected['authored_source_root']), args.root, python)
            env['PATH'] = str(python.parent) + ':/usr/local/bin:/usr/bin:/bin'
            log = (args.output / ('worker-' + str(index) + '.log')).open('xb')
            command = [str(python), '-B', str(Path(__file__).resolve()), '--root', str(args.root), '--prepared', str(args.prepared), '--output', str(args.output), '--index', str(index)]
            processes.append((subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT), log))
        codes = [process.wait() for process, _ in processes]
        require(not any(codes), 'Confirmation workers failed: ' + str(codes))
        write(args.output / 'completion.json', {'status': 'complete', 'workers': 8, 'captures': 48, 'training_admission': False})
    finally:
        for process, log in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    worker(args) if args.index is not None else controller(args)


if __name__ == '__main__':
    main()
