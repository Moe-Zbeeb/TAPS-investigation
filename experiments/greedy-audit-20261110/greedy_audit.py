import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(root, output):
    runs = {'eagle3': 'official-eagle3-20261109-v2', 'eagle2': 'official-eagle2-hass-20261109-v1', 'hass': 'official-eagle2-hass-20261109-v1'}
    selected = {}
    for method, run_id in runs.items():
        pairs = {}
        directory = root / 'outputs/runs' / run_id
        for path in sorted((directory / 'raw').glob(method + '-shard*.jsonl')):
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['temperature'] != 0 or row['repeat'] != 0:
                        continue
                    key = (row['benchmark'], row['question_id'], row['turn_id'])
                    fields = ['input_token_ids', 'input_sha256', 'output_token_ids', 'comparison_token_ids', 'seed', 'case_id', 'benchmark', 'question_id', 'turn_id', 'target_model_identity_sha256', 'draft_model_identity_sha256', 'source_identity_sha256']
                    pairs.setdefault(key, {})[row['generation_mode']] = {k: row[k] for k in fields}
        candidates = {}
        for key, pair in pairs.items():
            ar, spec = pair['target_only'], pair['speculative']
            assert ar['input_token_ids'] == spec['input_token_ids']
            divergence = next((i for i, (a, b) in enumerate(zip(ar['comparison_token_ids'], spec['comparison_token_ids'])) if a != b), None)
            if divergence is not None and divergence > 0:
                candidates.setdefault(key[0], []).append({'divergence': divergence, 'pair': pair})
        for values in candidates.values():
            values.sort(key=lambda x: (x['divergence'], x['pair']['target_only']['question_id'], x['pair']['target_only']['turn_id']))
        chosen = []
        while len(chosen) < 8:
            for benchmark in sorted(candidates):
                if candidates[benchmark] and len(chosen) < 8:
                    chosen.append(candidates[benchmark].pop(0))
            assert any(candidates.values()) or len(chosen) == 8
        manifest = json.loads((directory / 'manifest.json').read_text())
        assets = json.loads(Path(manifest['asset_receipt_path']).read_text())
        sources = json.loads(Path(manifest['source_receipt_path']).read_text())
        config = manifest['effective_config']
        definition = config['methods'][method]
        selected[method] = {'run_id': run_id, 'manifest_sha256': hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest(), 'cases': chosen, 'definition': definition, 'config': config, 'target_path': assets['models'][definition['target']]['path'], 'draft_path': assets['models'][definition['drafter']]['path'], 'source_path': sources['sources'][definition['source']]['path'], 'authored_source_root': manifest['authored_source_root']}
    write(output / 'cases.json', selected)
    return selected


def worker(root, output, method, index, precision):
    import numpy
    import torch
    from taps_rebuild.backends.official import OfficialBackend
    from taps_rebuild.runner import seed_runtime

    selected = json.loads((output / 'cases.json').read_text())[method]
    case = selected['cases'][index]
    config, definition = selected['config'], selected['definition']
    backend = OfficialBackend(method, selected['source_path'], selected['target_path'], selected['draft_path'], total_token=config['total_token'], depth=definition['depth'], top_k=config['top_k'])
    if precision == 'float32':
        backend.close()
        arguments = dict(base_model_path=selected['target_path'], ea_model_path=selected['draft_path'], total_token=config['total_token'], depth=definition['depth'], top_k=config['top_k'], torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map={'': 0}, local_files_only=True)
        if method != 'hass':
            arguments['use_eagle3'] = method == 'eagle3'
        backend.model = backend.module.EaModel.from_pretrained(**arguments).eval()
    d = case['divergence']
    pair = case['pair']
    prompt = pair['target_only']['input_token_ids']
    expected_prefix = prompt + pair['target_only']['output_token_ids'][:d]
    records, vectors = {}, {}
    try:
        for mode in ('target_only', 'speculative'):
            captured = []
            original_forward = backend.model.base_model.forward
            original_tree = backend.module.tree_decoding
            original_posterior = backend.module.evaluate_posterior
            state = {'prefix': [], 'tree_prefix': None}

            def save(vector, prefix):
                if not captured:
                    captured.append((vector.detach().float().cpu(), prefix.detach().cpu().tolist()))

            def forward(*args, **kwargs):
                ids = args[0] if args else kwargs['input_ids']
                state['prefix'].extend(ids[0].detach().cpu().tolist())
                result = original_forward(*args, **kwargs)
                if len(state['prefix']) - len(prompt) == d:
                    save(result.logits[0, -1], torch.tensor(state['prefix']))
                return result

            def tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices):
                state['tree_prefix'] = input_ids
                return original_tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices)

            def posterior(logits, candidates, logits_processor):
                result = original_posterior(logits, candidates, logits_processor)
                best, accepted = int(result[0]), int(result[1])
                prefix = state['tree_prefix']
                committed = prefix.shape[1] - len(prompt)
                if committed < d <= committed + accepted + 1:
                    offset = d - committed
                    actual_prefix = torch.cat((prefix[0], candidates[best, :offset].to(prefix.device)))
                    save(logits[best, offset - 1], actual_prefix)
                return result

            if mode == 'target_only':
                backend.model.base_model.forward = forward
            else:
                backend.module.tree_decoding = tree
                backend.module.evaluate_posterior = posterior
            try:
                outputs = []
                for repeat in range(2):
                    state['prefix'] = []
                    seed_runtime(pair[mode]['seed'], torch, numpy)
                    result = backend.generate(torch.tensor([prompt]), temperature=0, max_new_tokens=min(config['max_new_tokens'], d + 4), max_length=config['max_length'], generation_mode=mode)
                    outputs.append(result['output_ids'][0, len(prompt):].cpu().tolist())
            finally:
                backend.model.base_model.forward = original_forward
                backend.module.tree_decoding = original_tree
                backend.module.evaluate_posterior = original_posterior
            record = {'replay_prefix_matches_original_through_divergence': outputs[0][:d + 1] == pair[mode]['output_token_ids'][:d + 1], 'two_replays_identical': outputs[0] == outputs[1], 'output_token_ids': outputs, 'captured': bool(captured)}
            if captured:
                vector, prefix = captured[0]
                values, indices = vector.topk(5)
                a, s = pair['target_only']['output_token_ids'][d], pair['speculative']['output_token_ids'][d]
                record.update(prefix_matches_original_common_prefix=prefix == expected_prefix, prefix_sha256=digest(prefix), top5_token_ids=indices.tolist(), top5_logits=values.tolist(), top2_margin=float(values[0] - values[1]), original_ar_minus_spec_logit=float(vector[a] - vector[s]))
                vectors[mode] = vector
            records[mode] = record
        comparison = {}
        if len(vectors) == 2:
            delta = vectors['target_only'] - vectors['speculative']
            comparison = {'max_abs_logit_delta': float(delta.abs().max()), 'mean_abs_logit_delta': float(delta.abs().mean()), 'same_captured_prefix': records['target_only']['prefix_sha256'] == records['speculative']['prefix_sha256'], 'argmax_equal': int(vectors['target_only'].argmax()) == int(vectors['speculative'].argmax())}
        receipt = {'status': 'complete', 'scope': 'selected_failure_diagnostic_not_benchmark', 'method': method, 'worker': index, 'precision': precision, 'gpu_name': torch.cuda.get_device_name(0), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'cases_sha256': hashlib.sha256((output / 'cases.json').read_bytes()).hexdigest(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'source_run_id': selected['run_id'], 'divergence': d, 'benchmark': pair['target_only']['benchmark'], 'question_id': pair['target_only']['question_id'], 'turn_id': pair['target_only']['turn_id'], 'records': records, 'comparison': comparison, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
        write(output / f'{method}-{precision}-{index}.json', receipt)
    finally:
        backend.close()


def controller(root, output):
    from taps_rebuild.pool import worker_environment

    assert os.environ['SLURM_JOB_ID'] == '2212576'
    inventory = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader'], text=True).strip().splitlines()
    assert len(inventory) == 8
    existing = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not existing
    output.mkdir(parents=True, exist_ok=False)
    script = output / 'greedy_audit.py'
    script.write_bytes(Path(__file__).read_bytes())
    selected = prepare(root, output)
    write(output / 'manifest.json', {'scope': 'selected_failure_diagnostic_not_benchmark', 'inventory': inventory, 'cases_sha256': hashlib.sha256((output / 'cases.json').read_bytes()).hexdigest(), 'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(), 'selection': 'Eight earliest repeat-zero token failures per method, round-robin over benchmarks; descriptive selection, no rate estimate', 'original_inference_inputs_modified': False, 'created_at_utc': datetime.now(timezone.utc).isoformat()})
    for precision in ('float16', 'float32'):
        for method, definition in selected.items():
            python = Path(definition['config']['legacy_runtime_root']) / 'envs' / ('taps-' + definition['definition']['profile']) / 'bin/python'
            processes = []
            for index in range(8):
                env = worker_environment(str(index), Path(definition['authored_source_root']), root, python)
                env['PATH'] = str(python.parent) + ':/usr/local/bin:/usr/bin:/bin'
                log = (output / f'{method}-{precision}-{index}.log').open('xb')
                args = [str(python), '-B', str(script), '--root', str(root), '--output', str(output), '--method', method, '--index', str(index), '--precision', precision]
                process = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                processes.append((process, log))
            codes = []
            for process, log in processes:
                codes.append(process.wait())
                log.close()
            if any(codes):
                raise RuntimeError(f'Worker wave failed: {method} {precision} {codes}')
            print(json.dumps({'wave_complete': method, 'precision': precision}), flush=True)
    write(output / 'completion.json', {'status': 'complete', 'workers': 48, 'paired_replays': 96, 'benchmark_claim': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['eagle3', 'eagle2', 'hass'])
    parser.add_argument('--index', type=int)
    parser.add_argument('--precision', choices=['float16', 'float32'])
    args = parser.parse_args()
    try:
        if args.method:
            worker(args.root, args.output, args.method, args.index, args.precision)
        else:
            controller(args.root, args.output)
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
