import argparse
import importlib
import json
import os
from pathlib import Path

from taps_rebuild import native_training as native, pool
from taps_rebuild.native_gradients import install_checkpoint_policy, prime_rotary_cache
from taps_rebuild.pilot_cache import read, records, require
from taps_rebuild.pilot_gate import backend, devices, upstream, verify_cache
from taps_rebuild.pilot_study import mask_positions, verify_identity


def run(plan_path, output, restore_teacher_buffers=False, restore_preinitialization_buffers=False):
    import deepspeed
    import numpy as np
    import torch
    from accelerate.utils import set_seed

    require(os.environ.get('SLURM_JOB_ID') and int(os.environ['WORLD_SIZE']) == 8, 'Probe requires an owned eight-GPU allocation')
    plan = verify_identity(read(plan_path))
    rank = int(os.environ['LOCAL_RANK'])
    require(rank == int(os.environ['RANK']) and torch.cuda.device_count() == 8, 'Probe rank mapping differs')
    torch.cuda.set_device(rank)
    deepspeed.init_distributed(dist_backend='nccl')
    runtime, hardware = backend(torch), devices()
    require(runtime == plan['scan_runtime'] and hardware == plan['hardware'], 'Probe differs from registered eligibility runtime or hardware')
    require(all(pool.digest(Path(plan['source_directory']) / name) == value for name, value in plan['source_sha256'].items()), 'Registered pilot source changed')
    output = Path(output).resolve()
    require(output.is_relative_to(Path(plan['project_root']) / 'outputs/diagnostics'), 'Probe must use an isolated engineering output directory')
    output.mkdir(parents=True, exist_ok=True)
    destination = output / ('rank-' + str(rank) + '.json')
    require(not destination.exists(), 'Probe output already exists')
    cache = Path(plan['cache'])
    receipt = verify_cache(cache, rank)
    require(receipt['identity_sha256'] == plan['cache_identity_sha256'], 'Probe cache changed')
    eligibility = verify_identity(read(Path(plan['eligibility']) / 'completion.json'))
    require(eligibility['identity_sha256'] == plan['eligibility_identity_sha256'], 'Probe eligibility identity changed')
    eligible_path = Path(plan['eligibility']) / ('rank-' + str(rank) + '.eligible.jsonl')
    require(pool.digest(eligible_path) == eligibility['files'][eligible_path.name]['sha256'], 'Probe eligible rows changed')
    rows = {row['index']: row for row in records(eligible_path)}
    descriptor = plan['schedules'][rank]
    require(pool.digest(descriptor['path']) == descriptor['sha256'], 'Probe schedule changed')
    schedule = list(records(descriptor['path']))
    ids = np.memmap(cache / ('rank-' + str(rank) + '.ids'), mode='r', dtype='<i4', shape=(receipt['rank_examples'][rank], 2, 2048))
    source = upstream(plan['project_root'])
    set_seed(plan['seed'])
    cnets = importlib.import_module('cnets')
    native.install_model_adapter(cnets, {'draft': plan['ancestor'], 'workdir': str(output)}, torch)
    checkpoint_policy = install_checkpoint_policy('non_reentrant')
    config = importlib.import_module('configs').EConfig.from_pretrained(str(source / 'config.json'))
    training = {'bs': 1, 'num_epochs': 1, 'num_workers': 0, 'max_len': 2048, 'config_path': str(source / 'config.json'), 'gradient_checkpoint': True}
    model = cnets.Model(config, plan['recipe']['ds_config'], training, path=plan['target']['path'], load_emb=True, load_head=True)
    model.scandata(None, plan['target']['path'])
    checkpoint_policy['rotary_cache'] = prime_rotary_cache(model, torch)
    before_buffers = {name: {'dtype': str(value.dtype), 'shape': list(value.shape), 'identity': native.tensor_state_identity(value, torch)} for name, value in model.target_model.named_buffers()}
    pristine = None
    if restore_preinitialization_buffers:
        from taps_rebuild.pilot_train import teacher_buffer_snapshot
        pristine = teacher_buffer_snapshot(model.target_model, torch)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=plan['recipe']['ds_config'])
    engine.train()
    model.target_model.eval()
    after_buffers = {name: {'dtype': str(value.dtype), 'shape': list(value.shape), 'identity': native.tensor_state_identity(value, torch)} for name, value in model.target_model.named_buffers()}
    standalone = importlib.import_module('modeling_llama_kv').LlamaForCausalLM.from_pretrained(plan['target']['path'], torch_dtype=torch.float16).to(rank).eval()
    standalone_buffers = {name: {'dtype': str(value.dtype), 'shape': list(value.shape), 'identity': native.tensor_state_identity(value, torch)} for name, value in standalone.named_buffers()}
    restoration = None
    if restore_preinitialization_buffers:
        from taps_rebuild.pilot_train import restore_teacher_buffers as restore_pristine
        import taps_rebuild.pilot_train as producer
        restoration = {**restore_pristine(model.target_model, pristine, torch), 'producer_source': {'path': producer.__file__, 'sha256': pool.digest(producer.__file__)}}
        restored_buffers = {name: {'dtype': str(value.dtype), 'shape': list(value.shape), 'identity': native.tensor_state_identity(value, torch)} for name, value in model.target_model.named_buffers()}
        require(restored_buffers == standalone_buffers, 'Production preinitialization restoration differs from standalone buffers')
    if restore_teacher_buffers:
        require(set(standalone_buffers) == set(after_buffers), 'Teacher buffer name sets differ')
        for name, value in standalone.named_buffers():
            parent, leaf = name.rsplit('.', 1)
            model.target_model.get_submodule(parent)._buffers[leaf] = value.detach().clone()
        restored_buffers = {name: {'dtype': str(value.dtype), 'shape': list(value.shape), 'identity': native.tensor_state_identity(value, torch)} for name, value in model.target_model.named_buffers()}
        require(restored_buffers == standalone_buffers, 'Restored teacher buffers differ from exact standalone buffers')
        restoration = {'scope': 'Isolated engineering intervention after native initialization; no scientific source or run changed', 'buffers': restored_buffers, 'all_exact_standalone_buffers_restored': True}
    differences = []
    actual_parameters = dict(model.target_model.named_parameters())
    for name, value in standalone.named_parameters():
        other = actual_parameters[name]
        if value.dtype != other.dtype or value.shape != other.shape or not torch.equal(value, other):
            differences.append(name)
    cases = []
    attention = torch.ones((1, 2048), dtype=torch.long, device=rank)
    for step in (0, 63, 127, 206, 207):
        item = schedule[step]
        row = rows[item['index']]
        positions = mask_positions(row, plan['quota'], plan['seed'], item['occurrence'], plan['policy_index'])
        require(pool.identity(positions) == item['mask_sha256'][plan['policy_index']], 'Probe selected mask changed')
        tokens = np.array(ids[item['index'], plan['policy_index']], dtype='int64')
        inputs = torch.as_tensor(tokens, device=rank)[None]
        values = {}
        predictions = {}
        for label, teacher in (('standalone_scan', standalone), ('native_after_deepspeed', model.target_model)):
            require(not teacher.training and all(not module.training for module in teacher.modules()), 'Probe teacher is not in evaluation mode')
            torch.cuda.synchronize(rank)
            with torch.no_grad():
                result = teacher(input_ids=inputs, attention_mask=attention)
                selected = result.logits[0, [position + 1 for position in positions]]
                top = selected.topk(5, dim=-1)
                prediction = selected.argmax(-1)
                covered = model.t2d[prediction]
                require(bool(torch.isfinite(selected).all()), 'Probe selected logits are nonfinite')
                predictions[label] = prediction.detach().cpu()
                values[label] = {'argmax': predictions[label].tolist(), 'covered': covered.cpu().tolist(), 'uncovered_positions': [position for position, keep in zip(positions, covered.cpu().tolist()) if not keep], 'top5_ids': top.indices.cpu().tolist(), 'top5_logits': top.values.float().cpu().tolist(), 'selected_logits_identity': native.tensor_state_identity(selected, torch)}
                torch.save(selected.detach().cpu(), output / ('rank-' + str(rank) + '-step-' + str(step) + '-' + label + '.pt'))
                del result, selected, top
            torch.cuda.synchronize(rank)
        cases.append({'step': step, 'example_id': item['example_id'], 'index': item['index'], 'positions': positions, 'argmax_disagreements': int((predictions['standalone_scan'] != predictions['native_after_deepspeed']).sum()), 'predictions': values})
    result = {'status': 'completed_engineering_teacher_gate_probe', 'scientific_training_admitted': False, 'scientific_run_resumed': False, 'optimizer_updates': 0, 'rank': rank, 'plan': {'path': str(plan_path), 'sha256': pool.digest(plan_path)}, 'probe_sha256': pool.digest(__file__), 'hardware': hardware, 'runtime': runtime, 'checkpoint_policy': checkpoint_policy, 'teacher_parameter_differences': differences, 'native_before_deepspeed_buffers': before_buffers, 'native_after_deepspeed_buffers': after_buffers, 'standalone_scan_buffers': standalone_buffers, 'engineering_buffer_restoration': restoration, 'cases': cases}
    pool.atomic_json(destination, result)
    print(json.dumps({'rank': rank, 'status': result['status'], 'cases': [{'step': case['step'], 'disagreements': case['argmax_disagreements'], 'native_uncovered': case['predictions']['native_after_deepspeed']['uncovered_positions']} for case in cases]}), flush=True)
    torch.distributed.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--restore-teacher-buffers', action='store_true')
    parser.add_argument('--restore-preinitialization-buffers', action='store_true')
    args = parser.parse_args()
    require(not (args.restore_teacher_buffers and args.restore_preinitialization_buffers), 'Choose one engineering intervention')
    run(args.plan, args.output, args.restore_teacher_buffers, args.restore_preinitialization_buffers)
