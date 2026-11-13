import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


V2_SHA256 = '5e11dd792adb840be1e861f6a0f2e235d483b977e7dca3cfef31f78e3f0f4091'
V2_PATH = Path(__file__).with_name('independent_pilot_qualification_audit_v2_20261112.py')
EXPORT_HELPER_SHA256 = '7e63cfa11907eed11d8abaad091b639071bd35b9ea13d178943765678e368aa1'
STOCK_CONTROLLER_SHA256 = '19951e9b498f06afe6f903d9057f5da2b4a5c1fe68c64bac281037a5c8c3e07e'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


if digest(V2_PATH) != V2_SHA256:
    raise ValueError('frozen_export_semantic_hasher_changed')
spec = importlib.util.spec_from_file_location('native_export_frozen_v2', V2_PATH)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
v1, require = v2.v1, v2.require


class Audit(v2.Audit):
    def run(self, preparation, receipt_path, stock, torch):
        self.phase = 'independent_native_export_raw_tensor_identity'
        queue = self.json(str(preparation / 'queue-plan.json'))
        receipt = self.json(str(receipt_path))
        require(receipt['status'] == 'complete_corrected_native_export_inventory_requires_stock_inference' and receipt['scientific_training_admitted'] is False and receipt['source_sha256'] == queue['source_sha256'], 'native_export_receipt_scope_mismatch')
        require(receipt['plan'] == queue['plans']['uninterrupted'] and receipt['helper']['sha256'] == EXPORT_HELPER_SHA256, 'native_export_plan_or_helper_not_reviewed')
        self.artifact(receipt['helper'])
        plan = self.json(receipt['plan'])
        require(receipt['plan_identity_sha256'] == plan['plan_identity_sha256'] and plan['config']['full_body_v3']['source_sha256'] == queue['source_sha256'], 'native_export_plan_source_mismatch')
        export = Path(receipt['export_directory'])
        require(export == Path(plan['trained_models']['resolved_path']) / 'budget_step_4', 'native_export_directory_not_registered')
        for name, item in receipt['files'].items():
            require(name in ('config.json', 'pytorch_model.bin'), 'native_export_file_inventory_unexpected')
            self.artifact({'path': str(export / name), **item})
        require(set(receipt['files']) == {'config.json', 'pytorch_model.bin'}, 'native_export_file_inventory_incomplete')
        state = torch.load(export / 'pytorch_model.bin', map_location='cpu', weights_only=True)
        manifest = self.json(receipt['raw_checkpoint_manifest'])
        raw_path = self.artifact(receipt['raw_model_source'])
        require(manifest['plan_identity_sha256'] == plan['plan_identity_sha256'] and manifest['files'][str(raw_path.relative_to(Path(receipt['raw_checkpoint_manifest']['path']).parent))] == receipt['raw_model_source']['sha256'], 'native_export_raw_manifest_binding_mismatch')
        raw = torch.load(raw_path, map_location='cpu', weights_only=False)
        require(raw['global_steps'] - raw['skipped_steps'] == 4, 'native_export_raw_not_four_successful_updates')
        names = {name for group in raw['param_shapes'] for name in group}
        require(len(names) == 13 and set(state) == names | {'d2t', 't2d'} and names == set(receipt['trainable_names']), 'native_export_complete_trainable_inventory_mismatch')
        require(all(bool(torch.isfinite(state[name]).all()) for name in names), 'native_export_trainable_nonfinite')
        expected = {name: raw['module'][name].to(state[name].dtype) for name in state}
        require(v2.semantic_identity(expected, torch) == v2.semantic_identity(state, torch), 'independent_native_export_raw_byte_identity_mismatch')
        actual_trainables = v2.semantic_identity({name: state[name] for name in names}, torch)
        require(actual_trainables == receipt['trainable_state_identity_sha256'], 'independent_native_export_trainable_identity_mismatch')
        cache_path = self.artifact({'path': str(Path(plan['workdir']) / 'cache.pt'), 'sha256': receipt['vocabulary_cache_sha256']})
        cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        require(all(v2.semantic_identity(state[name], torch) == v2.semantic_identity(cache[name], torch) for name in ('d2t', 't2d')), 'independent_native_export_vocabulary_bytes_differ')
        mapping = state['d2t'] + torch.arange(32000)
        require(state['d2t'].dtype == torch.long and state['t2d'].dtype == torch.bool and state['t2d'].shape == (128256,) and torch.equal(mapping, torch.nonzero(state['t2d'], as_tuple=True)[0]), 'native_export_vocabulary_geometry_invalid')
        del state, expected, raw, cache
        self.phase = 'independent_native_stock_actual_generation_source_inventory'
        launch = self.json(str(stock.parent / 'stock-launch.json'))
        require(launch['controller']['sha256'] == STOCK_CONTROLLER_SHA256 and launch['hardware']['node'] == 'deep-chungus-11' and len(launch['hardware']['device_bindings']) == 8, 'native_stock_controller_or_hardware_not_reviewed')
        self.artifact(launch['controller'])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plan['target']['path'], local_files_only=True)
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Write the next integer after one.'}], tokenize=True, add_generation_prompt=True)
        rows = []
        for rank in range(8):
            value = self.json(str(stock / ('rank-' + str(rank) + '.json')))
            require(value['status'] == 'passed_stock_corrected_native_export_generation' and value['rank'] == rank and value['plan_identity_sha256'] == plan['plan_identity_sha256'] and value['source_sha256'] == queue['source_sha256'], 'native_stock_actual_rank_plan_scope_mismatch')
            require(value['helper'] == receipt['helper'] and value['export_receipt'] == {'path': str(receipt_path), 'sha256': digest(receipt_path)}, 'native_stock_export_receipt_or_helper_mismatch')
            require(value['stock_source_files'], 'native_stock_source_inventory_empty')
            for name, expected_sha in value['stock_source_files'].items():
                require(not Path(name).is_absolute() and '..' not in Path(name).parts and name.startswith('eagle/model/'), 'native_stock_source_inventory_escape')
                self.artifact({'path': str(self.root / 'repos/EAGLE' / name), 'sha256': expected_sha})
            a, b = value['speculative_ids'], value['autoregressive_ids']
            require(value['prompt_ids'] == prompt and a[:len(prompt)] == b[:len(prompt)] == prompt and min(len(a), len(b)) > len(prompt), 'native_stock_did_not_generate_registered_prompt_continuations')
            require(all(type(token) is int and 0 <= token < 128256 for token in [*a, *b]) and value['tokens_equal'] == (a == b) and math.isfinite(value['generation_wall_seconds']) and value['generation_wall_seconds'] > 0, 'native_stock_generation_metadata_invalid')
            rows.append({'rank': rank, 'tokens_equal': value['tokens_equal'], 'speculative_continuation_tokens': len(a) - len(prompt), 'autoregressive_continuation_tokens': len(b) - len(prompt), 'wall_seconds': value['generation_wall_seconds']})
        return {'status': 'passed_independent_corrected_native_export_and_stock_compatibility', 'scope': 'complete_raw_model_export_vocabulary_identity_and_eight_rank_one_synthetic_prompt_compatibility_only', 'plan_identity_sha256': plan['plan_identity_sha256'], 'source_sha256': queue['source_sha256'], 'trainable_state_identity_sha256': actual_trainables, 'ranks': rows, 'hardware': launch['hardware'], 'limits': ['One synthetic prompt per rank is an engineering compatibility screen, not a quality, speedup or universal-losslessness result.', 'Different SD/AR length or token outputs are recorded honestly; this source does not register a common-budget greedy benchmark.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--stock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'native_export_independent_replay_requires_cpu_slurm')
    require(not args.output.exists(), 'native_export_independent_report_must_be_fresh')
    queue = json.loads((args.preparation / 'queue-plan.json').read_bytes())
    root = Path(json.loads(Path(queue['plans']['uninterrupted']['path']).read_bytes())['project_root']).resolve()
    import torch
    import deepspeed
    torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    audit, start = Audit(root), time.monotonic()
    try:
        report = audit.run(args.preparation.resolve(), args.receipt.resolve(), args.stock.resolve(), torch)
        audit.finish()
    except Exception as error:
        report = {'status': 'failed', 'phase': audit.phase, 'error_code': str(error) if isinstance(error, v1.AuditFailure) else type(error).__name__}
    report.update(scientific_training_admitted=False, automatic_admission_emitted=False, artifact_inventory=audit.inventory, artifact_inventory_sha256=v1.identity(audit.inventory), analyzer_sha256=digest(__file__), semantic_hasher_v2_sha256=V2_SHA256, slurm_job_id=os.environ['SLURM_JOB_ID'], wall_seconds=time.monotonic() - start, cpu_only=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': digest(args.output)}))
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
