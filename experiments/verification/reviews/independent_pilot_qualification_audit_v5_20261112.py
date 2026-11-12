import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


V4_SHA256 = '592ab1fcc7592b1b3d36e758126d3177b1a1dd34f6d119077876e180e8284a05'
V4_PATH = Path(__file__).with_name('independent_pilot_qualification_audit_v4_20261112.py')


def digest(path):
    import hashlib
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


if digest(V4_PATH) != V4_SHA256:
    raise ValueError('frozen_v4_analyzer_changed')
spec = importlib.util.spec_from_file_location('frozen_independent_pilot_v4', V4_PATH)
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)
v3, v2, v1 = v4.v3, v4.v2, v4.v1
require = v1.require


def reviewed_checkpoint_file(path, torch_root, environment_root, expected_sha256):
    return path == torch_root / 'utils/checkpoint.py' and path.is_relative_to(environment_root) and expected_sha256 == v3.CHECKPOINT_SHA256


class Audit(v4.Audit):
    def artifact(self, descriptor):
        if isinstance(descriptor, str):
            return super().artifact(descriptor)
        path = Path(descriptor['path'])
        path = (self.root / path if not path.is_absolute() else path).resolve()
        import torch
        torch_root = Path(torch.__file__).resolve().parent
        environment_root = (self.root / 'envs/native-eagle3').resolve()
        if reviewed_checkpoint_file(path, torch_root, environment_root, descriptor.get('sha256')):
            actual = {'sha256': digest(path), 'size_bytes': path.stat().st_size}
            require(actual['sha256'] == v3.CHECKPOINT_SHA256 and ('size_bytes' not in descriptor or descriptor['size_bytes'] == actual['size_bytes']), 'reviewed_runtime_checkpoint_file_changed')
            require(str(path) not in self.inventory or self.inventory[str(path)] == actual, 'runtime_checkpoint_file_changed_during_audit')
            self.inventory[str(path)] = actual
            return path
        return super().artifact(descriptor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--runtime-pre', type=Path)
    parser.add_argument('--runtime-post', type=Path)
    parser.add_argument('--execution-envelope', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'v5_qualification_audit_requires_cpu_slurm')
    study = args.study / 'manifest.json' if args.study.is_dir() else args.study
    root = Path(json.loads(study.read_bytes())['common']['project_root']).resolve()
    qualification = args.qualification.resolve()
    output = args.output.resolve() if args.output.suffix == '.json' else args.output.resolve() / 'report.json'
    require(output.is_relative_to(root / 'outputs') and not output.exists(), 'choose_fresh_v5_report_under_outputs')
    import torch
    import deepspeed
    torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    audit, start = Audit(root), time.monotonic()
    try:
        request = audit.automatic_request(args.study.resolve(), qualification)
        schedules = audit.schedules(request['study_manifest'])
        callbacks = audit.checkpoint_policies(qualification)
        gradients = audit.gradients(request['gradient_cases'], torch)
        traces = audit.traces(request['state_comparisons'])
        suffix = audit.suffix(qualification, torch)
        raw = audit.raw_qualifications(qualification, torch)
        stock = audit.stock(qualification, torch)
        pre = args.runtime_pre or qualification / 'runtime/pre.json'
        post = args.runtime_post or qualification / 'runtime/post.json'
        interval = args.execution_envelope or qualification / 'runtime/interval-accounting.json'
        runtime = audit.runtime_envelope(pre if pre.is_file() else None, post if post.is_file() else None, interval if interval.is_file() else None)
        checks = {'schedules': schedules['status'], 'ce_gradients': gradients['status'], 'outside_mask_gradients': gradients['outside_mask_status'], **traces['statuses'], 'suffix_probe': suffix['status'], 'raw_zero_state_reconstruction': raw['status'], 'all13_named_hooks_and_corrected_policy': callbacks['status'], 'rotary_priming': callbacks['rotary_priming']['status'], 'all13_named_adam_and_fp32_updates': 'passed' if raw['status'] == 'passed' and all(value.get('all13_named_adam_and_master_update', {}).get('status') == 'passed' for value in raw['checkpoints']) else 'partial_evidence', 'stock_export_compatibility': stock['status'], 'runtime_envelope': runtime['status']}
        report = {'status': 'all_checked_evidence_passed_requires_independent_review' if all(value == 'passed' for value in checks.values()) else 'partial_evidence', 'scope': 'independent_primed_full_body_cpu_pilot_qualification_analysis_not_admission', 'checks': checks, 'schedules': schedules, 'checkpoint_and_named_hook_coverage': callbacks, 'gradients': gradients, 'state_comparisons': traces, 'suffix': suffix, 'raw_states': raw, 'stock': stock, 'runtime': runtime}
        audit.finish()
    except Exception as error:
        report = {'status': 'failed', 'phase': audit.phase, 'error_code': str(error) if isinstance(error, v1.AuditFailure) else type(error).__name__}
    report.update(scientific_training_admitted=False, automatic_admission_emitted=False, artifact_inventory=audit.inventory, artifact_inventory_sha256=v1.identity(audit.inventory), analyzer_v1_sha256=v2.V1_SHA256, analyzer_v2_sha256=v3.V2_SHA256, analyzer_v3_sha256=v4.V3_SHA256, analyzer_v4_sha256=V4_SHA256, analyzer_v5_sha256=digest(Path(__file__)), verifier_change_scope='Allow only the exact hash-reviewed Torch checkpoint source file resolved from the imported Torch package within the declared native environment; no project/data/model allowlist expansion.', slurm_job_id=os.environ['SLURM_JOB_ID'], cpu_only=True, wall_seconds=time.monotonic() - start, project_helpers_imported=False, runtime_packages={name: importlib.metadata.version(name) for name in ('torch', 'deepspeed', 'numpy', 'transformers', 'accelerate')})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': digest(output), 'output': str(output)}), flush=True)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
