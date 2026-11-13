import argparse
import ast
import hashlib
import os
from pathlib import Path
import sys


def run(root):
    bundle = Path(__file__).resolve().parent
    sys.path.insert(0, str(bundle / 'src'))
    from taps_rebuild import pool
    from taps_rebuild.pilot_cache import read, require
    from taps_rebuild.pilot_queue import verify_bundle
    from taps_rebuild.pilot_study import prepare, verify_identity

    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Teacher study preparation requires CPU Slurm')
    require(verify_bundle()[0] == bundle, 'Teacher preparation source bundle differs')
    review = read(bundle / 'source-review.json')
    require(pool.digest(bundle / 'source-review.json') == 'fa1485adffbd5494c196ba08efc204448af2da758511fe285f314ea4ced3162c' and review['decision'] == 'approved_for_fresh_bounded_pilot_engineering_only' and review['scientific_training_admitted'] is False, 'Exact independent bounded source approval differs')
    for name, expected in review['source_sha256'].items():
        require(pool.digest(bundle / name) == expected, 'Reviewed teacher policy source changed: ' + name)
    require(pool.digest(bundle / 'qualify_teacher_buffers_20261113.py') == review['source_sha256']['experiments/verification/qualify_teacher_buffers_20261113.py'], 'Executed qualification wrapper differs from exact source review')
    proof = Path(root) / 'outputs/diagnostics/pilot-teacher-gate-20261113-v3'
    collector = Path(root) / 'outputs/preparation/pilot-teacher-gate-probe-20261113-v3/pilot_teacher_gate_probe_20261113.py'
    collector_sha256 = '1f3ab82da730bac154d34909101624aa0d912b6bd5d7593ded4bdbb8b770ab2c'
    launch = read(proof / 'launch.json')
    require(pool.digest(collector) == collector_sha256 == launch['probe_sha256'] and '--restore-preinitialization-buffers' in launch['command'], 'Exact-helper executed collector source or intervention differs')
    completion = read(proof / 'completion.json')
    require(completion['status'] == 'completed_eight_rank_engineering_probe' and completion['optimizer_updates'] == 0 and completion['scientific_training_admitted'] is False and completion['scientific_run_resumed'] is False and [item['rank'] for item in completion['ranks']] == list(range(8)), 'Exact-helper engineering probe is incomplete')
    old = verify_identity(read(Path(root) / 'outputs/pilot-training/rq1-pilot-24-20261112-v2/manifest.json'))
    require(pool.digest(old['plans'][0]['path']) == old['plans'][0]['sha256'], 'Original failed plan bytes differ from registered manifest')
    failing = verify_identity(read(old['plans'][0]['path']))
    require(failing['cohort'] == 'code' and failing['policy'] == 'original' and failing['seed'] == 0 and failing['quota'] == 64, 'Original failed condition differs')
    current_ast = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse((bundle / 'src/taps_rebuild/pilot_train.py').read_bytes()).body if isinstance(node, ast.FunctionDef) and node.name in review['teacher_buffer_helper_ast_sha256']}
    require(set(current_ast) == set(review['teacher_buffer_helper_ast_sha256']), 'Final reviewed helper inventory differs')
    evidence = {}
    for item in completion['ranks']:
        rank = item['rank']
        path = proof / ('rank-' + str(rank) + '.json')
        require(pool.digest(path) == item['sha256'], 'Exact-helper rank evidence changed')
        value = read(path)
        require(value['probe_sha256'] == collector_sha256, 'Exact-helper rank collector source differs')
        restoration = value['engineering_buffer_restoration']
        producer = Path(restoration['producer_source']['path'])
        require(producer.resolve() == Path(root) / 'outputs/preparation/pilot-teacher-gate-probe-20261113-v3/src/taps_rebuild/pilot_train.py' and pool.digest(producer) == restoration['producer_source']['sha256'], 'Exact-helper executed producer differs')
        executed_ast = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(producer.read_bytes()).body if isinstance(node, ast.FunctionDef) and node.name in current_ast}
        require(executed_ast == current_ast, 'Final production helpers differ from actual GPU probe')
        require(value['rank'] == rank and value['optimizer_updates'] == 0 and value['scientific_training_admitted'] is False and value['scientific_run_resumed'] is False and value['hardware'] == failing['hardware'] and value['runtime'] == failing['scan_runtime'] and value['plan'] == {key: old['plans'][0][key] for key in ('path', 'sha256')} and not value['teacher_parameter_differences'], 'Exact-helper runtime/model/study identity differs')
        require(value['native_before_deepspeed_buffers'] == value['standalone_scan_buffers'] and restoration['policy'] == 'exact_pre_deepspeed_fp32_teacher_buffers' and restoration['pristine_buffer_identity_sha256'] == restoration['restored_buffer_identity_sha256'] and restoration['teacher_parameters_remain_frozen'] is True and restoration['teacher_remains_in_evaluation_mode'] is True, 'Exact-helper pristine buffer evidence differs')
        require([case['step'] for case in value['cases']] == [0, 63, 127, 206, 207], 'Exact-helper failing/control cases differ')
        for case in value['cases']:
            left, right = [case['predictions'][name] for name in ('standalone_scan', 'native_after_deepspeed')]
            require(len(case['positions']) == failing['quota'] and case['argmax_disagreements'] == 0 and not left['uncovered_positions'] and not right['uncovered_positions'] and left['selected_logits_identity'] == right['selected_logits_identity'], 'Exact-helper complete selected logits differ')
        evidence[str(path)] = item['sha256']
    output = Path(root) / 'outputs/pilot-training' / review['study_id']
    manifest = prepare(root, failing['cache'], failing['eligibility'], bundle / 'configs/rq1-pilot-24-20261113.json', output)
    require(manifest['common']['cache_identity_sha256'] == failing['cache_identity_sha256'] and manifest['common']['eligibility_identity_sha256'] == failing['eligibility_identity_sha256'] and manifest['common']['hardware'] == failing['hardware'] and manifest['common']['quota'] == failing['quota'], 'Fresh study changed sealed data, hardware orQ')
    old_schedules = {name: item['sha256'] for name, item in old['files'].items() if name.startswith('schedules/')}
    new_schedules = {name: item['sha256'] for name, item in manifest['files'].items() if name.startswith('schedules/')}
    require(old_schedules == new_schedules and len(new_schedules) == 96, 'Fresh study changed any exact schedule or mask')
    receipt = {'status': 'prepared_fresh_teacher_policy_study_for_bounded_engineering', 'scientific_training_admitted': False, 'failed_scientific_run_resumed': False, 'source_review_sha256': pool.digest(bundle / 'source-review.json'), 'exact_gpu_helper_ast_sha256': {name: hashlib.sha256(value.encode()).hexdigest() for name, value in current_ast.items()}, 'reviewed_helper_ast_sha256_python314': review['teacher_buffer_helper_ast_sha256'], 'helper_equivalence': 'Exact independently reviewed full worker bytes plus same-interpreter AST equality between retained executed GPU helpers and final worker; AST display hashes from different Python versions are retained separately', 'all40_complete_selected_logit_identities_match': True, 'probe_completion_sha256': pool.digest(proof / 'completion.json'), 'probe_rank_json_sha256': evidence, 'all96_schedule_file_sha256_unchanged': True, 'manifest': {'path': str(output / 'manifest.json'), 'sha256': pool.digest(output / 'manifest.json')}}
    pool.atomic_json(bundle / 'preparation.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(run(args.root)))
