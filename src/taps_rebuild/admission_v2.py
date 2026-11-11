import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

from . import admission as legacy
from . import numerical_validation


CHECKS = {'conditional', 'sampling', 'complete_registered_population', 'common_budget_panel', 'numerical_classification', 'live_source_model_content'}
COVERAGE = {'conditional_states': 60, 'conditional_draws': 7864320, 'conditional_events': 960, 'sampling_continuations': 61440, 'sampling_events': 36, 'greedy_pairs': 1120, 'greedy_exact_pairs': 1119, 'classified_first_divergences': 1, 'numerical_cases': 8, 'retained_logit_vectors': 48}
FORMAT = 'taps-pipeline-replay-v2'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt_inputs(prepared):
    from transformers import AutoTokenizer

    inputs, vocabularies = {}, {}
    for method, selected in prepared['methods'].items():
        tokenizer = AutoTokenizer.from_pretrained(selected['models']['target']['path'], local_files_only=True)
        vocabularies[method] = len(tokenizer)
        for index, prompt in enumerate(selected['sampling_prompts']):
            messages = [{'role': 'system', 'content': selected['source']['system_prompt']['text']}, {'role': 'user', 'content': prompt['text']}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs[method, index] = tokenizer([text], add_special_tokens=False)['input_ids'][0]
    return inputs, vocabularies


def strict_conditional(row, capture, driver, stops, expected_input, vocabulary):
    import torch

    weights, logits, candidates, prefix = (capture[key] for key in ('native_ar_weights', 'tree_logits', 'candidates', 'prefix'))
    legacy.require(weights.shape == (vocabulary,) and bool(torch.isfinite(weights).all()) and bool((weights >= 0).all()) and float(weights.double().sum()) > 0, 'Invalid original AR sampling weights')
    legacy.require(capture['normalized_ar_q'].shape == (vocabulary,) and logits.ndim == 3 and logits.shape[:2] == candidates.shape and logits.shape[2] == vocabulary and bool(torch.isfinite(logits).all()), 'Conditional tree logits have invalid vocabulary/shape/values')
    legacy.require(candidates.dtype == torch.long and bool(((candidates >= 0) & (candidates < vocabulary) | (candidates == -1)).all()) and bool((candidates[:, 0] >= 0).all()), 'Invalid conditional candidate token')
    legacy.require(prefix.ndim == 2 and prefix.shape[0] == 1 and prefix.dtype == torch.long and prefix.shape[1] > len(expected_input) and prefix[0, :len(expected_input)].tolist() == expected_input and int(prefix[0, -1]) == int(candidates[0, 0]) and bool(((prefix >= 0) & (prefix < vocabulary)).all()), 'Conditional prefix is not rooted in the frozen rendered prompt')
    tree = torch.softmax(logits[0, 0], dim=0).double()
    tree /= tree.sum()
    tv = float((tree - capture['normalized_ar_q']).abs().sum() / 2)
    legacy.require(abs(tv - row['tree_vs_ar_full_vocabulary_tv']) < 1e-12, 'Conditional tree-versus-AR TV does not reconstruct')
    return legacy.conditional_checks(row, capture, driver, stops)


def sampling_inputs(screen, inputs, vocabularies, stops, bind):
    checked = 0
    for path in sorted(Path(screen).glob('sampling-*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            method, index = row['method'], row['prompt_index']
            legacy.require(row['input_token_ids'] == inputs[method, index], 'End-to-end input differs from the independently rendered registered prompt')
            raw, tokens = row['output_token_ids'], row['comparison_token_ids']
            legacy.require(all(type(token) is int and 0 <= token < vocabularies[method] for token in raw), 'Invalid end-to-end vocabulary ID')
            legacy.require(len(raw) >= 32 or bool(tokens) and tokens[-1] in stops[method], 'Short unterminated end-to-end continuation')
            checked += 1
        bind(path, 'sampling_raw')
        bind(path.with_suffix('.receipt.json'), 'sampling_receipt')
    legacy.require(checked == COVERAGE['sampling_continuations'], 'Incomplete end-to-end raw population')


def historical_stop_ids(prepared):
    from transformers import AutoTokenizer

    values = {}
    for method, selected in prepared['methods'].items():
        tokenizer = AutoTokenizer.from_pretrained(selected['models']['target']['path'], local_files_only=True)
        eos = tokenizer.eos_token_id
        values[method] = {value for value in ([*eos] if isinstance(eos, (list, tuple)) else [eos]) + [tokenizer.convert_tokens_to_ids('<|eot_id|>')] if type(value) is int and value >= 0}
    return values


def common_budget(root, prepared, directory, bundle, bind):
    driver = load(Path(bundle) / 'driver.py', 'pipeline_common_budget_control')
    args = argparse.Namespace(root=Path(root), prepared=Path(prepared).parent, output=Path(directory), report_name='pipeline-v2-replay.json')
    report = driver.replay(args)
    legacy.require(report['pairs'] == 1120 and report['exact_mismatches'] == 1, 'Common-budget population or exact count differs')
    failures = report['failures']
    legacy.require(len(failures) == 1, 'Additional unclassified greedy failure')
    failure = failures[0]
    legacy.require(failure['method'] == 'eagle3' and failure['benchmark'] == 'mt_bench' and failure['question_id'] == 146 and failure['turn_id'] == 1 and failure['first_token_difference'] == 200 and failure['input_sha256'] == numerical_validation.INPUT_SHA256, 'Greedy failure differs from the registered numerical site')
    for name in ('driver.py', 'collector.py', 'validation_budget.py', 'PREREGISTRATION.md', 'context.py', 'stale-context.json'):
        bind(Path(bundle) / name, 'stopping_source')
    for path in Path(directory).glob('greedy-*.jsonl'):
        bind(path, 'greedy_raw')
        bind(path.with_suffix('.receipt.json'), 'greedy_receipt')
    for name in ('manifest.json', 'completion.json', 'report.json'):
        bind(Path(directory) / name, 'stopping_manifest')
    return report


def content_inventory(root, prepared, bind):
    models, sources = {}, {}
    for method, selected in prepared['methods'].items():
        manifest = legacy.read(selected['run_manifest'])
        bind(selected['run_manifest'], 'run_provenance')
        bind(selected['run_config'], 'run_provenance')
        for field in ('asset_receipt_path', 'source_receipt_path'):
            legacy.require(legacy.digest(manifest[field]) == manifest[field.replace('_path', '_sha256')], 'Collection-time provenance receipt changed')
            bind(manifest[field], 'run_provenance')
        for name, expected in manifest['authored_source_hashes'].items():
            path = Path(manifest['authored_source_root']) / name
            bind(path, 'authored_source', expected)
        for role, model in selected['models'].items():
            for name, record in model['files'].items():
                relative = Path(name)
                legacy.require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid registered model path')
                bind(Path(model['path']) / relative, role + '_model', record['sha256'])
            models[method + ':' + role] = model['model_identity_sha256']
        source = selected['source']
        for name, record in source['tracked_files'].items():
            relative = Path(name)
            legacy.require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid registered upstream source path')
            path = Path(source['path']) / relative
            if record['kind'] == 'symlink':
                legacy.require(path.is_symlink() and legacy.hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest() == record['sha256'], 'Registered upstream symlink changed')
                bind(path, 'upstream_source')
            else:
                legacy.require(record['kind'] == 'regular_file' and not path.is_symlink(), 'Registered upstream file type changed')
                bind(path, 'upstream_source', record['sha256'])
        sources[method] = source['source_identity_sha256']
    return {'models': models, 'sources': sources, 'collection_time_model_content_reverified': False, 'admission_time_full_content_verified': True, 'runtime_scope': 'Retained raw manifests, versions, native dtype, batch-one kernels and eight-A100 bindings only; no portability or serving admission'}


def audit(root, config):
    legacy.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Pipeline reconstruction requires CPU Slurm')
    root = Path(root).resolve()
    allowed = (root, (root / 'models').resolve())
    artifacts = {}
    def bind(path, category, expected=None):
        path = Path(path).resolve()
        legacy.require(any(path.is_relative_to(base) for base in allowed), 'Bound artifact escaped registered project/model storage')
        key = str(path)
        if key not in artifacts:
            before = legacy.stamp(path)
            artifacts[key] = {'path': key, 'sha256': legacy.digest(path), 'stat': before, 'categories': []}
            legacy.require(legacy.stamp(path) == before, 'Artifact changed during content hashing')
        item = artifacts[key]
        legacy.require(expected is None or item['sha256'] == expected, 'Live source/model content differs from registered SHA256: ' + key)
        if category not in item['categories']:
            item['categories'].append(category)
    p = lambda name: root / config[name]
    prepared = legacy.read(p('prepared'))
    legacy.require(legacy.digest(p('prepared')) == legacy.PREPARED_SHA256 and legacy.digest(p('driver')) == legacy.DRIVER_SHA256, 'Original raw registration changed')
    inputs, vocabularies = prompt_inputs(prepared)
    stops = historical_stop_ids(prepared)
    original = legacy.conditional_checks
    def strengthened(row, capture, driver, stops):
        function = legacy.conditional_checks
        legacy.conditional_checks = original
        try:
            return strict_conditional(row, capture, driver, stops, inputs[row['method'], row['prompt_index']], vocabularies[row['method']])
        finally:
            legacy.conditional_checks = function
    legacy.conditional_checks = strengthened
    try:
        replay = legacy.audit(root, p('driver'), p('prepared'), p('primary'), p('screen'), p('proof'), phases=('conditional', 'sampling'))
    finally:
        legacy.conditional_checks = original
    for item in replay['artifacts']:
        category = 'conditional_vectors' if item['path'].endswith('.pt') else 'registered_evidence'
        bind(item['path'], category, item['sha256'])
    sampling_inputs(p('screen'), inputs, vocabularies, stops, bind)
    for path in p('primary').glob('conditional-*.jsonl'):
        bind(path, 'conditional_raw')
        bind(path.with_suffix('.receipt.json'), 'conditional_receipt')
    panel = common_budget(root, p('prepared'), p('stopping_output'), p('stopping_bundle'), bind)
    numerical = numerical_validation.replay(root, p('numerical_prepared'), p('numerical_output'), p('numerical_bundle'), bind, prepared)
    provenance = content_inventory(root, prepared, bind)
    for key in ('proof', 'protocol', 'sampling_proposal', 'policy', 'historical_report'):
        bind(p(key), key)
    legacy.require(legacy.digest(p('protocol')) == prepared['protocol_sha256'] and legacy.digest(p('sampling_proposal')) == prepared['sampling_proposal_sha256'], 'Controlling registered protocol/proposal changed')
    for path in (Path(__file__), Path(legacy.__file__), Path(numerical_validation.__file__)):
        bind(path, 'validator_source')
    bind(p('prepared'), 'registered_inputs', legacy.PREPARED_SHA256)
    bind(p('driver'), 'registered_collector', legacy.DRIVER_SHA256)
    bind(p('native_training'), 'training_source')
    checks = {'conditional': replay['checks']['conditional'], 'sampling': replay['checks']['sampling'], 'complete_registered_population': True, 'common_budget_panel': True, 'numerical_classification': numerical['status'] == 'passed', 'live_source_model_content': True}
    report = {'format': FORMAT, 'status': 'replay_passed_review_required' if all(checks.values()) else 'blocked', 'scope': 'Native released Llama8B EAGLE3/EAGLE2/HASS pipeline evidence; scratch native EAGLE3 generalist training only; no universal lossless/serving/pooled admission', 'checks': checks, 'coverage': COVERAGE, 'paper_matching_required': False, 'historical_report': {'path': str(p('historical_report')), 'sha256': legacy.digest(p('historical_report')), 'agreement_relabelled': False}, 'numerical': numerical, 'greedy': panel, 'provenance': provenance, 'rendered_prompt_identities': {method + ':' + str(index): legacy.identity(value) for (method, index), value in inputs.items()}, 'input_freeze_timing': 'Retrospective tokenizer/system reconstruction of retained draws, not a pre-draw token-ID freeze', 'validator_sha256': legacy.digest(__file__), 'numerical_validator_sha256': legacy.digest(numerical_validation.__file__), 'native_training_sha256': legacy.digest(p('native_training')), 'artifacts': sorted(artifacts.values(), key=lambda item: item['path']), 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'scientific_training_authorized': False}
    report['artifact_categories'] = dict(sorted(Counter(category for item in report['artifacts'] for category in item['categories']).items()))
    report['identity_sha256'] = legacy.identity(report)
    return report


def verify_certificate(root, report, review_descriptor):
    root = Path(root).resolve()
    legacy.require(report.get('format') == FORMAT and report.get('status') == 'replay_passed_review_required' and report.get('coverage') == COVERAGE and report.get('paper_matching_required') is False and set(report.get('checks', {})) == CHECKS and all(value is True for value in report['checks'].values()), 'Complete source-bound v2 CPU replay is required')
    legacy.require(legacy.identity({key: value for key, value in report.items() if key != 'identity_sha256'}) == report.get('identity_sha256') and report['validator_sha256'] == legacy.digest(__file__) and report['numerical_validator_sha256'] == legacy.digest(numerical_validation.__file__), 'Pipeline replay identity or verifier bytes changed')
    path = (root / review_descriptor['path']).resolve()
    legacy.require(path.is_relative_to(root) and legacy.digest(path) == review_descriptor['sha256'], 'Pinned independent review artifact changed')
    review = legacy.read(path)
    legacy.require(review.get('decision') == 'approved' and review.get('evidence_admission') is True and review.get('reviewer') == 'independent-agent-validation-review' and review.get('evidence_identity_sha256') == report['identity_sha256'] and review.get('validator_sha256') == report['validator_sha256'] and review.get('numerical_validator_sha256') == report['numerical_validator_sha256'] and review.get('native_training_sha256') == report['native_training_sha256'], 'Independent evidence-level review does not approve this exact replay and implementation')
    expected = Counter(category for item in report['artifacts'] for category in item['categories'])
    legacy.require(dict(sorted(expected.items())) == report['artifact_categories'] and len({item['path'] for item in report['artifacts']}) == len(report['artifacts']), 'Artifact category inventory is inconsistent')
    for category, count in {'conditional_raw': 24, 'conditional_receipt': 24, 'conditional_vectors': 60, 'sampling_raw': 24, 'sampling_receipt': 24, 'greedy_raw': 24, 'greedy_receipt': 24, 'numerical_worker': 8, 'numerical_vectors': 8, 'validator_source': 3, 'training_source': 1, 'proof': 1, 'protocol': 1, 'sampling_proposal': 1, 'policy': 1, 'historical_report': 1}.items():
        legacy.require(expected[category] == count, 'Incomplete required artifact category: ' + category)
    legacy.require(expected['target_model'] >= 10 and expected['drafter_model'] >= 3 and expected['upstream_source'] >= 20 and expected['authored_source'] >= 20 and expected['run_provenance'] >= 4, 'Incomplete actual source/model/tokenizer inventory')
    registered = [item for item in report['artifacts'] if 'registered_inputs' in item['categories']]
    collectors = [item for item in report['artifacts'] if 'registered_collector' in item['categories']]
    legacy.require(len(registered) == len(collectors) == 1 and registered[0]['sha256'] == legacy.PREPARED_SHA256 and collectors[0]['sha256'] == legacy.DRIVER_SHA256, 'Missing original hash-pinned prepared population/collector')
    prepared = legacy.read(registered[0]['path'])
    inventory = {item['path']: item for item in report['artifacts']}
    for method, selected in prepared['methods'].items():
        for role, model in selected['models'].items():
            legacy.require(report['provenance']['models'][method + ':' + role] == model['model_identity_sha256'], 'Model identity differs from fixed registration')
            for name, record in model['files'].items():
                item = inventory.get(str((Path(model['path']) / name).resolve()), {})
                legacy.require(item.get('sha256') == record['sha256'] and role + '_model' in item.get('categories', []), 'Missing registered model/tokenizer content')
        for name, record in selected['source']['tracked_files'].items():
            path = Path(selected['source']['path']) / name
            item = inventory.get(str(path.resolve()), {})
            legacy.require('upstream_source' in item.get('categories', []) and (record['kind'] == 'symlink' and path.is_symlink() and legacy.hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest() == record['sha256'] or record['kind'] == 'regular_file' and item.get('sha256') == record['sha256']), 'Missing registered upstream content')
    allowed = (root, (root / 'models').resolve())
    for item in report['artifacts']:
        path = Path(item['path']).resolve()
        legacy.require(any(path.is_relative_to(base) for base in allowed) and legacy.digest(path) == item['sha256'], 'Admission artifact content changed after CPU replay: ' + str(path))
    legacy.require(review.get('artifact_inventory_sha256') == legacy.identity(report['artifacts']), 'Independent review did not bind the full evidence inventory')
    return {'status': 'passed', 'scope': report['scope'], 'scientific_training_authorized': True, 'certificate_identity_sha256': report['identity_sha256'], 'independent_review_sha256': review_descriptor['sha256'], 'paper_matching_required': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, legacy.read(args.config))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
