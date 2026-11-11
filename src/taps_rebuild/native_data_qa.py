import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from . import native_training as native


class VocabularyContext:
    def __init__(self, vocabulary, draft_vocabulary):
        self.vocab_size, self.draft_vocab_size = vocabulary, draft_vocabulary
        self.train_config = SimpleNamespace(max_len=2048)

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)


def audit(root, output, *, corpus=None, engineering_fixture=None):
    native.require_slurm()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) < 48:
        raise ValueError('Exact native vocabulary preparation requires CPU Slurm with at least 48 CPUs')
    root = Path(root).resolve()
    output = Path(output).resolve()
    corpus = Path(corpus).resolve() if corpus is not None else None
    engineering_fixture = Path(engineering_fixture).resolve() if engineering_fixture is not None else None
    if not output.is_relative_to(root / 'outputs'):
        raise ValueError('Native data audit outputs belong under project outputs')
    output.mkdir(parents=True, exist_ok=True)
    attempt = output / ('attempt-' + os.environ.get('SLURM_RESTART_COUNT', '0'))
    attempt.mkdir(exist_ok=False)
    work = attempt / 'work'
    work.mkdir()
    sources = json.loads((root / 'outputs/preparation/sources.json').read_bytes())
    upstream = Path(sources['sources']['EAGLE']['path']) / 'eagle/traineagle3'
    for name, expected in native.UPSTREAM.items():
        if native.digest(upstream / name) != expected:
            raise ValueError('Pinned native vocabulary/preprocessing source changed')
        shutil.copyfile(upstream / name, work / name)
    assets = json.loads((root / 'outputs/preparation/assets.json').read_bytes())
    target = native.model_record(assets, 'llama31_instruct_8b', verify_content=False)
    for name, item in target['files'].items():
        if 'tokenizer' in name or name in ('config.json', 'special_tokens_map.json'):
            if native.digest(Path(target['path']) / name) != item['sha256']:
                raise ValueError('Native tokenizer identity changed')
    if engineering_fixture is not None:
        paths = {'train': engineering_fixture / 'train.jsonl', 'dev': engineering_fixture / 'dev.jsonl'}
        for split, path in paths.items():
            native.engineering_data(path, split, 8)
        scope, draft_vocabulary = 'engineering_vocabulary_preparation', 16
        corpus_proof = None
    else:
        completion = json.loads((corpus / 'completion.json').read_bytes())
        if completion.get('status') != 'complete' or completion.get('scope') != 'full_training_data_preparation' or completion.get('response_regime') != 'target_generated' or completion['target_identity_sha256'] != target['model_identity_sha256']:
            raise ValueError('A completed full target-generated corpus for this target is required')
        for name, item in completion['files'].items():
            if native.digest(corpus / name) != item['sha256']:
                raise ValueError('Finalized target-generated corpus changed')
        paths = {split: corpus / ('native-' + split + '.jsonl') for split in ('train', 'dev', 'test')}
        scope = 'native_data_preparation'
        draft_vocabulary = json.loads((work / 'config.json').read_bytes())['draft_vocab_size']
        corpus_proof = {'path': str(corpus / 'completion.json'), 'sha256': native.digest(corpus / 'completion.json')}
    inventories = {split: native.dataset_inventory(path) for split, path in paths.items()}
    for a in inventories:
        for b in inventories:
            if a < b and (set(inventories[a]['ids']) & set(inventories[b]['ids']) or set(inventories[a]['prompt_identities']) & set(inventories[b]['prompt_identities'])):
                raise ValueError('Native split IDs or complete user-prompt sequences overlap')
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tree = ast.parse((work / 'main.py').read_bytes())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_dataset_rank']
    if len(functions) != 1:
        raise ValueError('Exact native dataset builder is missing')
    namespace = {'torch': torch, 'load_dataset': load_dataset, 'train_config': {'max_len': 2048}, '__name__': '__native_data_quality__'}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(work / 'main.py'), 'exec'), namespace)
    tokenizer = AutoTokenizer.from_pretrained(target['path'], local_files_only=True)
    summaries = {}
    for split, path in paths.items():
        dataset = namespace['build_dataset_rank'](tokenizer, str(path))
        lengths, supervised = [], 0
        for row in dataset:
            ids, mask = row['input_ids'], row['loss_mask']
            if ids.shape != mask.shape or ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 2048 or not bool(((mask == 0) | (mask == 1)).all()) or int(mask.sum()) < 1:
                raise ValueError('Exact native preprocessing produced an invalid or unsupervised example')
            lengths.append(int(ids.shape[1]))
            supervised += int(mask.sum())
        if len(lengths) < 8:
            raise ValueError('Native filtering leaves fewer than eight examples in a split')
        summaries[split] = {'input_sha256': inventories[split]['sha256'], 'rows_before_filter': inventories[split]['rows_before_official_length_filter'], 'rows_after_filter': len(lengths), 'token_slots': sum(lengths), 'supervised_mask_tokens': supervised, 'minimum_length': min(lengths), 'maximum_length': max(lengths), 'no_truncation': True}
    old_directory, old_path = Path.cwd(), sys.path[:]
    try:
        os.chdir(work)
        sys.path.insert(0, str(work))
        if 'cnets' in sys.modules:
            raise ValueError('Another native model module is already imported')
        module = importlib.import_module('cnets')
        if Path(module.__file__).resolve() != work / 'cnets.py':
            raise ValueError('Native vocabulary module came from another checkout')
        vocabulary = json.loads((work / 'config.json').read_bytes())['vocab_size']
        context = VocabularyContext(vocabulary, draft_vocabulary)
        module.Model.scandata(context, str(paths['train']), target['path'])
        native.validate_vocabulary(torch, context.d2t, context.t2d, vocabulary, draft_vocabulary)
    finally:
        os.chdir(old_directory)
        sys.path[:] = old_path
    report = {'status': 'passed', 'scope': scope, 'source_sha256': native.digest(__file__), 'upstream_files_sha256': native.UPSTREAM, 'corpus': corpus_proof, 'target_identity_sha256': target['model_identity_sha256'], 'draft_vocab_size': draft_vocabulary, 'splits': summaries, 'cache': {'path': str(work / 'cache.pt'), 'sha256': native.digest(work / 'cache.pt')}, 'native_scandata_num_processes': 48, 'vocabulary_uses_train_only': True, 'hidden_feature_cache_generated': False, 'training_started': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    native.atomic_json(output / 'report.json', report)
    return report


def bind_vocabulary(plan_path, report_path, cache_path=None):
    native.require_slurm()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Vocabulary binding requires CPU Slurm before launch')
    plan_path, report_path = Path(plan_path), Path(report_path)
    plan, report = native.load_plan(plan_path), json.loads(report_path.read_bytes())
    directory, work = plan_path.parent, Path(plan['workdir'])
    if (directory / 'launch.json').exists() or (directory / 'launch.lock').exists() or 'work/cache.pt' in plan['frozen_files'] or (work / 'cache.pt').exists():
        raise ValueError('Only a fresh unlaunched native plan may receive its vocabulary cache')
    if report.get('status') != 'passed' or report.get('scope') != 'native_data_preparation' or report.get('source_sha256') != native.digest(__file__) or report.get('upstream_files_sha256') != native.UPSTREAM or report.get('target_identity_sha256') != plan['target']['model_identity_sha256'] or report['splits']['train']['input_sha256'] != plan['train']['sha256'] or report.get('draft_vocab_size') != json.loads((work / 'config.json').read_bytes())['draft_vocab_size']:
        raise ValueError('Vocabulary audit does not cover this exact native training contract')
    cache_path = Path(cache_path or report['cache']['path'])
    if native.digest(cache_path) != report['cache']['sha256']:
        raise ValueError('Audited native vocabulary cache changed')
    shutil.copyfile(plan_path, directory / 'plan.before-vocabulary.json')
    shutil.copyfile(cache_path, work / 'cache.pt')
    shutil.copyfile(report_path, directory / 'inputs/vocabulary-audit.json')
    plan['frozen_files'].update({'work/cache.pt': native.digest(work / 'cache.pt'), 'inputs/vocabulary-audit.json': native.digest(directory / 'inputs/vocabulary-audit.json')})
    plan['native_recipe']['vocabulary_preparation'] = {'path': str(directory / 'inputs/vocabulary-audit.json'), 'sha256': native.digest(directory / 'inputs/vocabulary-audit.json'), 'exact_upstream_scandata': True, 'precomputed_before_gpu_model_load': True}
    plan.pop('plan_identity_sha256')
    plan['plan_identity_sha256'] = hashlib.sha256(native.canonical(plan)).hexdigest()
    native.atomic_json(plan_path, plan)
    native.atomic_json(Path(plan['trained_models']['resolved_path']) / 'native-plan.json', {'plan_identity_sha256': plan['plan_identity_sha256'], 'plan_path': str(plan_path)})
    native.load_plan(plan_path)
    return plan


def bind_profile(root, profile_run_id, report_path):
    directory = Path(root) / 'outputs/training-queue' / profile_run_id
    status = json.loads((directory / 'status.json').read_bytes())
    if status['status'] != 'passed' or native.digest(status['native_plan']) != status['native_plan_sha256']:
        raise ValueError('Profile admission or its prepared plan differs')
    plan = bind_vocabulary(status['native_plan'], report_path)
    native.audit(Path(plan['run_dir']) / 'plan.json')
    status.update(native_plan_sha256=native.digest(status['native_plan']), vocabulary_preparation=plan['native_recipe']['vocabulary_preparation'])
    native.atomic_json(directory / 'status.json', status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('audit', 'bind-profile'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--engineering-fixture', type=Path)
    parser.add_argument('--profile-run-id')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.action == 'audit':
        if args.output is None or (args.corpus is None) == (args.engineering_fixture is None):
            parser.error('Audit requires output and exactly one corpus or engineering fixture')
        audit(args.root, args.output, corpus=args.corpus, engineering_fixture=args.engineering_fixture)
    else:
        if args.profile_run_id is None or args.report is None:
            parser.error('Profile binding requires run ID and audit report')
        bind_profile(args.root, args.profile_run_id, args.report)


if __name__ == '__main__':
    main()
