import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from taps_rebuild import sources


class SourcePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / 'project'
        self.registry_path = self.base / 'registry.json'
        self.prompt = '  Keep this prompt\nexactly.  '
        self.eval_path = 'eagle/evaluation/evaluator.py'
        self.data_path = 'eagle/data/mt_bench/question.jsonl'
        self.rows = [{'question_id': i, 'turns': ['first ' + str(i), 'second ' + str(i)]} for i in range(80)]
        self.contents = self.jsonl(self.rows)
        self.source = {'name': 'EAGLE', 'url': 'https://github.com/SafeAILab/EAGLE.git', 'revision': 'a' * 40, 'path': 'repos/EAGLE', 'system_prompt_file': self.eval_path}
        self.benchmark = {'name': 'mt_bench', 'source': 'EAGLE', 'path': self.data_path, 'sha256': hashlib.sha256(self.contents).hexdigest(), 'rows': 80, 'turns_per_row': 2}
        self.registry = {'schema_version': 1, 'sources': [self.source], 'benchmarks': [self.benchmark]}
        self.files = {self.eval_path: ('raise RuntimeError("must not execute")\nmessages = [{"role": "system", "content": ' + repr(self.prompt) + '}]\n').encode(), self.data_path: self.contents, 'requirements.txt': b'torch==2.6.0\n'}
        self.head = self.source['revision']
        self.origin = self.source['url']
        self.status = b''
        environment = patch.dict(os.environ, {'SLURM_JOB_ID': 'test-allocation', 'CUDA_VISIBLE_DEVICES': ''})
        environment.start()
        self.addCleanup(environment.stop)

    @staticmethod
    def jsonl(rows):
        return ''.join(json.dumps(row) + '\n' for row in rows).encode()

    def git(self, directory, *arguments):
        if arguments[0] in ('init', 'fetch') or arguments[:2] == ('remote', 'add'):
            return b''
        if arguments[0] == 'checkout':
            for name, value in self.files.items():
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            return b''
        if arguments == ('rev-parse', '--show-toplevel'):
            return (str(directory.resolve()) + '\n').encode()
        if arguments == ('rev-parse', 'HEAD'):
            return (self.head + '\n').encode()
        if arguments == ('remote', 'get-url', 'origin'):
            return (self.origin + '\n').encode()
        if arguments[0] == 'status':
            return self.status
        if arguments == ('ls-files', '-z'):
            return b'\0'.join(name.encode() for name in self.files) + b'\0'
        raise AssertionError(arguments)

    def prepare(self):
        self.registry_path.write_text(json.dumps(self.registry))
        return sources.prepare_sources(self.root, self.registry_path)

    def test_pinned_fetch_extract_copy_and_resume_are_immutable(self):
        with patch.object(sources, 'run_git', side_effect=self.git) as git:
            first = self.prepare()
            fetches = [call for call in git.call_args_list if call.args[1] == 'fetch']
            self.assertEqual(len(fetches), 1)
            self.assertEqual(fetches[0].args[-1], 'a' * 40)
            self.assertIn('--depth=1', fetches[0].args)
            source = first['sources']['EAGLE']
            self.assertEqual(source['system_prompt']['text'], self.prompt)
            self.assertEqual(source['system_prompt']['sha256'], hashlib.sha256(self.prompt.encode()).hexdigest())
            self.assertEqual(set(source['tracked_files']), set(self.files))
            target = Path(first['benchmarks']['mt_bench']['path'])
            self.assertEqual(target.read_bytes(), self.contents)
            stamp = target.stat().st_mtime_ns
            git.reset_mock()
            second = self.prepare()
            self.assertFalse(any(call.args[1] in ('init', 'fetch', 'checkout') for call in git.call_args_list))
            self.assertEqual(target.stat().st_mtime_ns, stamp)
            self.assertEqual(source['source_identity_sha256'], second['sources']['EAGLE']['source_identity_sha256'])
            receipt = self.root / 'outputs/preparation/sources.json'
            self.assertEqual(json.loads(receipt.read_text()), second)
            self.assertEqual(second['status'], 'complete')
            self.assertFalse(second['upstream_modified'])
            self.assertFalse(second['gpu_execution_performed'])
            self.assertEqual(list(receipt.parent.glob('sources.json.*.tmp')), [])

    def test_existing_wrong_head_origin_or_dirty_checkout_is_preserved(self):
        with patch.object(sources, 'run_git', side_effect=self.git):
            self.prepare()
            directory = self.root / self.source['path']
            before = {name: (directory / name).read_bytes() for name in self.files}
            for field, bad in [('head', 'b' * 40), ('origin', 'https://github.com/other/repo.git'), ('status', b' M requirements.txt\n')]:
                with self.subTest(field=field):
                    old = getattr(self, field)
                    setattr(self, field, bad)
                    with self.assertRaises(sources.SourcePreparationError):
                        self.prepare()
                    setattr(self, field, old)
                    self.assertEqual(before, {name: (directory / name).read_bytes() for name in self.files})

    def test_unexpected_reference_is_not_overwritten(self):
        with patch.object(sources, 'run_git', side_effect=self.git):
            self.prepare()
            target = self.root / 'data/reference/mt_bench.jsonl'
            target.write_bytes(b'existing-unexpected-data')
            with self.assertRaisesRegex(sources.SourcePreparationError, 'refusing overwrite'):
                self.prepare()
            self.assertEqual(target.read_bytes(), b'existing-unexpected-data')
            receipt = json.loads((self.root / 'outputs/preparation/sources.json').read_text())
            self.assertEqual(receipt['status'], 'failed')

    def test_literal_prompt_preserves_content_and_rejects_ambiguity(self):
        literal = b'messages=[{"role":"system","content":"literal\\n  text"}]\n'
        self.assertEqual(sources.extract_system_prompt(literal), 'literal\n  text')
        self.assertEqual(sources.extract_system_prompt(literal + literal), 'literal\n  text')
        conflicting = literal + b'messages=[{"role":"system","content":"different"}]\n'
        for source in [b'messages=[]', conflicting, b'messages=[{"role":"system","content":prompt}]', b'messages=[{"role":"system","content":""}]']:
            with self.subTest(source=source), self.assertRaises(sources.SourcePreparationError):
                sources.extract_system_prompt(source)

    def test_reference_hash_rows_ids_and_turns_are_checked(self):
        self.assertEqual(sources.validate_benchmark(self.contents, self.benchmark), list(range(80)))
        with self.assertRaisesRegex(sources.SourcePreparationError, 'SHA256'):
            sources.validate_benchmark(self.contents + b' ', self.benchmark)
        malformed = []
        duplicate = copy.deepcopy(self.rows)
        duplicate[1]['question_id'] = 0
        malformed.append(duplicate)
        empty = copy.deepcopy(self.rows)
        empty[0]['turns'][0] = '  '
        malformed.append(empty)
        turns = copy.deepcopy(self.rows)
        turns[0]['turns'] = ['one']
        malformed.append(turns)
        malformed.append(self.rows[:-1])
        missing = copy.deepcopy(self.rows)
        del missing[0]['question_id']
        malformed.append(missing)
        for rows in malformed:
            with self.subTest(rows=rows[0]):
                contents = self.jsonl(rows)
                expected = dict(self.benchmark, sha256=hashlib.sha256(contents).hexdigest())
                with self.assertRaises(sources.SourcePreparationError):
                    sources.validate_benchmark(contents, expected)

    def test_source_symlink_is_rejected_without_git_or_changes(self):
        target = self.base / 'other'
        target.mkdir()
        directory = self.root / self.source['path']
        directory.parent.mkdir(parents=True)
        directory.symlink_to(target, target_is_directory=True)
        with patch.object(sources, 'run_git') as git:
            with self.assertRaisesRegex(sources.SourcePreparationError, 'existing directory'):
                sources.ensure_checkout(self.root, self.source)
            git.assert_not_called()
        self.assertTrue(directory.is_symlink())

    def test_allocation_and_cuda_guard_precedes_file_access(self):
        for environment in [{}, {'SLURM_JOB_ID': 'job'}, {'SLURM_JOB_ID': 'job', 'CUDA_VISIBLE_DEVICES': '0'}]:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(sources.SourcePreparationError):
                    sources.prepare_sources(self.root, self.registry_path)
        self.assertFalse(self.root.exists())

    def test_git_disables_home_config_credentials_and_alternate_worktrees(self):
        with patch.dict(os.environ, {'GIT_DIR': '/unexpected', 'GIT_WORK_TREE': '/unexpected', 'GIT_CONFIG_PARAMETERS': 'unexpected'}), patch.object(sources.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=b'fine', stderr=b'')) as run:
            self.assertEqual(sources.run_git(self.base, 'status'), b'fine')
            environment = run.call_args.kwargs['env']
            self.assertEqual(environment['GIT_CONFIG_GLOBAL'], '/dev/null')
            self.assertEqual(environment['GIT_CONFIG_NOSYSTEM'], '1')
            self.assertEqual(environment['GIT_CONFIG_COUNT'], '2')
            self.assertEqual(environment['GIT_CONFIG_VALUE_0'], '/dev/null')
            self.assertEqual(environment['GIT_CONFIG_VALUE_1'], '/dev/null')
            self.assertEqual(environment['GIT_TERMINAL_PROMPT'], '0')
            self.assertEqual(environment['GIT_ALLOW_PROTOCOL'], 'https')
            self.assertNotIn('GIT_DIR', environment)
            self.assertNotIn('GIT_WORK_TREE', environment)
            self.assertNotIn('GIT_CONFIG_PARAMETERS', environment)

    def test_registry_rejects_unpinned_sources_and_path_escapes(self):
        for key, value in [('revision', 'main'), ('path', '../escape'), ('url', 'git@github.com:owner/repo.git'), ('system_prompt_file', '/outside')]:
            with self.subTest(key=key):
                registry = copy.deepcopy(self.registry)
                registry['sources'][0][key] = value
                with self.assertRaises(sources.SourcePreparationError):
                    sources.validate_registry(registry)

    def test_real_registry_has_official_commits_and_five_pinned_80_row_sets(self):
        registry = json.loads((PROJECT / 'configs/sources.json').read_text())
        sources.validate_registry(registry)
        self.assertEqual({source['name']: source['revision'] for source in registry['sources']}, {'EAGLE': 'cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b', 'HASS': '051c4020141f5d2a365994ca9a8148700044c6d7'})
        self.assertEqual({item['name'] for item in registry['benchmarks']}, {'mt_bench', 'humaneval', 'gsm8k', 'alpaca', 'cnn_dm'})
        self.assertTrue(all(item['rows'] == 80 for item in registry['benchmarks']))
        self.assertEqual(next(item['turns_per_row'] for item in registry['benchmarks'] if item['name'] == 'mt_bench'), 2)


if __name__ == '__main__':
    unittest.main()
