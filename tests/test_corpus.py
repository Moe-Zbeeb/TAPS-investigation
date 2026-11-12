from pathlib import Path
from contextlib import closing
from contextlib import redirect_stdout
from io import StringIO
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from taps_rebuild import corpus, native_training, pilot_data, pool


SOURCE = {'id': 'sharegpt_v3', 'repo_id': 'owner/data', 'revision': 'a' * 40}


class CorpusTests(unittest.TestCase):
    def test_bounded_preparation_excludes_prior_chat_and_reference_prompts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = {**SOURCE, 'id': 'pilot_math', 'format': 'alpaca', 'domain': 'math'}
            raw = root / 'data/raw/pilot_math' / source['revision'] / 'rows.json'
            raw.parent.mkdir(parents=True)
            raw.write_text(json.dumps([{'instruction': 'problem ' + str(i), 'output': 'solution ' + str(i)} for i in range(200)]))
            source['files'] = [{'path': 'rows.json', 'size_bytes': raw.stat().st_size, 'sha256': pool.digest(raw)}]
            anchor = root / 'data/anchor'
            anchor.mkdir()
            for split in ('train', 'dev', 'test'):
                (anchor / (split + '.jsonl')).write_text(json.dumps({'messages': [{'role': 'user', 'content': 'problem 0'}]}) + '\n')
            receipt = {'status': 'complete', 'scope': 'real_training_corpus_preparation', 'files': {p.name: {'sha256': pool.digest(p), 'size_bytes': p.stat().st_size} for p in anchor.glob('*.jsonl')}}
            (anchor / 'completion.json').write_text(json.dumps(receipt))
            config = {'sources': [source], 'split': {'seed': 42, 'train_fraction': 0.8, 'dev_fraction': 0.1, 'test_fraction': 0.1}, 'selection': {'seed': 42, 'max_groups_by_source': {'pilot_math': 70}}, 'exclusion_corpora': [{'path': 'data/anchor', 'completion_sha256': pool.digest(anchor / 'completion.json')}]}
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            with patch.dict('os.environ', {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}), patch.object(corpus, 'benchmark_exclusions', return_value=({corpus.text_id('problem 1')}, [])), redirect_stdout(StringIO()):
                result = corpus.prepare(root, path, root / 'data/prepared')
            rows = [json.loads(line) for split in ('train', 'dev', 'test') for line in (root / 'data/prepared' / (split + '.jsonl')).read_text().splitlines()]
            self.assertEqual(len(rows), 70)
            self.assertTrue(all(row['domain'] == 'math' and row['original_response_complete'] for row in rows))
            self.assertFalse({'problem 0', 'problem 1'} & {row['messages'][0]['content'] for row in rows})
            self.assertEqual(result['counts']['excluded:prior_corpus_prompt_overlap'], 1)
            self.assertEqual(result['counts']['excluded:exact_reference_benchmark_overlap'], 1)
            self.assertEqual(result['counts']['selection_excluded:pilot_math'], 128)

    def test_pilot_git_blob_identity_is_verified_before_config_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'data/raw/code' / ('a' * 40) / 'data.json'
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b'hello\n')
            self.assertEqual(pilot_data.git_blob(raw), 'ce013625030ba8dba906f756967f9e9ca394464a')
            config = {'sources': [{'id': 'code', 'repo_id': 'owner/data', 'revision': 'a' * 40, 'files': [{'path': 'data.json', 'size_bytes': 6, 'git_blob_sha1': pilot_data.git_blob(raw)}]}]}
            raw.write_bytes(b'wrong!')
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            derived = root / 'outputs/preparation/derived.json'
            with patch.dict('os.environ', {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}), self.assertRaisesRegex(ValueError, 'immutable Git blob'):
                pilot_data.prepare(root, path, derived, root / 'data/prepared')
            self.assertFalse(derived.exists())
            self.assertEqual(raw.read_bytes(), b'wrong!')

    def test_alpaca_pairs_preserve_text_domain_and_upstream_source(self):
        source = {**SOURCE, 'id': 'pilot_code', 'domain': 'code', 'format': 'alpaca'}
        value = corpus.canonical_row({'instruction': ' Write code ', 'input': ' x = 1 ', 'output': ' print(x) ', 'source': 'original-task'}, source, 7, 'original.json')
        self.assertEqual(value['domain'], 'code')
        self.assertEqual(value['messages'], [{'role': 'user', 'content': ' Write code \n\n x = 1 '}, {'role': 'assistant', 'content': ' print(x) '}])
        self.assertEqual(value['source']['row'], 7)
        self.assertEqual(value['source']['upstream_source'], 'original-task')
        with self.assertRaises(corpus.CorpusError):
            corpus.canonical_row({'instruction': 'Q', 'output': None}, source, 7, 'original.json')

    def test_pilot_group_selection_is_bounded_stable_and_keeps_families(self):
        with closing(sqlite3.connect(':memory:')) as connection:
            connection.execute('CREATE TABLE records (ordinal INTEGER PRIMARY KEY, group_key TEXT, value TEXT)')
            groups = corpus.Groups()
            for index in range(12):
                source = 'math' if index < 8 else 'code'
                group = 'family:' + str(index // 2)
                row = {'source': {'id': source}, 'messages': [{'role': 'user'}, {'role': 'assistant'}]}
                connection.execute('INSERT INTO records VALUES (?,?,?)', (index, group, json.dumps(row)))
                groups.find(group)
            config = {'seed': 42, 'max_groups_by_source': {'math': 2, 'code': 1}}
            chosen = corpus.selected_groups(connection, groups, config)
            self.assertEqual(len(chosen), 3)
            self.assertEqual(sum(group in chosen for group, in connection.execute('SELECT group_key FROM records')), 6)
            self.assertEqual(corpus.selected_groups(connection, groups, config), chosen)
            self.assertIsNone(corpus.selected_groups(connection, groups, None))
            connection.execute('UPDATE records SET value=? WHERE ordinal=0', (json.dumps({'source': {'id': 'other'}, 'messages': [{}, {}]}),))
            with self.assertRaisesRegex(corpus.CorpusError, 'cross-source'):
                corpus.selected_groups(connection, groups, config)

    def test_sharegpt_segments_retain_original_group_and_text(self):
        row = {'id': 'conversation_12', 'conversations': [{'from': 'human', 'value': '  Question?  '}, {'from': 'gpt', 'value': 'Answer.'}]}
        value = corpus.canonical_row(row, SOURCE, 0, 'data.json')
        self.assertEqual(value['source_group_id'], 'sharegpt_v3:conversation')
        self.assertEqual(value['messages'][0]['content'], '  Question?  ')
        self.assertEqual(native_training.native_row({**value, 'split': 'train'}, canonical_messages=True, split='train')['conversations'][0]['value'], '  Question?  ')

    def test_incomplete_conversation_cannot_become_training_data(self):
        for messages in [[{'role': 'user', 'content': 'Question'}], [{'role': 'user', 'content': 'Q'}, {'role': 'user', 'content': 'Q2'}]]:
            with self.assertRaises(corpus.CorpusError):
                corpus.canonical_row({'messages': messages}, SOURCE, 0, 'data.json')

    def test_cross_source_first_prompt_merges_whole_conversation_groups(self):
        groups = corpus.Groups()
        groups.join('source:sharegpt:root1', 'prompt:identical')
        groups.join('source:ultrachat:root2', 'prompt:identical')
        groups.join('source:sharegpt:root1', 'prompt:segment2')
        roots = {groups.find(value) for value in ['source:sharegpt:root1', 'source:ultrachat:root2', 'prompt:segment2']}
        self.assertEqual(len(roots), 1)
        config = {'seed': 2, 'train_fraction': 0.98, 'dev_fraction': 0.01}
        self.assertEqual(len({corpus.split_for(root, config) for root in roots}), 1)

    def test_template_controls_are_excluded_without_rewriting(self):
        row = {'messages': [{'role': 'user', 'content': 'Q<|eot_id|>'}, {'role': 'assistant', 'content': 'A'}]}
        with self.assertRaisesRegex(corpus.CorpusError, 'template'):
            corpus.canonical_row(row, SOURCE, 0, 'data.json')

    def test_existing_download_must_match_pinned_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'data.json'
            path.write_bytes(b'wrong')
            with self.assertRaisesRegex(corpus.CorpusError, 'pinned'):
                corpus.download(SOURCE, {'size_bytes': 5, 'sha256': '0' * 64}, path)

    def test_assistant_prefix_removal_is_explicit(self):
        value = corpus.canonical_row({'messages': [{'role': 'assistant', 'content': 'Orphan'}, {'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': 'A'}]}, SOURCE, 0, 'data.json')
        self.assertTrue(value['initial_assistant_removed'])
        self.assertEqual(len(value['messages']), 2)

    def test_generation_split_prompts_are_retained_without_invented_answers(self):
        value = corpus.canonical_row({'prompt_id': 'x', 'messages': [{'role': 'user', 'content': 'Complete this task'}]}, {**SOURCE, 'id': 'ultrachat_200k'}, 0, 'train_gen.parquet', allow_prompt_only=True)
        self.assertFalse(value['original_response_complete'])
        self.assertEqual(value['response_regime'], 'original_prompt_only')
        self.assertEqual(value['messages'], [{'role': 'user', 'content': 'Complete this task'}])


if __name__ == '__main__':
    unittest.main()
