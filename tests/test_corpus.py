from pathlib import Path
import tempfile
import unittest

from taps_rebuild import corpus, native_training


SOURCE = {'id': 'sharegpt_v3', 'repo_id': 'owner/data', 'revision': 'a' * 40}


class CorpusTests(unittest.TestCase):
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
