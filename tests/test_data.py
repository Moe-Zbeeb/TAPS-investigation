import unittest
from unittest.mock import patch

from taps_rebuild import data


class Template:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        value = [1]
        for message in messages:
            value += [2 if message['role'] == 'user' else 3]
            value += [ord(char) + 10 for char in message['content']]
            value += [4]
        if add_generation_prompt:
            value += [3]
        return value


class DataTests(unittest.TestCase):
    def test_sharegpt_roles_and_literal_content(self):
        text = 'def f():\n    return 1\n'
        messages = data.messages_from({'conversations': [{'from': 'human', 'value': text}, {'from': 'gpt', 'value': 'ok'}]})
        self.assertEqual(messages[0]['content'], text)
        self.assertEqual([row['role'] for row in messages], ['user', 'assistant'])

    def test_original_response_does_not_change_split(self):
        rows = [{'prompt': 'question', 'response': 'answer A'}, {'prompt': 'question', 'response': 'answer B'}]
        values, excluded = data.normalize_rows(rows, domain='chat', source_id='source', source_revision='fixed', seed=1, train_fraction=.8, dev_fraction=.1)
        self.assertEqual(len(values), 1)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(values[0]['prompt_id'], excluded[0]['prompt_id'])

    def test_holdout_exclusion_is_explicit(self):
        row = {'prompt': 'evaluation', 'response': 'answer'}
        key = data.prompt_identity(data.messages_from(row))
        values, excluded = data.normalize_rows([row], domain='math', source_id='source', source_revision='fixed', seed=1, train_fraction=.8, dev_fraction=.1, excluded_prompt_ids=[key])
        self.assertEqual(values, [])
        self.assertEqual(len(excluded), 1)

    def test_response_variants_share_prompt_identity(self):
        first = data.messages_from({'prompt': 'why?', 'response': 'old'})
        second = data.messages_from({'prompt': 'why?', 'response': 'new'})
        self.assertEqual(data.prompt_identity(first), data.prompt_identity(second))

    def test_multiple_assistant_spans_and_termination(self):
        messages = [{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': 'a'}, {'role': 'user', 'content': 'y'}, {'role': 'assistant', 'content': 'b'}]
        tokens, mask = data.assistant_token_mask(Template(), messages)
        self.assertEqual(len(tokens), len(mask))
        self.assertEqual(sum(mask), 4)
        self.assertEqual(mask[-1], 1)
        self.assertEqual(mask[0], 0)

    def test_invalid_roles_are_rejected(self):
        with self.assertRaises(ValueError):
            data.messages_from({'messages': [{'role': 'tool', 'content': 'x'}]})

    def test_split_preserves_test_fraction(self):
        with self.assertRaises(ValueError):
            data.split_for('key', 1, .9, .2)
        self.assertEqual(data.split_for('key', 1, .8, .1), data.split_for('key', 1, .8, .1))

    def test_cpu_guard_does_not_allow_gpu_processing(self):
        with patch.dict(data.os.environ, {'SLURM_JOB_ID': '1', 'CUDA_VISIBLE_DEVICES': '0'}, clear=True):
            with self.assertRaises(ValueError):
                data.require_slurm(gpu=False)

    def test_outputs_cannot_escape_data(self):
        with self.assertRaises(ValueError):
            data.output_path('/project', '../outside')


if __name__ == '__main__':
    unittest.main()
