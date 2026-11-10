import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from taps_rebuild import corpus_generate as generation


class Tokenizer:
    eos_token_id = 2

    def __init__(self):
        self.histories = []

    def convert_tokens_to_ids(self, token):
        return 3

    def apply_chat_template(self, messages, **kwargs):
        self.histories.append(messages.copy())
        return list(range(len(messages) + 1))


class Model:
    def __init__(self, finish='stop', paired=True):
        self.finish, self.paired = finish, paired
        self.calls = []

    def generate(self, prompts, parameters, **kwargs):
        self.calls.append((prompts, parameters))
        return [SimpleNamespace(prompt_token_ids=prompt['prompt_token_ids'] if self.paired else [999], outputs=[SimpleNamespace(text='new answer', token_ids=[10, 3], finish_reason=self.finish, stop_reason=3)]) for prompt in prompts]


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.settings = {'seed': 4, 'max_context': 100, 'max_new_tokens': 20, 'temperature': 0, 'top_p': 1}
        self.record = {'ordinal': 0, 'record': {'example_id': 'a', 'split': 'train', 'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'old answer'}, {'role': 'user', 'content': 'q2'}]}}

    def generate(self, model=None, tokenizer=None, settings=None):
        return generation.generate_batch(model or Model(), tokenizer or Tokenizer(), [self.record], settings or self.settings, 'system', SimpleNamespace)

    def test_every_assistant_turn_is_regenerated_with_the_new_history(self):
        tokenizer, model = Tokenizer(), Model()
        row = self.generate(model, tokenizer)['outcomes'][0]
        self.assertEqual(row['status'], 'complete')
        self.assertEqual([m['content'] for m in row['record']['messages']], ['q', 'new answer', 'q2', 'new answer'])
        self.assertEqual(tokenizer.histories[1][2]['content'], 'new answer')
        self.assertNotIn('old answer', str(tokenizer.histories))
        self.assertEqual(len(row['generation_turns']), 2)

    def test_prompt_only_rows_receive_real_completions(self):
        self.record['record']['messages'] = [{'role': 'user', 'content': 'q'}]
        result = self.generate()['outcomes'][0]
        self.assertTrue(result['record']['original_response_complete'])
        self.assertEqual(len(result['record']['messages']), 2)

    def test_censored_or_overlong_examples_are_excluded(self):
        self.assertEqual(self.generate(Model('length'))['outcomes'][0]['reason'], 'response_censored_at_generation_limit')
        model = Model()
        value = self.generate(model, settings={**self.settings, 'max_context': 1})
        self.assertEqual(value['outcomes'][0]['reason'], 'prompt_exceeds_generation_context')
        self.assertFalse(model.calls)

    def test_outputs_cannot_be_paired_with_another_prompt(self):
        with self.assertRaisesRegex(generation.GenerationError, 'another prompt'):
            self.generate(Model(paired=False))

    def test_resume_recovers_only_the_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'worker.jsonl'
            row = {'plan_identity_sha256': 'proof', 'outcomes': [{'ordinal': 0, 'example_id': 'a'}]}
            valid = (json.dumps(row) + '\n').encode()
            path.write_bytes(valid + b'{"incomplete":')
            self.assertEqual(generation.committed(path, 'proof'), {0: 'a'})
            self.assertEqual(path.read_bytes(), valid)
            self.assertEqual(len(list(path.parent.glob('*.uncommitted-*'))), 1)
            with self.assertRaisesRegex(generation.GenerationError, 'another plan'):
                generation.committed(path, 'wrong')
            path.write_bytes(valid * 2)
            with self.assertRaisesRegex(generation.GenerationError, 'Duplicate'):
                generation.committed(path, 'proof')


if __name__ == '__main__':
    unittest.main()
