import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from taps_rebuild import admission, admission_v2, native_training


class PipelineAdmissionTests(unittest.TestCase):
    def test_claimed_metadata_and_embedded_review_cannot_admit(self):
        with self.assertRaisesRegex(admission.AdmissionError, 'source-bound v2 CPU replay'):
            admission_v2.verify_certificate('.', {'status': 'passed', 'checks': {'passed': True}, 'review': {'decision': 'approved'}}, {})

    def test_retrospective_policy_is_explicit_in_native_config(self):
        source = json.loads((Path(__file__).resolve().parents[1] / 'configs/native-eagle3-training.json').read_text())
        source.update(scope='native_training', initialize_from_released_draft=False, research_gate={'policy': 'pipeline-v2', 'pipeline_certificate': {'path': 'report', 'sha256': 'a' * 64}, 'independent_review': {'path': 'review', 'sha256': 'b' * 64}})
        self.assertEqual(native_training.validate_config(source)['research_gate']['policy'], 'pipeline-v2')
        source['research_gate']['policy'] = 'ignore-correctness'
        with self.assertRaisesRegex(native_training.NativeTrainingError, 'source-bound validation'):
            native_training.validate_config(source)

    def test_sampling_rejects_short_unterminated_and_substituted_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'sampling-eagle3-0.jsonl'
            row = {'method': 'eagle3', 'prompt_index': 0, 'input_token_ids': [1, 2], 'output_token_ids': [], 'comparison_token_ids': []}
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(admission.AdmissionError, 'Short unterminated'):
                admission_v2.sampling_inputs(root, {('eagle3', 0): [1, 2]}, {'eagle3': 10}, {'eagle3': {9}}, lambda *args: None)
            row.update(input_token_ids=[1, 3], output_token_ids=[9], comparison_token_ids=[9])
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(admission.AdmissionError, 'independently rendered'):
                admission_v2.sampling_inputs(root, {('eagle3', 0): [1, 2]}, {'eagle3': 10}, {'eagle3': {9}}, lambda *args: None)

    def test_pending_review_never_reaches_native_execution(self):
        config = {'scope': 'native_training', 'research_gate': {'policy': 'pipeline-v2', 'pipeline_certificate': {}, 'independent_review': {}}}
        with patch.object(native_training, 'gate_document', return_value={'native_training_sha256': 'different'}):
            with self.assertRaisesRegex(native_training.NativeTrainingError, 'another native training'):
                native_training.research_gate('.', config)

    def test_pipeline_contract_cannot_admit_another_target_or_skip_profile(self):
        report = {'training_contract': admission_v2.TRAINING_CONTRACT}
        config = {**admission_v2.TRAINING_CONTRACT, 'budget': {'optimizer_steps': 200}, 'pipeline_stage': 'real-data-profile'}
        admission_v2.verify_training_scope('.', report, config)
        with self.assertRaisesRegex(admission.AdmissionError, 'target/runtime/recipe'):
            admission_v2.verify_training_scope('.', report, {**config, 'target_id': 'another-target'})
        with self.assertRaisesRegex(admission.AdmissionError, '200-update'):
            admission_v2.verify_training_scope('.', report, {**config, 'budget': {'optimizer_steps': None}})
        with self.assertRaisesRegex(admission.AdmissionError, 'qualification inventory'):
            admission_v2.verify_training_scope('.', report, {**config, 'pipeline_stage': 'full-generalist', 'budget': {'optimizer_steps': None}, 'recovery': {'checkpoint_interval': 1000, 'stop_after_optimizer_step': None}})


if __name__ == '__main__':
    unittest.main()
