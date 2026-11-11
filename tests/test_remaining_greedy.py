import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[1] / 'experiments/remaining-greedy-20261111/driver.py'
SPEC = importlib.util.spec_from_file_location('remaining_greedy', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row(benchmark, question, turn=0, failure=False, length_only=False):
    ids = [question + 1000, turn + 1000]
    ar = [1] * 220
    spec = ar.copy()
    if failure:
        if length_only:
            spec.append(2)
        else:
            spec[200] = 2
    case = {'benchmark': benchmark, 'question_id': question, 'turn_id': turn, 'input_token_ids': ids, 'input_sha256': audit.identity(ids), 'seed': 42}
    outputs = {mode: {'output_token_ids': tokens, 'comparison_token_ids': tokens} for mode, tokens in [('target_only', ar), ('speculative', spec)]}
    return {'method': 'eagle3', 'precision': 'float32', 'case': case, 'outputs': outputs, 'equal': ar == spec}


def panel():
    rows = [row('mt_bench', 146, 1, failure=True), row('cnn_dm', 295, failure=True, length_only=True), row('cnn_dm', 288, failure=True, length_only=True)]
    rows.extend(row(audit.BENCHMARKS[index % 5], index + 10000) for index in range(477))
    return rows


class RemainingGreedyTests(unittest.TestCase):
    def test_selection_is_order_invariant_and_covers_all_control_domains(self):
        rows = panel()
        cases, tasks = audit.select(rows)
        reverse_cases, reverse_tasks = audit.select(list(reversed(rows)))
        self.assertEqual((cases, tasks), (reverse_cases, reverse_tasks))
        self.assertEqual(len(cases), 8)
        self.assertEqual(tasks[0]['role'], 'remaining_token_failure')
        self.assertEqual(cases[0]['divergence'], 200)
        self.assertEqual({task['case']['benchmark'] for task in tasks[1:]}, set(audit.BENCHMARKS))
        self.assertEqual(len({audit.identity(task['case']) for task in tasks}), 8)
        self.assertTrue(all(task['probe_offset'] == 32 for task in tasks[1:]))

    def test_duplicate_or_missing_original_case_is_rejected(self):
        rows = panel()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete or duplicate'):
            audit.select(rows[:-1])
        rows[-1] = copy.deepcopy(rows[-2])
        with self.assertRaisesRegex(RuntimeError, 'Incomplete or duplicate'):
            audit.select(rows)

    def test_changed_token_input_or_comparison_is_rejected(self):
        rows = panel()
        rows[0]['case']['input_token_ids'].append(99)
        with self.assertRaisesRegex(RuntimeError, 'Input token identity'):
            audit.select(rows)
        rows = panel()
        rows[0]['equal'] = True
        with self.assertRaisesRegex(RuntimeError, 'Stored comparison'):
            audit.select(rows)

    def test_shifted_or_additional_failure_is_rejected(self):
        rows = panel()
        rows[0]['outputs']['speculative']['comparison_token_ids'][199] = 2
        with self.assertRaisesRegex(RuntimeError, 'first token disagreement'):
            audit.select(rows)
        rows = panel()
        rows[-1]['equal'] = False
        rows[-1]['outputs']['speculative']['comparison_token_ids'][5] = 2
        with self.assertRaisesRegex(RuntimeError, 'failure population'):
            audit.select(rows)

    def test_missing_mode_cannot_be_summarized_as_valid(self):
        cases, tasks = audit.select(panel())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = {'tasks': tasks, 'cases_sha256': 'cases', 'audit_worker_sha256': 'worker'}
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            receipt = {'status': 'complete', 'worker': 0, 'precision': 'float32', 'method': 'eagle3', 'cases_sha256': 'cases', 'script_sha256': 'worker', 'benchmark': 'mt_bench', 'question_id': 146, 'turn_id': 1, 'divergence': 200, 'records': {}}
            (directory / 'eagle3-float32-0.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RuntimeError, 'Missing paired mode'):
                audit.summarize(directory)
            self.assertFalse((directory / 'report.json').exists())

    def test_scheduler_cleanup_can_finish_without_signalling_existing_processes(self):
        now = [0]
        answers = iter(['123\n456', ''])
        records = []
        def sleep(seconds):
            now[0] += seconds
        observations = audit.wait_for_idle(lambda: next(answers), lambda values: records.append(copy.deepcopy(values)), clock=lambda: now[0], sleep=sleep)
        self.assertEqual(now[0], 5)
        self.assertEqual(observations[-1]['reported_processes'], '')
        self.assertEqual(records[0][0]['reported_processes'], '123\n456')

    def test_busy_gpus_never_become_an_implicit_permission_to_load_models(self):
        now = [0]
        records = []
        def sleep(seconds):
            now[0] += seconds
        with self.assertRaisesRegex(RuntimeError, 'no models were loaded'):
            audit.wait_for_idle(lambda: '123', lambda values: records.append(copy.deepcopy(values)), clock=lambda: now[0], sleep=sleep, timeout=10)
        self.assertEqual(now[0], 10)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(record[-1]['reported_processes'] == '123' for record in records))

    def test_only_the_registered_absent_process_context_can_be_excluded(self):
        app = {'gpu_uuid': 'gpu', 'pid': 12, 'process_name': '[Not Found]', 'used_memory_mib': 10000}
        policy = {'contexts': [{'gpu_uuid': 'gpu', 'pid': 12, 'process_name': '[Not Found]', 'max_used_memory_mib': 11000}], 'min_free_gpu_mib': 51200}
        self.assertEqual(audit.blocking_contexts([app], {'gpu': 70000}, policy, lambda _: False, True), [])
        self.assertEqual(audit.blocking_contexts([app], {'gpu': 70000}, None, lambda _: False, True), [app])
        self.assertEqual(audit.blocking_contexts([app], {'gpu': 70000}, policy, lambda _: True, True), [app])
        self.assertEqual(audit.blocking_contexts([app], {'gpu': 70000}, policy, lambda _: False, False), [app])

    def test_unregistered_or_larger_context_and_insufficient_memory_are_blocked(self):
        app = {'gpu_uuid': 'gpu', 'pid': 12, 'process_name': '[Not Found]', 'used_memory_mib': 10000}
        policy = {'contexts': [{'gpu_uuid': 'gpu', 'pid': 12, 'process_name': '[Not Found]', 'max_used_memory_mib': 11000}], 'min_free_gpu_mib': 51200}
        changed = {**app, 'pid': 99}
        self.assertEqual(audit.blocking_contexts([changed], {'gpu': 70000}, policy, lambda _: False, True), [changed])
        changed = {**app, 'used_memory_mib': 12000}
        self.assertEqual(audit.blocking_contexts([changed], {'gpu': 70000}, policy, lambda _: False, True), [changed])
        with self.assertRaisesRegex(RuntimeError, 'Insufficient free GPU memory'):
            audit.blocking_contexts([app], {'gpu': 50000}, policy, lambda _: False, True)


if __name__ == '__main__':
    unittest.main()
