import copy
import fcntl
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from taps_rebuild import corpus_parallel as parallel


def drain(path, plan):
    while True:
        selected = parallel.claim(path, plan['tasks'])
        if selected is None:
            return
        index, lock = selected
        if index < 0:
            time.sleep(0.002)
            continue
        try:
            time.sleep(0.002)
            parallel.commit_chunk(path, plan, index, {'outcomes': [index]}, {'worker': multiprocessing.current_process().pid})
        finally:
            lock.close()


class ParallelTeacherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        for name in ('results', 'locks', 'attempts'):
            (self.path / name).mkdir()
        self.plan = {'identity_sha256': 'plan', 'tasks': [{'sha256': str(i)} for i in range(24)]}
        self.source = {'ordinal': 2, 'record': {'example_id': 'a', 'split': 'train', 'source': 'ultrachat', 'messages': [{'role': 'user', 'content': 'question'}]}}
        self.outcome = {'ordinal': 2, 'example_id': 'a', 'status': 'complete', 'reason': None, 'record': {**self.source['record'], 'messages': [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}], 'response_regime': 'target_generated', 'original_response_complete': True}}
        self.generator = SimpleNamespace(native_training=SimpleNamespace(native_row=lambda row, **kwargs: {'id': row['example_id']}))

    def test_competing_processes_commit_each_task_once(self):
        context = multiprocessing.get_context('fork')
        children = [context.Process(target=drain, args=(self.path, self.plan)) for _ in range(4)]
        for child in children:
            child.start()
        for child in children:
            child.join(10)
            self.assertEqual(child.exitcode, 0)
        self.assertTrue(parallel.queue_complete(self.path, self.plan))
        self.assertEqual(sorted(parallel.read(p)['task_index'] for p in (self.path / 'results').glob('*.json')), list(range(24)))
        self.assertIsNone(parallel.claim(self.path, self.plan['tasks']))

    def test_busy_claim_is_not_completion_and_closed_claim_can_be_recovered(self):
        task = self.plan['tasks'][:1]
        index, lock = parallel.claim(self.path, task)
        self.assertEqual(parallel.claim(self.path, task), (-1, None))
        lock.close()
        reclaimed, lock = parallel.claim(self.path, task)
        self.assertEqual(reclaimed, index)
        lock.close()

    def test_atomic_no_clobber_preserves_completed_evidence(self):
        path = self.path / 'proof.json'
        parallel.atomic(path, {'proof': 1}, replace=False)
        with self.assertRaises(FileExistsError):
            parallel.atomic(path, {'proof': 2}, replace=False)
        self.assertEqual(parallel.read(path), {'proof': 1})
        self.assertFalse(list(self.path.glob('*.tmp')))

    def test_incomplete_journal_tail_is_excluded_without_mutating_original(self):
        path = self.path / 'journal.jsonl'
        complete = json.dumps({'plan_identity_sha256': 'old', 'rank': 1, 'outcomes': [self.outcome]}).encode() + b'\n'
        content = complete + b'{"unfinished":'
        path.write_bytes(content)
        self.assertEqual(list(parallel.journal(path, 'old', 1)), [self.outcome])
        self.assertEqual(path.read_bytes(), content)
        self.assertEqual(parallel.journal_boundary(path)['committed_bytes'], len(complete))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            list(parallel.journal(path, 'other', 1))

    def test_source_fields_user_turns_and_example_pairing_are_immutable(self):
        parallel.validate_outcomes([self.source], [self.outcome], self.generator)
        for mutate in ('source', 'messages', 'ordinal', 'example_id'):
            outcome = copy.deepcopy(self.outcome)
            if mutate == 'source':
                outcome['record']['source'] = 'sharegpt'
            elif mutate == 'messages':
                outcome['record']['messages'][0]['content'] = 'another question'
            elif mutate == 'ordinal':
                outcome['ordinal'] += 1
            else:
                outcome['example_id'] = 'other'
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                parallel.validate_outcomes([self.source], [outcome], self.generator)

    def test_exclusions_require_a_known_reason_and_no_record(self):
        excluded = {**self.outcome, 'status': 'excluded', 'reason': 'response_censored_at_generation_limit', 'record': None}
        parallel.validate_outcomes([self.source], [excluded], self.generator)
        with self.assertRaises(ValueError):
            parallel.validate_outcomes([self.source], [{**excluded, 'reason': 'unreviewed_filter'}], self.generator)

    def test_retryable_states_exclude_cancellation_and_ordinary_failures(self):
        self.assertTrue(parallel.retryable('NODE_FAIL'))
        self.assertTrue(parallel.retryable('TIMEOUT'))
        for state in ('CANCELLED', 'FAILED', 'OUT_OF_MEMORY', 'COMPLETED'):
            self.assertFalse(parallel.retryable(state))

    def test_completed_node_waits_for_nfs_visibility_without_resubmitting(self):
        value = {'directory': str(self.path)}
        plan = {**self.plan, 'secondary_job': '100'}
        job = {'JobState': 'COMPLETED', 'JobName': 'taps-parallel-gen', 'UserId': 'mohamadzbib(1)'}
        with patch.object(parallel, 'queue_complete', side_effect=[False, False, True]), patch.object(parallel, 'job_fields', return_value=job), patch.object(parallel.time, 'sleep') as pause, patch.object(parallel.subprocess, 'check_output') as output:
            self.assertIsNone(parallel.ensure_secondary(value, plan))
            pause.assert_called_once_with(2)
            output.assert_not_called()

    def test_retry_schedule_recovers_a_submitted_job_without_duplicate_submission(self):
        schedule = self.path / 'schedule.json'
        parallel.atomic(schedule, {'directory': str(self.path), 'comment': 'unique', 'command': ['sbatch', '--comment=unique']})
        value = {'directory': str(self.path), 'secondary_schedule': {'path': str(schedule)}}
        plan = {**self.plan, 'secondary_job': '100'}
        old = {'JobState': 'NODE_FAIL', 'JobName': 'taps-parallel-gen', 'UserId': 'mohamadzbib(1)'}
        jobs = 'JobId=101 JobState=PENDING JobName=taps-parallel-gen UserId=mohamadzbib(1) Comment=unique Partition=high-priority\n'
        with patch.object(parallel, 'job_fields', return_value=old), patch.object(parallel.subprocess, 'check_output', return_value=jobs) as output:
            self.assertEqual(parallel.ensure_secondary(value, plan), '101')
            self.assertEqual(output.call_count, 1)
        self.assertEqual(parallel.read(self.path / 'secondary-current.json')['previous_job_id'], '100')

    def test_recovery_counts_cpu_jobs_as_zero_and_recovers_after_history_publish(self):
        schedule = self.path / 'schedule.json'
        parallel.atomic(schedule, {'directory': str(self.path), 'comment': 'unique', 'command': ['sbatch', '--comment=unique']})
        value = {'directory': str(self.path), 'secondary_schedule': {'path': str(schedule)}}
        plan = {**self.plan, 'secondary_job': '100'}
        old = {'JobState': 'NODE_FAIL', 'JobName': 'taps-parallel-gen', 'UserId': 'mohamadzbib(1)'}
        jobs = 'JobId=99 JobState=RUNNING UserId=mohamadzbib(1) Partition=high-priority AllocTRES=cpu=64,gres/gpu=8\nJobId=98 JobState=RUNNING UserId=mohamadzbib(1) Partition=high-priority AllocTRES=cpu=16,mem=64G\n'
        real_atomic = parallel.atomic
        def interrupt_pointer(path, data, replace=True):
            if Path(path).name == 'secondary-current.json':
                raise RuntimeError('interrupted before pointer')
            real_atomic(path, data, replace=replace)
        with patch.object(parallel, 'job_fields', return_value=old), patch.object(parallel.subprocess, 'check_output', side_effect=[jobs, '101\n']), patch.object(parallel, 'atomic', side_effect=interrupt_pointer):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                parallel.ensure_secondary(value, plan)
        self.assertTrue((self.path / 'secondary-recovery-101.json').exists())
        self.assertFalse((self.path / 'secondary-current.json').exists())
        recovered = jobs + 'JobId=101 JobState=PENDING JobName=taps-parallel-gen UserId=mohamadzbib(1) Comment=unique Partition=high-priority\n'
        with patch.object(parallel, 'job_fields', return_value=old), patch.object(parallel.subprocess, 'check_output', return_value=recovered) as output:
            self.assertEqual(parallel.ensure_secondary(value, plan), '101')
            self.assertEqual(output.call_count, 1)
        self.assertEqual(parallel.read(self.path / 'secondary-current.json'), parallel.read(self.path / 'secondary-recovery-101.json'))

    def test_recovery_does_not_exceed_gpu_allowance_or_retry_user_cancellation(self):
        schedule = self.path / 'schedule.json'
        parallel.atomic(schedule, {'directory': str(self.path), 'comment': 'unique', 'command': ['sbatch', '--comment=unique']})
        value = {'directory': str(self.path), 'secondary_schedule': {'path': str(schedule)}}
        plan = {**self.plan, 'secondary_job': '100'}
        old = {'JobState': 'TIMEOUT', 'JobName': 'taps-parallel-gen', 'UserId': 'mohamadzbib(1)'}
        jobs = 'JobId=99 JobState=RUNNING UserId=mohamadzbib(1) Partition=high-priority AllocTRES=cpu=64,gres/gpu=16\n'
        with patch.object(parallel, 'job_fields', return_value=old), patch.object(parallel.subprocess, 'check_output', return_value=jobs) as output:
            with self.assertRaisesRegex(ValueError, 'allowance'):
                parallel.ensure_secondary(value, plan)
            self.assertEqual(output.call_count, 1)
        with patch.object(parallel, 'job_fields', return_value={**old, 'JobState': 'CANCELLED'}), patch.object(parallel.subprocess, 'check_output') as output:
            with self.assertRaisesRegex(ValueError, 'nonretryable'):
                parallel.ensure_secondary(value, plan)
            output.assert_not_called()

    def test_merge_is_ordered_idempotent_and_detects_modified_published_files(self):
        base = self.path / 'base.jsonl'
        item = {**self.outcome, 'ordinal': 0}
        base.write_bytes(parallel.native.canonical(item) + b'\n')
        parallel.atomic(self.path / 'bootstrap.json', {'test': True})
        plan = {'identity_sha256': 'plan', 'tasks': [], 'examples': 1, 'secondary_job': '100', 'second_node_qualification': [], 'base': {'path': str(base), 'sha256': parallel.pool.digest(base)}, 'original_completed_outcomes_preserved': 1}
        parallel.atomic(self.path / 'plan.json', plan)
        value = {'root': str(self.path), 'legacy': {'identity_sha256': 'old', 'target_id': 'target', 'target': {'model_identity_sha256': 'model'}}}
        with patch.object(parallel, 'cpu'), patch.object(parallel, 'bootstrap', return_value=value), patch.object(parallel, 'checked', return_value=plan), patch.object(parallel, 'original', return_value=self.generator):
            receipt = parallel.finish_locked(self.path, '100')
            self.assertEqual(receipt['counts'], {'processed': 1, 'accepted:train': 1})
            output = self.path / 'data/corpora' / (self.path.name + '-target-generated')
            self.assertEqual(list(parallel.records(output / 'native-train.jsonl')), [{'id': 'a'}])
            self.assertEqual(parallel.finish_locked(self.path, '100'), receipt)
            (output / 'train.jsonl').write_text('modified\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                parallel.finish_locked(self.path, '100')

    def test_merge_rejects_duplicate_ordinals_before_publication(self):
        base = self.path / 'base.jsonl'
        item = {**self.outcome, 'ordinal': 0}
        base.write_bytes((parallel.native.canonical(item) + b'\n') * 2)
        plan = {'identity_sha256': 'plan', 'tasks': [], 'examples': 1, 'secondary_job': '100', 'second_node_qualification': [], 'base': {'path': str(base), 'sha256': parallel.pool.digest(base)}}
        value = {'root': str(self.path), 'legacy': {}}
        with patch.object(parallel, 'cpu'), patch.object(parallel, 'bootstrap', return_value=value), patch.object(parallel, 'checked', return_value=plan), patch.object(parallel, 'original', return_value=self.generator):
            with self.assertRaisesRegex(ValueError, 'duplicate ordinal'):
                parallel.finish_locked(self.path, '100')
        self.assertFalse((self.path / 'data').exists())


if __name__ == '__main__':
    unittest.main()
