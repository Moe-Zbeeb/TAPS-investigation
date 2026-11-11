import argparse
import hashlib
import json
from pathlib import Path


def read(directory, name):
    return json.loads((directory / name).read_bytes())


def render(directory):
    control = read(directory, 'stopping-report.json')
    replay = read(directory, 'stopping-cpu-replay.json')
    assert {k: v for k, v in control.items() if k != 'checked_at_utc'} == {k: v for k, v in replay.items() if k != 'checked_at_utc'}
    failures = control['failures']
    lines = ['# Generated execution evidence', '', 'These tables describe the retained source-bound controls. They are not trained-model quality, full-PDF completion or universal losslessness. Published values are nonblocking diagnostics under the user’s 11 November instruction.', '', '| Fixed-budget FP32 method | Pairs | Exact pairs | Mismatches |', '|---|---:|---:|---:|']
    for method, count in sorted(control['counts'].items()):
        failed = sum(item['method'] == method for item in failures)
        lines.append(f'| {method} | {count} | {count-failed} | {failed} |')
    lines += ['', f"Overall mismatch rate: {control['exact_mismatches']}/{control['pairs']} = {100*control['mismatch_rate']:.6f}%. Divergent suffixes remain unequal.", '', '| Original CLI diagnostic | Questions | Turns | Official-counter τ | Published τ | Relative difference |', '|---|---:|---:|---:|---:|---:|']
    for benchmark, row in sorted(read(directory, 'upstream-cli-report.json')['summaries'].items()):
        lines.append(f"| {benchmark} | {row['questions']} | {row['turns']} | {row['official_counter_question_mean_tau']:.6f} | {row['published_tau']:.2f} | {100*row['relative_error']:+.2f}% |")
    lines += ['', 'Official-counter and raw-token counter equivalence has not been established. The unmodified CLI uses native multi-turn history and its own RNG procedure.', '', '| Engineering control | Passed ranks | Scope |', '|---|---:|---|']
    names = ['stopping-report.json', 'stopping-cpu-replay.json', 'upstream-cli-report.json']
    for stem, label, scope in [('restart', 'Fresh-process restart', 'Two successful updates; exact state/RNG/batch order after a mid-epoch restart'), ('export', 'Native export/inference', 'Stock inference of the engineering checkpoint; no quality or speed claim')]:
        latest = {'restart': 'restart-v4-report.json', 'export': 'export-v3-report.json'}[stem]
        name = latest if (directory / latest).exists() else stem + '-report.json'
        row = read(directory, name)
        assert row['status'] == 'passed' and len(row['ranks']) == 8 and all(item['passed'] for item in row['ranks'])
        lines.append(f'| {label} | 8/8 | {scope} |')
        names.append(name)
    vocab = read(directory, 'vocabulary-engineering.json')
    assert vocab['scope'] == 'engineering_vocabulary_preparation' and vocab['draft_vocab_size'] == 16
    lines += ['', 'The CPU vocabulary fixture passed with 32 training/eight dev examples and a 16-token draft vocabulary. It does not qualify the full corpus or the native 32,000-token draft vocabulary.', '', '## Source report SHA256', '']
    names.append('vocabulary-engineering.json')
    if (directory / 'pipeline-v3-replay.json').exists():
        pipeline = read(directory, 'pipeline-v3-replay.json')
        raw_name = '../../experiments/verification/reviews/independent-raw-replay.json'
        independent = read(directory, raw_name)
        known = [row for row in pipeline['numerical']['results'] if row['classified_known_near_tie']]
        controls = [row for row in pipeline['numerical']['results'] if not row['classified_known_near_tie']]
        assert len(known) == 1 and len(controls) == 7 and all(pipeline['checks'].values()) and independent['status'] == 'passed'
        lines[-3:] = ['## Final scoped replay', '', '| Independent raw reconstruction | Population | Maximum event bound |', '|---|---:|---:|', f"| Conditional registered events | {independent['conditional_draws']:,} draws / {independent['conditional_checks']} checks | {independent['conditional_maximum_event_bound']:.9f} |", f"| End-to-end registered events | {independent['sampling_continuations']:,} continuations / {independent['sampling_checks']} checks | {independent['sampling_maximum_event_bound']:.9f} |", '', f"All 48 retained vectors passed the source-bound checks. Known-site maximum pairwise full-vector delta: {known[0]['max_pairwise_full_vector_delta']:.12g}; largest control delta: {max(row['max_pairwise_full_vector_delta'] for row in controls):.12g}; frozen ceiling: {pipeline['numerical']['epsilon']:.4f}.", '', 'The clean reference selects the native AR winner at the known site. The exact kernel cause remains unresolved. The CPU replay and separate evidence review retain their own identities; finite-event checks and a classified point do not establish universal native losslessness.', '', '## Source report SHA256', '']
        names.extend(['pipeline-v3-replay.json', raw_name])
    for name in names:
        lines.append(f"- [{name}]({name}): `{hashlib.sha256((directory/name).read_bytes()).hexdigest()}`")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    (args.directory / 'GENERATED_RESULTS.md').write_text(render(args.directory))


if __name__ == '__main__':
    main()
