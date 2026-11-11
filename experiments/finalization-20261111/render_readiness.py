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
        name = stem + '-report.json'
        row = read(directory, name)
        assert row['status'] == 'passed' and len(row['ranks']) == 8 and all(item['passed'] for item in row['ranks'])
        lines.append(f'| {label} | 8/8 | {scope} |')
        names.append(name)
    vocab = read(directory, 'vocabulary-engineering.json')
    assert vocab['scope'] == 'engineering_vocabulary_preparation' and vocab['draft_vocab_size'] == 16
    lines += ['', 'The CPU vocabulary fixture passed with 32 training/eight dev examples and a 16-token draft vocabulary. It does not qualify the full corpus or the native 32,000-token draft vocabulary.', '', '## Source report SHA256', '']
    names.append('vocabulary-engineering.json')
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
