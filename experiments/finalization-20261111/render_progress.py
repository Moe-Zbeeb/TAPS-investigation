import argparse
import hashlib
import json
from pathlib import Path


def render_logits(report):
    rows = [row for row in report['results'] if row['task']['role'] == 'remaining_token_failure']
    if len(rows) != 1 or not report['all_original_prefix_captures_valid']:
        raise ValueError('Valid original-prefix capture is required')
    records = rows[0]['records']
    ar, sd = records['target_only'], records['speculative']
    a, b = ar['top5_token_ids'][0], sd['top5_token_ids'][0]
    if a == b:
        raise ValueError('The registered token disagreement is missing')
    values = [dict(zip(record['top5_token_ids'], record['top5_logits'])) for record in (ar, sd)]
    gaps = [value[a] - value[b] for value in values]
    if any(abs(gap - record['original_ar_minus_spec_logit']) > 1e-12 for gap, record in zip(gaps, (ar, sd))):
        raise ValueError('Saved top logits disagree with the reported token-pair gap')
    table = ['| Quantity | Target-only AR | Speculative tree |', '|---|---:|---:|']
    for token in (a, b):
        table.append(f'| Logit for token {token} | {values[0][token]:.17g} | {values[1][token]:.17g} |')
    table.append(f'| Logit({a}) − logit({b}) | {gaps[0]:+.17g} | {gaps[1]:+.17g} |')
    table.append(f'| Selected top token | {a} | {b} |')
    return '\n'.join(table)


def render_seed(report):
    if report['measured_generations'] != 1600 or report['training_admission']:
        raise ValueError('A complete diagnostic report is required')
    rows = [row for row in report['cells'] if 'arm' in row]
    if len(rows) != 4:
        raise ValueError('Both arms and benchmarks are required')
    table = ['| Benchmark | RNG policy | Question-mean τ | Published reference | Relative difference |', '|---|---|---:|---:|---:|']
    for row in rows:
        table.append(f"| {row['benchmark']} | {row['arm']} | {row['question_mean_tokens_per_cycle']:.6f} | {row['published_reference_tau']:.2f} | {100 * row['relative_error']:+.2f}% |")
    table.extend(['', '| Benchmark | Seed-zero minus hashed τ | Paired question bootstrap 95% interval |', '|---|---:|---:|'])
    for row in report['cells']:
        if 'contrast' in row:
            low, high = row['bootstrap_ci95']
            table.append(f"| {row['benchmark']} | {row['paired_question_mean_difference']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    return '\n'.join(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('logits', 'seed'), required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = args.input.read_bytes()
    report = json.loads(data)
    if report['status'] != 'complete':
        raise ValueError('Incomplete reports cannot produce result tables')
    text = {'logits': render_logits, 'seed': render_seed}[args.kind](report)
    args.output.write_text(text + '\n\nGenerated from report SHA256 `' + hashlib.sha256(data).hexdigest() + '`.\n')


if __name__ == '__main__':
    main()
