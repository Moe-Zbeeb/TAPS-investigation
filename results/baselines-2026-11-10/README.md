# Full released-baseline results · 10 November 2026

The registered panel completed all 22 cells and 22,400 generation records with complete paired coverage. **The reproduction gate failed: 8/22 cells passed.** Acceptance length was within the registered 5% relative tolerance in 19/22 cells. Every temperature-zero cell contained token disagreements with its paired target-only control. These are execution results, not successful losslessness or training validation.

The immutable [report](report.json), [input and hardware manifest](manifest.json), and [collection receipt](collection.json) were regenerated and verified from both completed runs on 10 November. Their hashes are bound by the collection receipt. The original raw generation files remain on the cluster and are excluded from Git; no benchmark inputs or generation records were changed during collection.

## Measured reference cells

Tau is raw emitted tokens divided by explicitly counted verification cycles, pooled across a question's turns within each repetition, averaged across repetitions and then questions. The [frozen protocol](../../experiments/official-reproduction-20261109/PREREGISTRATION.md) specifies this reduction. The JSON report includes question-level bootstrap intervals and separately labeled pooled and equal-turn diagnostics.

| Method | Panel | T | Measured tau | Published target | Relative error | T=0 mismatched pairs | Strict gate |
|---|---|---:|---:|---:|---:|---:|---|
| eagle2 | gsm8k | 0 | 4.343 | 4.42 | 1.75% | 20 | fail |
| eagle2 | gsm8k | 1 | 4.192 | 4.30 | 2.51% | — | pass |
| eagle2 | humaneval | 0 | 4.919 | 4.93 | 0.23% | 50 | fail |
| eagle2 | humaneval | 1 | 4.685 | 4.73 | 0.95% | — | pass |
| eagle2 | mt_bench | 0 | 4.219 | 4.21 | 0.21% | 90 | fail |
| eagle2 | mt_bench | 1 | 3.894 | 3.90 | 0.16% | — | pass |
| eagle3 | alpaca | 0 | 6.827 | 6.70 | 1.90% | 60 | fail |
| eagle3 | alpaca | 1 | 5.578 | 5.56 | 0.32% | — | pass |
| eagle3 | cnn_dm | 0 | 5.395 | 5.34 | 1.04% | 71 | fail |
| eagle3 | cnn_dm | 1 | 4.453 | 4.39 | 1.44% | — | pass |
| eagle3 | gsm8k | 0 | 6.342 | 6.23 | 1.79% | 65 | fail |
| eagle3 | gsm8k | 1 | 5.449 | 4.59 | 18.71% | — | fail |
| eagle3 | humaneval | 0 | 6.830 | 6.74 | 1.33% | 45 | fail |
| eagle3 | humaneval | 1 | 6.128 | 5.82 | 5.29% | — | fail |
| eagle3 | mt_bench | 0 | 6.156 | 6.13 | 0.43% | 125 | fail |
| eagle3 | mt_bench | 1 | 4.546 | 4.24 | 7.23% | — | fail |
| hass | gsm8k | 0 | 4.952 | 5.02 | 1.35% | 25 | fail |
| hass | gsm8k | 1 | 4.740 | 4.85 | 2.26% | — | pass |
| hass | humaneval | 0 | 5.524 | 5.54 | 0.29% | 40 | fail |
| hass | humaneval | 1 | 5.260 | 5.30 | 0.75% | — | pass |
| hass | mt_bench | 0 | 4.679 | 4.68 | 0.02% | 115 | fail |
| hass | mt_bench | 1 | 4.322 | 4.26 | 1.46% | — | pass |

## What remains unresolved

There were 366/2,400 repeated greedy pairs with token mismatches for EAGLE-3, 160/1,600 for EAGLE-2 and 180/1,600 for HASS: 706/5,600 in total. These denominators count five repeated paired generations, not unique prompts. All reported first disagreements are token disagreements rather than length-only disagreements. The short engineering smoke used eight MT-Bench questions and a 32-token cap, so its earlier success does not establish this full-panel result.

The three acceptance-length outliers are EAGLE-3 at temperature one: GSM8K (+18.71%), MT-Bench (+7.23%) and HumanEval (+5.29%). Their cause is unestablished. GSM8K's separately reported pooled tau is 4.499 versus question-mean 5.449; the published target is 4.59. This difference motivates checking reduction conventions but does not justify substituting a post hoc metric or changing the registered gate. The pinned upstream speed script computes speed, without providing an acceptance-length reducer that resolves this question.

A selected-failure replay diagnostic is recorded in [the audit protocol](../../experiments/greedy-audit-20261110/README.md). Full-model temperature-one distributional correctness remains unvalidated. Specialist training remains gated.

## Hardware and interpretation

All drafters are the original released checkpoints, with no new training. EAGLE-3 uses Llama-3.1-8B-Instruct; EAGLE-2 and HASS use Llama-3-8B-Instruct. Exact source and checkpoint identities are retained in the manifest.

EAGLE-3 ran on nine A100 40GB GPUs on one node, with driver 565.57.01. EAGLE-2 and HASS ran in separate waves on eight A100 80GB GPUs on another node. Each method retained its co-located target-only control. Cross-node raw throughput does not establish a fair cross-method ranking. Tau is not wall-clock speedup, and no equal-output T=0 wall-time speedup is claimed for failed token comparisons.

The NFS collector needed a writable descriptor for its exclusive file lock. That isolated post-run patch changes neither generation nor statistical reduction. The collector SHA is recorded in the manifest; original generation source hashes remain frozen. The failed first collection attempt is retained on the cluster. The regression test simulates NFS rejecting an exclusive lock on a read-only descriptor and verifies that existing lock bytes remain intact.
