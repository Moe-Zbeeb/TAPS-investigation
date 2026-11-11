# Research stages and dependencies

See the [11 November completion plan](finalization-20261111/COMPLETION_PLAN.md) for current evidence, explicit exit criteria, resource scheduling and the remaining work for every memo section.

The [remaining FP32 token decision](../results/fp32-decision-2026-11-11/README.md) completed on eight GPUs with seven matched controls. The [seed-policy diagnostic](../results/seed-policy-2026-11-11/README.md) completed 1,600 generations and a separate CPU raw replay; upstream seed zero did not resolve the GSM8K outlier. The [review packet](verification/REVIEW_PACKET_20261111.md) makes the required validation review concrete. These diagnostics preserve the failed original gate.

The source of requirements is the unchanged `TAPS_rebuild_plan.pdf`, with the user's 11 November instruction superseding paper-number matching as a prerequisite. The current runnable milestone is independently admitted pipeline validation and real-data generalist training. Historical paper comparisons remain diagnostic. See the [v2 policy](verification/PIPELINE_VALIDATION_V2.md) and [live execution status](STATUS_20261111.md).

| Stage | Required evidence | Next action |
|---|---|---|
| Historical paper comparison | Retained raw records, source scripts and reported differences | Nonblocking diagnostic; no further paper-matching search is required |
| Pipeline validation and generalist | Pinned sources/checkpoints/data, complete paired logs, scoped numerical/statistical checks and independent review | Train and evaluate the native ShareGPT/UltraChat generalist after data QA |
| RQ1 pilot | One target, chat/math/code, original versus target-regenerated responses, matched tokens/compute, three seeds | Estimate the specialist-oracle minus generalist gap on held-out mixed traffic |
| RQ1 expansion | Interpretable pilot and fixed data/split definitions | Two families and sizes, a MoE target, five domains, capacity/data/mixture sweeps |
| RQ2 | A meaningful specialization gap and validated sampling | Fixed-budget pooled candidates, duplicate-path merging and held-out calibration |
| RQ3 | Matched specialists and a shared generalist initialization | Shared-body experts and fair weight-space composition |
| RQ4 | Validated methods and separate compatible serving environments | Measure latency, throughput and memory over batch sizes and traffic mixes |

The full grid's fifth domain, exact training-response generation policy, matched token/compute budgets and serving-environment versions must be fixed before those jobs are runnable. The plan suggests multilingual data or retrieval/summarization for the fifth domain. These are research choices, not silently selected defaults.

The roughly 10% specialization-gap branch is a decision criterion to assess with uncertainty. If a matched generalist closes the gap, report that result and focus on the explanation and mixture recipe. If composition helps only at batch one, report its latency benefit and measured batch-size crossover rather than a general throughput claim.

Historical TAPS tables and figures are preserved only as references. No historical value is imported as a new measurement. The local folder initially supplied only the rebuild memo; claims about the origin of old results require their original records and independent verification.
