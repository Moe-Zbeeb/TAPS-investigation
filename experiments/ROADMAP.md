# Research stages and dependencies

The source of requirements is the unchanged `TAPS_rebuild_plan.pdf`. The current runnable milestone is official-baseline reproduction. Later stages remain conditional on its results.

| Stage | Required evidence | Next action |
|---|---|---|
| Original-paper audit | Original raw records, source scripts and independently checked table/figure inconsistencies | Prepare a factual correction proposal for the authors; no automated arXiv change or external message |
| Official reproduction | Pinned sources/checkpoints/data, complete paired logs, acceptance agreement and verification checks | Retrain a generalist on the official ShareGPT/UltraChat recipe |
| RQ1 pilot | One target, chat/math/code, original versus target-regenerated responses, matched tokens/compute, three seeds | Estimate the specialist-oracle minus generalist gap on held-out mixed traffic |
| RQ1 expansion | Interpretable pilot and fixed data/split definitions | Two families and sizes, a MoE target, five domains, capacity/data/mixture sweeps |
| RQ2 | A meaningful specialization gap and validated sampling | Fixed-budget pooled candidates, duplicate-path merging and held-out calibration |
| RQ3 | Matched specialists and a shared generalist initialization | Shared-body experts and fair weight-space composition |
| RQ4 | Validated methods and separate compatible serving environments | Measure latency, throughput and memory over batch sizes and traffic mixes |

The full grid's fifth domain, exact training-response generation policy, matched token/compute budgets and serving-environment versions must be fixed before those jobs are runnable. The plan suggests multilingual data or retrieval/summarization for the fifth domain. These are research choices, not silently selected defaults.

The roughly 10% specialization-gap branch is a decision criterion to assess with uncertainty. If a matched generalist closes the gap, report that result and focus on the explanation and mixture recipe. If composition helps only at batch one, report its latency benefit and measured batch-size crossover rather than a general throughput claim.

Historical TAPS tables and figures are preserved only as references. No historical value is imported as a new measurement. The local folder initially supplied only the rebuild memo; claims about the origin of old results require their original records and independent verification.
