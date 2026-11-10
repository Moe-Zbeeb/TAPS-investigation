# Selected greedy-failure audit · 10 November 2026

This diagnostic investigates the full baseline panel's failed T=0 token comparisons before further training. It is not a benchmark rerun or an estimate of failure prevalence.

Select repeat zero only, retain both modes' original tokenized input, seeds, output tokens and source/model identities, and take eight failures per method. Select the earliest first disagreement within each benchmark, with round-robin benchmark coverage. Every replay must report whether it reproduces the original prefix and whether the captured logits condition on the original common prefix.

Run one worker on each of the eight held A100 80GB GPUs, separately for EAGLE-3, EAGLE-2 and HASS. Repeat every paired replay twice in FP16, then repeat the same selected cases with freshly loaded FP32 weights. The generation cap is shortened to four tokens beyond the original disagreement; tree budgets and source implementations remain unchanged. Observe the official posterior and forward functions without replacing their decisions. Capture the verifier logits that select the disputed token, their top-five tokens, top-two margin and the relative logits of the original AR/speculative token pair.

The driver records missing captures, prefix validity, exact replay repeatability and logit differences. A precision change may change earlier tokens; a capture whose prefix differs is not evidence about the original disputed decision. Selected-case behavior cannot establish global greedy equivalence or full-model sampling correctness. The original failed benchmark report remains the reference result.

The source is a frozen site-specific experiment driver, not a general allocation launcher. Its historical run IDs and allocation guard must never be treated as evidence that an allocation is still owned. Verify the live holder and successor before any new submission and use a fresh output directory. GPU work must remain inside the verified allocation. The recorded execution artifacts are under `outputs/greedy-audits/official-20261110-v1` on the cluster.

## Recorded outcome

All six waves completed on 10 November: 48 workers and 96 paired replays. The 24 original FP16 disagreements reproduced; all 24 FP32 controls agreed at the disputed token, with the original common prefix valid in every capture. See [the diagnostic results](../../results/greedy-audit-2026-11-10/README.md) for per-method measurements and limits. The failed full-panel benchmark gate remains unchanged.
