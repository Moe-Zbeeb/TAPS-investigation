# Greedy numerical control · 10 November 2026

All 24 selected FP16 failures reproduced the original token disagreement. Loading the same released checkpoints in FP32 produced agreement at the disputed token in all 24 cases. Every captured decision in both precisions used the original common token prefix, and both repeated replays were identical. This is evidence supporting numerical precision as the explanation for these selected failures. It does not establish full-panel correctness or change the failed registered reproduction gate.

## Measurements

The [frozen protocol and driver](../../experiments/greedy-audit-20261110/README.md) selected eight repeat-zero failures per method, prioritizing early disagreements with benchmark coverage. Six waves used all eight A100 80GB GPUs: three methods in FP16, then the same methods in FP32. The run completed 48 workers and 96 paired replays. Each worker ran both generation modes twice.

| Method | Loaded dtype | Original common prefix valid | Agreement at disputed token | Range of maximum absolute logit difference |
|---|---|---:|---:|---:|
| eagle3 | float16 | 8/8 | 0/8 | 0.0195312–0.0312500 |
| eagle3 | float32 | 8/8 | 8/8 | 0.0000315–0.0000496 |
| eagle2 | float16 | 8/8 | 0/8 | 0.0195312–0.0312500 |
| eagle2 | float32 | 8/8 | 8/8 | 0.0000343–0.0000439 |
| hass | float16 | 8/8 | 0/8 | 0.0224609–0.0429688 |
| hass | float32 | 8/8 | 8/8 | 0.0000362–0.0000648 |

At the disputed position, FP16 top-two margins were between zero and 0.03125. Sequential and tree verifier calls returned slightly different logits despite identical conditioning prefixes, changing the argmax at these close decisions. In FP32 the captured argmax agreed in all 24 cases and the observed logit differences were much smaller. FP32 can change which token the target-only path chooses; agreement is between the two FP32 paths, not a claim that FP32 preserves the original FP16 answer.

The [derived report](report.json) includes per-case token decisions, prefix hashes, top-five logits, repeatability checks and the SHA256 of every original worker receipt. The [run manifest](manifest.json) binds the frozen driver, selected cases and physical GPU inventory; the [completion receipt](completion.json) records successful execution. The driver in this repository has the recorded SHA256. Original inputs, complete replay tokens and raw receipts remain under the cluster output directory and local ignored `outputs/`; they are not republished as a dataset.

## Consequence for the next experiment

The diagnostic supports investigating a registered numerical correction before rerunning the full greedy panel. It cannot justify silently accepting the previous 706 mismatched pairs. The three EAGLE-3 T=1 acceptance outliers are a separate unresolved question, and full-model T=1 distributional validation remains required before specialist training. FP32 was a diagnostic control; it has not been adopted as the benchmark configuration or validated for throughput.

The [post-run cluster snapshot](post-run-cluster-state.json) records eight reserved GPUs, zero compute processes and an intact holder continuation after the diagnostic finished. It is a timestamped observation, not a guarantee about later queue state.
