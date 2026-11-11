# Seed-policy execution

**Completed:** job 2215909 finished all 1,600 measured generations on eight GPUs. The [generated results](../../results/seed-policy-2026-11-11/README.md) show that upstream seed zero does not resolve the GSM8K acceptance outlier. CPU replay 2215919 recomputed the statistics exactly and verified all 800 registered-hash token/cycle replays against the prior replication. The original failed gate remains unchanged.

Job **2215909** was submitted with a dependency on successful token-decision job 2215900, account `grad-students`, partition `high-priority`, QoS `spec-priority`, one exclusive eight-A100 node and a two-hour limit. The [registration](PREREGISTRATION.md) fixes all 160 first-turn inputs, both RNG policies, 1,600 measured generations, 24 excluded warmups and the unchanged question-mean acceptance reduction.

This is an independent pipeline diagnostic. It does not admit training, resolve a broader multi-turn protocol, or overwrite an earlier report. The exact copied source bundle is `outputs/preparation/seed-policy-20261111-v1`; raw outputs belong to `outputs/seed-policy/first-turn-20261111-v1` on the cluster. The original verifier, released drafter and native posterior are unchanged.

The [review packet](../verification/REVIEW_PACKET_20261111.md) identifies the independent review that remains required before scientific training. The real-data training profile remains separately gated.
