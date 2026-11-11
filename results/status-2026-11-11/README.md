# Live status and read-only diagnostics · 11 November 2026

These receipts record completed independent replications and a live cluster check. They do not replace the original failed reference panel or admit training. See the [completion plan](../../experiments/finalization-20261111/COMPLETION_PLAN.md) for remaining PDF requirements.

- [Diagnostic audit](diagnostic-audit.json): CPU-only Slurm step `2212576.150` verified completed worker/report hashes and compared two FP32 panels. The executed [audit source](../../experiments/finalization-20261111/audit_results.py) has SHA256 `f2dd2b3fa6138860ad033c601e31a9c939f24d22034c8677fa0b474ac3403e12`. It is a fixed-run diagnostic; its training-state statement is corroborated separately by the live Slurm queue, not a general training-discovery mechanism.
- [Cluster snapshot](cluster-snapshot.json): at 00:03 UTC / 02:03 Beirut, only holder `2212576` had GPUs allocated to this user: eight A100s on deep10. `2212580` is its pending protected successor. Data coordinator `2214977` was running; continuation `2214983`, admission `2214978` and profile `2214979` were pending. Validation/replication jobs `2214872` and `2215000` completed with exit zero. A successful job exit does not mean its scientific gate passed.
- [GPU telemetry](gpu-telemetry.json): an instantaneous read-only sample inside the owned allocation showed 93–96% utilization on all eight A100 80GB GPUs. This is not an average utilization or training-compute measurement.

| New replication | Acceptance cells within 5% | Strict cells passed |
|---|---:|---:|
| EAGLE-3 | 8/10 | 3/10 |
| EAGLE-2 | 6/6 | 3/6 |
| HASS | 6/6 | 3/6 |
| Combined diagnostic counts | 20/22 | 9/22 |

The union is a diagnostic summary of two completed runs, not a newly admitted reference group. The replication seed/configuration differs from the frozen original reference-group configuration. Question-mean acceptance is the registered statistic; the receipt retains pooled acceptance only as a diagnostic. The remaining acceptance outliers are EAGLE-3 T=1 GSM8K (+20.14%) and MT-Bench (+7.68%). All original FP16 greedy reference cells still fail their strict comparison requirements.

Both FP32 controls contain 1,120 pairs, with the same three EAGLE-3 failures and identical details on two nodes. CNN/DM questions 295 and 288 have identical shared token prefixes and different context-stop lengths. Metadata records AR overshoot by one and SD overshoot by three/two tokens at the same effective context limit. MT-Bench question 146 turn 1 disagrees at token 200; its cause remains unresolved. Explaining two failures does not change either failed report.

At 23:59 UTC / 01:59 Beirut, eight generation journals recorded 133,120 / 540,107 processed source inputs (24.65%) and no corpus completion marker. Generated-token counters include excluded outputs. The actual retained training population requires completion validation and native filtering. The conditional and end-to-end sampling checks remain bounded passing evidence from the [earlier replay](../data-2026-11-10/README.md); independent review and overall admission remain blocked.

SHA256 identities for these copied small receipts are in [SHA256SUMS](SHA256SUMS). Large raw logs, data and weights remain outside Git. No active source, models, gates, holder or healthy workload were changed by this diagnostic.
