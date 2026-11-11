# Actual allocation-timeout recovery · 11 November 2026

At 14:49 Beirut, all eight target-generation workers had committed new examples after the original allocation timed out. The same frozen plan resumed automatically on protected holder **2212580**, managed by CPU continuation **2214983**. The next protected holder, **2215920**, is already queued. No holder was cancelled or released by this check.

The original holder 2212576 reached its normal 48-hour deadline. Original CPU coordinator 2214977 then exited with failure because its attached GPU step was terminated for that time limit. This expected failure triggered the queued continuation; it is not a new model-computation failure. The resumed attempt has the same plan identity, original deployed-source identity and target-generation settings.

| Worker | Prior committed inputs | After recovery | New committed inputs |
|---|---:|---:|---:|
| 0 | 40,896 | 41,024 | 128 |
| 1 | 38,976 | 39,040 | 64 |
| 2 | 37,568 | 37,632 | 64 |
| 3 | 38,592 | 38,656 | 64 |
| 4 | 40,960 | 41,088 | 128 |
| 5 | 40,896 | 41,024 | 128 |
| 6 | 40,768 | 40,832 | 64 |
| 7 | 40,768 | 40,832 | 64 |

Generated from report SHA256 `bc5672519ae33b161e75f42e0876d9d1eddbf1c13c537fd30374069e8e3d1278`.


The table is generated from [the recovery receipt](rollover.json) by [the renderer](../../experiments/finalization-20261111/render_progress.py). Each worker's count increased and its per-attempt elapsed counter restarted, demonstrating actual committed response generation rather than merely a running scheduler state.

At the progress snapshot, **320,128/540,107 inputs were processed (59.27%)**. These include exclusions; retained training examples still require final corpus completion and native tokenization/filtering. A read-only telemetry sample at 14:46 Beirut recorded all eight A100 80GB GPUs active at 84–94% utilization, with 68,781–69,297 MiB used per GPU. This sample is not an average performance measurement.

The [receipt](rollover.json) retains the verified holder fields, exact attempt/plan, before-and-after counters, original timeout accounting, queue dependencies and telemetry. The deployed-source identity remains `6375933ffe6e688eb4eb032360eabe519efadd01e13233a56869a022bf6f2caa`. The stale local allocation-pointer mirror was refreshed from the verified remote holder; no remote holder authority or service was changed.

CPU native admission and the 200-update profile remain pending on successful data completion and the unchanged scientific gate. No scientific training or full forty-epoch reproduction is claimed. The two completed [numerical](../fp32-decision-2026-11-11/README.md) and [seed-policy](../seed-policy-2026-11-11/README.md) experiments refine the reproduction diagnosis; independent review and the published-acceptance discrepancy remain unresolved.

Receipt SHA256 values are in [SHA256SUMS](SHA256SUMS). Data generation and its continuation are Slurm-managed and do not depend on the local SSH session staying open.
