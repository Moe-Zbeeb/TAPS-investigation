# Execution status

The [registered experiment](PREREGISTRATION.md) completed successfully as job **2215900** on eight A100 80GB GPUs. The [result and evidence](../../results/fp32-decision-2026-11-11/README.md) identify the remaining FP32 decision as a near-tie argmax flip on an identical prefix. Seven controls retained their top token. Both replays of each mode reproduced their original probe tokens.

Version 1 stopped before model loading, and version 2's bounded cleanup wait also stopped without model loading. Version 3 verified and recorded the specific stale driver context before using the allocated free memory. All failed attempts remain recorded. Existing data generation, protected holders, source, models and scientific gates were not changed.

The next independent experiment is the [seed-policy control](../seed-policy-20261111/PREREGISTRATION.md), following the observed difference between upstream per-question seed zero and our registered hashed seeds. This diagnostic does not silently change the registered acceptance reduction or replace the original failed baseline.
