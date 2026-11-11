# Recorded engineering evidence

The [actual allocation-timeout recovery](data-2026-11-11/README.md) verified committed progress on all eight data workers under the protected successor allocation, with the next holder queued. This is completed recovery evidence; the full corpus remains in progress.

The [completed seed-policy comparison](seed-policy-2026-11-11/README.md) contains 1,600 measured generations and a separate CPU raw replay. It preserves the failed published-value comparison and verifies that the registered-hash arm reproduces all 800 earlier paired token/cycle records exactly.

The [completed remaining-token diagnostic](fp32-decision-2026-11-11/README.md) captures a same-prefix near-tie argmax flip and seven controls on eight GPUs. It refines the failure diagnosis without changing the original reproduction gate.

The [11 November status and diagnostics](status-2026-11-11/README.md) contain the latest completed replications, repeated FP32 failure classification and live allocation/data receipts. The [completion plan](../experiments/finalization-20261111/COMPLETION_PLAN.md) distinguishes these results from the research still required.

The [9 November 2026 receipts](engineering-2026-11-09/manifest.json) are unchanged copies from completed engineering runs. Their manifest records the original artifact paths, byte counts and SHA256 identities. Large raw logs, datasets and checkpoints remain outside this repository.

| Receipt | Evidence | Scope |
|---|---|---|
| [Official-backend smoke](engineering-2026-11-09/baseline-smoke-report.json) | EAGLE-2, EAGLE-3 and HASS; 48 paired greedy turns matched | Eight MT-Bench questions, two turns, 32 requested output tokens, one measured repetition |
| [Native training recovery](engineering-2026-11-09/native-training-recovery.json) | Eight ranks completed two optimizer updates; disk loading restored trainable model and optimizer hashes after deliberate perturbations | Authored engineering fixtures; same-engine recovery |
| [Stock serving smoke](engineering-2026-11-09/serving-smoke.json) | Eight servers completed 344 streamed requests, including warmup | Four target-only and four original EAGLE-3 vLLM servers |

These files establish the recorded execution checks. The smoke does not pass the full 22-cell reference gate, the checkpoint check does not establish mid-epoch restart, and the serving check does not establish a production speedup. EAGLE-3 and the other two methods use different target releases; the small smoke is not a controlled cross-backbone ranking.

Full reproduction tables and specialist, pooled-tree and serving research results will be generated from their complete registered raw records. No such completed research result is claimed here. The [implementation status](../experiments/IMPLEMENTATION_STATUS.md) identifies the remaining work.

The [full released-baseline panel](baselines-2026-11-10/README.md) completed 22,400 generations. Its strict reproduction gate failed; the linked report preserves all 22 cells, mismatches, intervals and input identities.

The [greedy numerical control](greedy-audit-2026-11-10/README.md) reproduced 24 selected FP16 failures and observed FP32 agreement at their disputed tokens, with original common prefixes verified. This diagnostic does not establish full-panel correctness.
