# Recorded engineering evidence

The [9 November 2026 receipts](engineering-2026-11-09/manifest.json) are unchanged copies from completed engineering runs. Their manifest records the original artifact paths, byte counts and SHA256 identities. Large raw logs, datasets and checkpoints remain outside this repository.

| Receipt | Evidence | Scope |
|---|---|---|
| [Official-backend smoke](engineering-2026-11-09/baseline-smoke-report.json) | EAGLE-2, EAGLE-3 and HASS; 48 paired greedy turns matched | Eight MT-Bench questions, two turns, 32 requested output tokens, one measured repetition |
| [Native training recovery](engineering-2026-11-09/native-training-recovery.json) | Eight ranks completed two optimizer updates; disk loading restored trainable model and optimizer hashes after deliberate perturbations | Authored engineering fixtures; same-engine recovery |
| [Stock serving smoke](engineering-2026-11-09/serving-smoke.json) | Eight servers completed 344 streamed requests, including warmup | Four target-only and four original EAGLE-3 vLLM servers |

These files establish the recorded execution checks. The smoke does not pass the full 22-cell reference gate, the checkpoint check does not establish mid-epoch restart, and the serving check does not establish a production speedup. EAGLE-3 and the other two methods use different target releases; the small smoke is not a controlled cross-backbone ranking.

Full reproduction tables and specialist, pooled-tree and serving research results will be generated from their complete registered raw records. No such completed research result is claimed here. The [implementation status](../experiments/IMPLEMENTATION_STATUS.md) identifies the remaining work.

The [full released-baseline panel](baselines-2026-11-10/README.md) completed 22,400 generations. Its strict reproduction gate failed; the linked report preserves all 22 cells, mismatches, intervals and input identities.
