# Eight-GPU engineering check, 9 November 2026

All eight A100 80GB workers passed. This was an engineering test, not a published baseline reproduction. The immutable raw evidence is `outputs/engineering/engineering-20261109-v2/`.

Each worker trained both reference expert modes for two optimizer steps, stopped and resumed from a committed checkpoint, checked all declared token/step counters, ran GPU tree verification and four merge algorithms, and exported aligned features from a tiny GPU teacher. Released EAGLE-3 and a real pooled adapter containing two identical original EAGLE-3 drafters then matched target-only greedy output on the same 30-token completion. The pooled test used uncalibrated scores explicitly; it says nothing about specialist quality.

| Single engineering prompt | Median synchronized wall time across eight GPUs | Verification cycles | Raw generated tokens |
|---|---:|---:|---:|
| Target-only, drafter still resident | 1928.8 ms | 30 | 30 |
| Released EAGLE-3 | 733.6 ms | 7 | 30 |
| Two identical pooled drafters | 799.4 ms | 7 | 30 |

Peak allocated memory was 18.95 GiB per GPU. Initialization temporarily loads an extra target in the pooled adapter; its raw metadata records this. This tiny un-warmed check is not a speedup estimate, and target-only memory is not a valid serving memory baseline.

The first attempt failed at the intended model-storage symlink guard before training. The second attempt used the corrected guard, and the failed manifest and logs remain available.

Actual official posterior audits also completed for EAGLE-3, EAGLE-2 and HASS: four finite-vocabulary cases each, 100,000 trials per case, simultaneous 95% bounds. The largest TV upper bound was 0.01402 against a predeclared 0.02 margin. These audits do not establish full-model output-distribution equivalence.

## Official-backend smoke

All three methods completed eight MT-Bench questions with both turns at T=0, 32 requested output tokens and one measured repetition. One warmup per mode was excluded. All 48 target-only/speculative turn pairs matched, including the first terminal token and common cap.

| Method | Canonical raw tokens/cycle | Comparison tokens/s | Paired measured time ratio | Greedy mismatches |
|---|---:|---:|---:|---:|
| eagle2 | 3.662 | 76.7 | 2.17 | 0 / 16 |
| eagle3 | 5.584 | 90.6 | 2.87 | 0 / 16 |
| hass | 4.245 | 91.4 | 2.37 | 0 / 16 |

These are small, capped smoke measurements, not the full reference benchmark or a cross-backbone ranking. EAGLE-3 uses Llama-3.1; EAGLE-2 and HASS use Llama-3. The 22-cell published-number gate remains ineligible. Full raw records, confidence intervals and the exact snapshot are under `outputs/runs/baseline-smoke-20261109-v1/`.

A separate CPU tokenizer preflight checked all 880 first-turn method/benchmark prompts. All fit the registered effective context limit of 1,979 tokens. Five EAGLE-3 CNN/DM prompts leave fewer than the requested 512 output tokens; the GPU runner records their actual caps and checks later turns before generation. No prompts were truncated. The receipt is `outputs/engineering/context-preflight-20261109.json`.

## Stock serving integration

`outputs/serving-pools/serving-smoke-20261109-v3/` passed all eight workers: four target-only and four original EAGLE-3 stock vLLM servers, one per A100. Each completed three warmup requests and five measured batches of eight concurrent requests. All 344 actual HTTP streams completed; response usage recorded 5,504 total tokens, including 5,120 tokens across 320 measured requests. Every request produced the required 16 tokens.

The controller cleaned up only its own eight process groups, and a fresh device check found all eight GPUs idle. The completion receipt binds the raw request logs and cleanup proof. Separate earlier failures exposed an IPC-path length limit and a missing setuptools dependency; those attempts remain archived. This validates startup and real concurrent requests, not production traffic, custom pooled-tree serving or a scientific speedup comparison.

## Native EAGLE-3 training

The pinned native architecture and online target-feature recipe completed two successful optimizer updates on every one of eight ranks in `outputs/native-training/native-startup-20261109-v6/`. There were zero skipped updates. Across all data-parallel ranks it processed 32 microbatches, 5,216 nonpadding tokens and 448 input assistant-mask tokens from authored engineering data. Reduced-vocabulary, future-position loss coverage is not exposed by these input counters, and no matched-compute claim follows from them.

The actual disk checkpoint recovered both the complete trainable-model state and complete local ZeRO2 optimizer state after explicit perturbations, on all eight ranks. Each worker changed one owned trainable weight and one owned FP32 Adam moment, proved both hashes changed, loaded the checkpoint, and recovered both original hashes exactly. Counters, learning rates and client state also matched. No frozen target parameter was selected. A fresh device check confirmed all eight GPUs idle after completion. This establishes same-engine recovery; mid-epoch data/RNG restart remains unsupported. The preceding v5 save/load invariance check is preserved separately.

The native upstream bytes, four data workers, sampler and pinning semantics are preserved; the external entrypoint adapter prevents spawn workers from re-executing training setup. Earlier runtime and audit-helper failures remain recorded. These tiny updates establish execution, not drafter quality or native-recipe reproduction on research data.

The final local source suite passed 323 tests, with eight Torch-dependent tests skipped locally. Actual CPU Torch integration and CUDA behavior are supported by separate cluster receipts. Source checks found no authored Python comments.

## Real-target data and feature integration

Both eight-GPU data exports completed: original authored responses, and target-regenerated responses followed by feature extraction. Each contains eight engineering examples. Features come from Llama-3.1-8B hidden-state indices2/16/29 and have12,288 dimensions. Per-turn regeneration records include actual seeds, token counts, termination and measured generation time. The complete small caches are copied under data/exports/ alongside their frozen plans and hashes.

Slurm CPU step88 loaded and validated all eight actual tensors, including alignment, masks, domains and hashes. One reference-prototype update used only the three training examples (3train/3dev/2test):68 supervised tokens,198 nonpadding tokens/input slots and one optimizer step, all matching the declared budget. The checkpoint reloaded successfully; CUDA stayed uninitialized during this additional CPU check. This is cache/trainer integration evidence, not a native EAGLE-3 or model-quality result.

## Cluster cleanup

All102 planned shared research targets were removed. Preservation verification rehashed17,623,732,403 bytes of original model files and passed both standalone runtime activations. The deep10 holder and continuation remain active. Accessible-node cache cleanup is recorded separately; drained nodes7 and8 were not inspected. Cleanup receipts are retained in the original workspace outputs/taps-setup-20261101/.
