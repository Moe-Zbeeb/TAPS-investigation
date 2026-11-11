# ShareGPT and UltraChat preparation

This stage prepares real conversation text for native EAGLE-3 training. It does not replace the failed released-checkpoint baseline or authorize specialist training. The registered choices are in [the corpus configuration](../../configs/corpus-sharegpt-ultrachat-20261110.json).

## Completed source preparation

CPU job `2214960` produced `data/corpora/sharegpt-ultrachat-20261110-v2`. All seven source files were checked against their pinned revisions, sizes and SHA256 identities: 2,135,351,205 bytes in total.

| Population | Train | Development | Held-out test | Total |
| --- | ---: | ---: | ---: | ---: |
| Canonical conversations | 529,031 | 5,539 | 5,537 | 540,107 |
| Complete original answers | 278,300 | 2,902 | 2,913 | 284,115 |
| Need target completions | 250,731 | 2,637 | 2,624 | 255,992 |

The raw inputs contain 94,145 ShareGPT rows and 463,897 UltraChat rows. ShareGPT uses `anon8231489123/ShareGPT_Vicuna_unfiltered`, revision `192ab2185289094fc556ec8ce5ce1e8e587154ca`, and its V3 cleaned split file. UltraChat uses `HuggingFaceH4/ultrachat_200k`, revision `8049631c405ae6576f93f445c6b8166f76f5505a`, with all six `train_sft` and `train_gen` files. Its official test files are excluded.

The EAGLE-3 paper describes approximately 68k ShareGPT and 464k UltraChat examples and target-written responses. It does not fully specify the source snapshots, split or regeneration parameters. These are newly registered preparation choices, not a claim to have recovered the original checkpoint's exact training set. See [the paper, section 4](https://arxiv.org/html/2503.01840v3).

Splits are deterministic 98/1/1 by transitive original-conversation and first-user-prompt groups. Complete normalized user-prompt sequences are deduplicated across sources. Exact normalized user turns overlapping the frozen benchmark panel are excluded; this is not semantic decontamination. Rejection logs retain all 17,935 exclusions. No answer is fabricated to fill a prompt-only row. The complete original-response files and target-generation inputs share the same grouped splits.

An earlier normalization receipt from job `2214937` is retained as version 1. It rejected prompt-only UltraChat generation rows and is superseded by version 2. It must not be used as the full corpus.

## GPU response generation

The frozen teacher bundle is `outputs/preparation/teacher-20261110-v3`. Coordinator job `2214977` attaches a full eight-GPU step to the current verified deep10 holder. It performs checksum and shard preparation on that node, runs a 64-conversation engineering smoke across all eight devices, then starts the full corpus only after that smoke succeeds. Both original source bytes and the running baseline deployment remain unchanged. The corrected smoke completed on all eight workers: 55 accepted conversations and 9 exclusions at the response limit. Full generation then started automatically.

The target is the original Llama-3.1-8B-Instruct verifier. Eight independent stock vLLM 0.8.5.post1 workers regenerate every assistant turn using newly generated history, with the pinned native system prompt, FP16, temperature 0, context 4096 and at most 2048 new tokens per turn. Complete EOS/eot-terminated conversations are accepted. Overlong inputs and censored responses are excluded with explicit reasons. The unchanged native 2048-token training filter is applied afterward. The original-answer regime retains its original text.

Every 64-conversation batch is flushed and synced with its input identity. Worker journals can resume committed batches; a partial trailing write is archived rather than treated as a completed answer. CPU completion job `2214983` runs after the coordinator. It verifies a completed corpus, or resumes the frozen plan after a Slurm allocation timeout or scheduler preemption. Explicit cancellation and computation failures stay blocked. Continuation source version 2 also handles requeue of the completion coordinator itself. **Actual timeout recovery was observed on 11 November:** holder 2212576 expired, the protected successor 2212580 started, and continuation 2214983 resumed the identical plan. By 14:49 Beirut, every worker had committed new examples; see the [recovery receipt](../../results/data-2026-11-11/README.md). The next protected holder is 2215920. A changed model, generation setting or source requires a fresh plan. Merged output is accepted only after all eight worker receipts and complete ordinal coverage verify.

Run this from a CPU Slurm task using a fresh run ID and the frozen source bundle:

```bash
export ROOT=/mnt/nfs/home/mohamadzbib/projects/taps-rebuild
export BUNDLE="$ROOT/outputs/preparation/teacher-20261110-v3"
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$BUNDLE/src"
/usr/bin/python3 -B -S -m taps_rebuild.data_queue launch \
  --root "$ROOT" \
  --config "$BUNDLE/configs/corpus-sharegpt-ultrachat-20261110.json" \
  --corpus "$ROOT/data/corpora/sharegpt-ultrachat-20261110-v2" \
  --run-id sharegpt-ultrachat-teacher-my-fresh-run
```

Models remain under the project's `models/` storage. Text, split manifests and generated conversations remain under `data/`; receipts and logs remain under `outputs/`. Native training computes target features online, so this preparation does not export a multi-terabyte prototype feature cache. Nothing is uploaded to Hugging Face by this stage.

The version-2 generation smoke stopped before model loading because its runtime assertion required `0.8.5` instead of the verified `0.8.5.post1`. Its failed receipts and automatically cancelled training dependencies are preserved. Version 3 corrects that assertion without changing the generation settings or dataset.

## Guarded training queue

CPU admission job `2214978` depends on successful data completion job `2214983` and validation/replication job `2214872`. High-priority training profile job `2214979` depends on successful admission. Dependencies reserve no GPUs. Failed admission cancels the dependent GPU job; an operationally complete validation job alone cannot authorize training.

The active training source bundle is `outputs/preparation/training-queue-20261110-v6`. CPU job `2214981` completed its separate raw-artifact recheck: conditional checks passed and greedy checks retained their failures. A partial audit cannot admit training. Full CPU recheck `2214995` completed all three phases: conditional and sampling passed; greedy and independent-review checks remained false. It caches the fixed vocabulary size once per method and replaces `2214987` without changing raw inputs, statistics or admission conditions. Copied receipts and the live GPU snapshot are in [the execution evidence](../../results/data-2026-11-10/README.md).

The [queue configuration](../../configs/training-queue-20261110.json) requests scratch initialization, all eight A100s and 200 successful optimizer updates with the unchanged native optimizer and schedule. This is a runtime profile using real data. Forty-epoch reproduction and the specialist grid are not queued, and mid-epoch restart has not been validated. Profile timings are needed to choose feasible subsequent training budgets.

The source-bound admission audit recomputes the registered checks from actual raw worker logs, conditional tensors and full histograms. It verifies complete coverage, exact budgets, prompt/seed identities, raw hashes, event definitions, statistical bounds and the original all-method reproduction gate. The native launcher checks that the admitted artifacts have not changed since the CPU audit. Claimed passed metadata is insufficient.

Training remains blocked. The full FP32 greedy panel has 3 mismatches among 1,120 pairs, and the original strict 22-cell gate did not pass. The 60-case conditional audit completed with zero failed registered checks; its 7,864,320 draws include both posterior and target-reference samplers. The separate end-to-end screen completed all 61,440 continuations with zero failed registered checks. The controlling [temperature-one proposal](../verification/TEMPERATURE_ONE_AUDIT_PLAN.md) and [workspace instructions](../../AGENTS.md) also require independent review of the validator and its admission interpretation. No approval is supplied or inferred by these scripts.

The protected deep10 holder and successor are preserved. At submission, the two running high-priority nodes account for 16 A100 80GB GPUs. The pending profile can start only after the current deep11 job has ended, preserving the 18-GPU allowance. Deep-h-1 failed its CUDA health check and is not used for computation.

## Scheduled independent follow-up

Job `2215000` follows the current node11 job with the complete FP32 greedy control and then the EAGLE-2/HASS reference replication. Its [registration](REPLICATION_FOLLOWUP.md) preserves all original cases and keeps new outputs separate from the failed original baseline. The successor uses the same eight-GPU node and keeps total running high-priority use at 16 GPUs when eligible.
