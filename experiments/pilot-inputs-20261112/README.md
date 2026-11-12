# Math/code pilot input registration

This is input preparation and target-response generation for the PDF's chat/math/code RQ1 pilot. It is not the 24-run training registration, a trained-quality result, or another full forty-epoch generalist. The existing admitted generalist run remains unchanged.

## Sources fixed before generation

Use the [pinned configuration](../../configs/pilot-math-code-inputs-20261112.json):

| Source-defined domain | Upstream source | Immutable revision | Size policy |
|---|---|---|---|
| Math | [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) | `b4fdc323a7be1379c9c7c0b67b1de72dfee2111a` | At most 20,000 eligible connected groups |
| Code | [sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | `152bb5e9a29651266b018106053980070a0521a1` | At most 20,000 eligible connected groups |

MathInstruct's LFS SHA256 is pinned. CodeAlpaca's ordinary Git blob identity and size are pinned; preparation verifies the canonical Git blob hash, then records the exact SHA256 in a fresh derived configuration. Source-reported licenses are retained in the configuration. These labels describe source datasets, not a semantic classifier of every prompt.

Preserve the instruction and original answer bytes. A nonempty Alpaca input is appended to the instruction with exactly two newlines. Retain the original source row, file/revision, ID and available upstream subdataset label. Reject malformed pairs and embedded native template controls rather than fabricating answers.

## Selection and separation

Exclude exact normalized individual user-turn matches against the pinned reference-panel inputs and every original ShareGPT/UltraChat conversation split. The latter anchor is the immutable original canonical corpus, not a new sample or regenerated chat corpus. This prevents exact prompt overlap with chat data that can later be reused. It is not a semantic contamination guarantee or a claim to have excluded all examples from every benchmark.

Deduplicate normalized complete prompt sequences, form the existing transitive conversation/first-prompt groups, then select groups by SHA256 priority with seed 20261112. Apply a 20,000-group cap per source before response generation. Groups stay whole; a group cap is not necessarily a row cap. This bounded selector only accepts complete single-turn instruction pairs and rejects a family that spans source owners. Assign 98/1/1 train/dev/test splits by the registered grouped hash rule. Sampling does not use target answer quality, likelihood or acceptance.

All source rows remain available in their pinned raw files. Record schema exclusions, exact reference/chat overlap, deduplication and selection counts separately. The resulting selected population can be smaller than 40,000 and will shrink further under target EOS/context and native 2,048-token rules.

## Teacher and scheduling

Regenerate answers with the already prepared Llama-3.1-8B-Instruct target, pinned stock vLLM runtime, FP16, temperature zero, context 4,096, at most 2,048 new tokens and 64 conversations per worker. Bind each turn's seed to the example identity and seed 20261112. Use all eight protected deep10 GPUs, starting only after successful calibration and successful input preparation. The existing real-response smoke precedes full selected-population generation. Preserve incomplete/censored exclusions and per-worker journals. No bulk target hidden features are exported.

Freeze a separate pilot bundle. Do not redeploy the changed corpus adapter into the active root or either training bundle. The approved native training, validation and recovery helpers are unchanged. Preserve both protected holder jobs, and keep total high-priority GPU allocations within eighteen. The generator adds a step to the existing eight-GPU holder rather than requesting another allocation.

The [independent source review](../verification/reviews/pilot-input-teacher-approval-20261112.json) approves only input/teacher source. Actual pinned downloads, full teacher coverage, EOS outcomes and native QA remain execution checks.

## Before pilot training

Intersect original and target-generated eligibility by the same example IDs and exact native tokenization. Register shared vocabulary, domain-balanced populations, training tokens, measured compute, seeds, evaluation traffic, uncertainty and oracle contrasts before launching any matched-budget pilot training. The original source responses form the ablation; do not compare policies on different retained populations. Reuse chat responses only with their existing source/model/settings provenance. The existing chat generalist/profile does not count as one of the pilot's 24 matched runs.
