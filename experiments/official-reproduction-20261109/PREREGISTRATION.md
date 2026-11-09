# Official reproduction protocol

This specification is frozen before collecting new GPU results. The controlling configuration is `configs/reproduction.json`. Reference numbers identify published comparison cells; they are not measurements from this project.

## Reference panel and generation

Use the exact five 80-question files pinned in `configs/sources.json`. MT-Bench has two turns per question. EAGLE-3 uses all five panels; HASS and EAGLE-2 use MT-Bench, HumanEval and GSM8K. Preserve source question order and all turns. The system prompt is extracted from the pinned official evaluators; identical warmup and measured copies must agree.

Use one FP16 target per GPU, batch size one, temperatures zero and one, a 60-node verification budget including its root, expansion width ten, and API depth seven for EAGLE-3 or five for HASS/EAGLE-2. Record actual node count and observed depth. The reference wrappers explicitly use the official effective output limit of 512 and cache-length argument of 2048; the upstream evaluator's advertised 1024-token CLI default is not the effective generator call. Record overshoot, EOS and context limits. Do not silently truncate or discard a prompt that exceeds cache headroom.

Each question has five measured repetitions. Exclude warmup. Reset Python, NumPy and Torch random seeds for each paired generation. Target-only and speculative methods receive the same tokenized prefix; multi-turn history follows the speculative answer, matching its official evaluation. Alternate paired execution order deterministically. Log the input identity and each output so this conditioning is auditable.

## Measurements and reduction

Record every verification cycle and every raw appended token, including a verified block's overshoot. Count cycles explicitly and cross-check against the upstream last-iteration index plus one. Never divide by the unadjusted zero-based index.

For canonical acceptance, pool raw emitted tokens and true verification cycles across turns within each question and repetition, compute their ratio, average repetitions within a question, then average questions. Also report pooled acceptance and the equal-turn diagnostic with explicit labels. Prefill is recorded separately from verification cycles. Bootstrap whole questions, retaining their turns and repetitions, for 95% confidence intervals.

Comparison outputs retain the first EOS/eot token and are capped at the requested output limit. These comparison tokens define useful throughput; raw-token throughput is a diagnostic. Target-only greedy correctness compares the actual comparison-token sequences, including termination. GPU-synchronized generation wall time includes prefill, verification, drafting and the final unused draft; loading, tokenization, detokenization and disk logging are outside this measurement. Report those boundaries. CUDA-event components must retain the unused final draft as a separate cost. The official target-only wrapper retains its unused drafter, so its memory allocation is not a clean target-only memory baseline.

Every speedup uses measured time. An elapsed-wall-time ratio is eligible only for equal-output temperature-zero pairs. Temperature-one comparisons use measured normalized token rates, with no assumption that sampled output lengths match.

## Passing and stopping

A reference cell requires all 80 questions, all turns, all five repetitions, and all target-only/speculative pairs. Its canonical acceptance must lie within 5% relative error of the preregistered published value. Missing cases, inconsistent counters, malformed records, smoke samples and unexplained greedy mismatches prevent passing. EAGLE-2's primary reference is the matched HASS-paper table; its different original-paper table is not substituted after results are observed.

A passing numeric reproduction report does not establish losslessness. The finite-vocabulary audit calls the actual official posterior function and checks fixed candidate trees, duplicate children and residual sampling with simultaneous statistical bounds. It is a conditional unit audit, not a full-model output-distribution test. A separate large-sample temperature-one model check remains required before the research pipeline is declared validated or downstream training is started.

No specialist training, pooled-tree experiment or automatic publication is authorized by report generation. Scientific stage changes depend on the measured gate and the additional checks required by the rebuild plan.
