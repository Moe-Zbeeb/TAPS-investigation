# Source seed-policy diagnostic

This is a diagnostic for rebuild-plan §5.1, not a new reproduction gate. The original failed reports and the primary question-mean acceptance statistic remain unchanged.

The pinned EAGLE generator `gen_ea_answer_llama3chat.py` has SHA256 `6daaf6f8010ff986eadfd5a5da7546b7ea9107c4ecdcfed57d694335dc78aef3`. It resets Torch to the completion-choice index for every question (lines 214–218); its default choice count is one (lines 375–378). Thus the default single-choice run uses seed zero per question. Our frozen runner, SHA256 `7cc2327f83d76238a28e7539f094e1d8c387c0496c0c8ad13f6c079f84794d3c`, hashes the base seed, benchmark, question, turn and repeat and reseeds before each generation. This is a verified protocol difference, not an established cause of a published-value discrepancy.

Use all 80 registered first-turn questions from GSM8K, the largest remaining acceptance outlier, and all 80 Alpaca first-turn questions as a control. Read the exact tokenized inputs from the frozen validation preparation, SHA256 `d3f236a73ab18940bd28f76112210449a2f6ef797061d50a1ab98fc059e0a865`. First-turn restriction avoids changing or freezing a different sampled conversation history.

Run two arms with the unchanged released EAGLE-3 target/drafter, native FP16 generator, 60-node tree, depth 7, width 10, temperature 1, requested cap 512 and cache argument 2048:

1. Reset seed zero for each question/generation, matching the upstream default first-turn RNG initialization.
2. Reset the registered hashed seed using base seed 20261110, benchmark, question ID, turn zero and repeat, matching the completed independent replication's seed definition.

Run five measured repetitions per arm and question, alternating arm order by fixed question/repeat parity. Discard three full-length warmups per worker. The budget is **1,600 measured speculative generations plus 24 excluded warmups**. No target-only generation or speedup claim is part of this experiment. All raw tokens, comparison tokens, seeds, cycles and component/wall times are retained.

Use eight independent workers on a full owned A100 80GB node. Each gets 20 questions and 200 measured generations. Bind source, models, tokenized inputs, runtime, physical device mapping, registration and resource policy. The exact observed stale-context policy from the preceding numerical diagnostic is retained; every active/unregistered process blocks startup and every GPU requires at least 50GiB free. The affected GPU is disclosed, and this is not a clean timing comparison. Do not change deep10 generation or its holder/successor.

The primary statistic remains the mean of per-question acceptance: compute emitted tokens divided by explicit cycles for each question/repeat, average repeats within each question, then average 80 questions. Retain the published references 4.59 for GSM8K and 5.56 for Alpaca as diagnostic comparisons only. Report seed-zero minus hashed-seed acceptance using a paired question bootstrap with 5,000 resamples, fixed seed 20261111 and nearest-rank percentile 95% limits. Replaying seed zero five times does not create five independent questions or five independent quality samples; report how often its token sequences repeat exactly.

All 160 questions and both arms must finish with all five repetitions. Missing or duplicate records, changed inputs/seeds, incomplete cycles or changed raw-file hashes cannot yield a complete report. Preserve every outcome. An improvement under seed zero does not establish that it is generally better, that it caused the original outlier, that every upstream setting has been reproduced, or that the unrelated MT-Bench outlier is resolved. Do not switch the acceptance reduction after seeing results.

This test uses the existing native collector, SHA256 `d6ea2b5216b0052aa68282646b343143c6523ed14e75f8b2b838f99bb6458934`, and the frozen resource utility from the completed token-decision experiment. No verifier, training gate or statistical margin is changed. The registered seed difference is a source-fidelity diagnostic; broader multi-turn reproduction still needs its own policy and review.
