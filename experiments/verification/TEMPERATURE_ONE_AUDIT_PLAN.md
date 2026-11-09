# Temperature-one audit proposal

Status: proposed on 2026-11-09; no validator has been implemented or run. This document changes no admission gate. The existing native-training gate remains closed even if a hand-written receipt claims success. No scientific result is supplied here.

The rebuild plan, section 5.5, requires both a proof of the verification rule and empirical token-distribution checks. A finite experiment cannot prove losslessness for every prompt, history, tree, hardware path and random stream. The proposed evidence is a real-model conditional sampler audit plus a bounded end-to-end distribution screen, each with a stated population and tolerance. Neither should be relabelled a universal losslessness proof.

## Source findings and the first-token pitfall

The pinned EAGLE implementation samples the initial root directly from target logits in `initialize_tree`, before posterior verification. Consequently, comparing only the first generated token mostly tests target sampling and can miss a completely broken speculative posterior. The posterior then traverses fixed candidate children, ignores duplicate children and the -1 padding token, removes a rejected child's mass, and renormalizes before trying another child. `update_inference_inputs` commits the accepted path, copies the selected KV entries and samples the next root from the returned residual weights. Those later stages need distinct coverage. See the pinned [EAGLE implementation](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/utils.py#L210-L431) and [HASS implementation](https://github.com/HArmonizedSS/HASS/blob/051c4020141f5d2a365994ca9a8148700044c6d7/model/utils.py#L292-L417).

Current `verification.py` loads the actual posterior functions but uses a four-token artificial distribution on CPU. Its receipt correctly states that no models were loaded and that full-model validation is still required. `OfficialBackend` loads the actual target and drafter in FP16 and runs the complete native generator, but its current trace reports counts and timings rather than retaining posterior inputs or independently recomputed AR distributions. A new observational collector is therefore needed; the existing inference or verification sources must not be silently replaced.

## Freeze before sampling

Freeze the original three method definitions, model revisions and complete weight-file identities, tokenizer and prompt bytes, upstream and authored hashes, runtime packages, dtype, CUDA/driver/GPU inventory and exact sampling processors. Use temperature 1, no top-k/top-p truncation, no repetition penalty, and the official 60-node tree configuration. Report actual node counts rather than assuming a configuration field has identical off-by-one semantics across implementations.

Select two first-turn questions from each of the five pinned benchmark files by ascending SHA256 of `[20261109, benchmark, question_id]`: ten prompts total. Freeze their exact rendered input token IDs before the audit. This small panel is a deliberate engineering coverage panel, not a representative estimate over all user traffic. All three methods receive the same selected question texts using their own pinned official prompt/tokenizer definitions. Model or source differences between methods remain explicit.

Use new independent PRNG streams for sampling after the contexts and trees are fixed. Do not pick questions by observed entropy, acceptance, p-values or a successful preliminary run. A timing-only 1,000-call calibration is allowed before the fixed sample budget; its outcomes are excluded from analysis, and it cannot change the questions, event definitions, margins or sample counts. If capacity is inadequate, record an incomplete audit rather than reduce the registered sample count after looking at outcomes.

## A. Real-model conditional posterior audit

For each of ten prompts and three methods, capture the initial verification cycle and the third verification cycle of a seeded native T=1 trace: 60 fixed cases. Each case consists of the exact committed prefix, its already selected root, the actual full drafted tree, native tree logits and the source-bound posterior callable. For a missing third cycle because a terminator was generated, retain a coverage failure; do not substitute a favourable trace. An implementation may instead preregister a separate fixed nonterminating continuation fixture before any audit draws, but that is a different registration.

Recompute the target-only next-token distribution independently at the identical prefix including the root, with tree masking disabled and clean AR cache state. This must not reuse the tree logits as its purported AR reference. Define the reference categorical law `q` from the actual native AR sampling weights, converted to FP64 and normalized; retain both original weights and their normalization. Report the complete-vocabulary TV difference between native tree and independently recomputed AR weights as a numerical diagnostic. This exposes target-forward/cache differences independently of sampling noise.

For each fixed case, perform exactly 65,536 independent repetitions of the unchanged native posterior and its actual CUDA residual sampler. The scored token is the first token after the fixed root: `candidates[best_candidate, 1]` when at least one draft token is accepted, otherwise the token drawn from the returned residual weights. It is not the original root. Preserve the full native candidate paths for the call, including normal duplicate/padding handling, even though only the first posterior-driven token is scored. The additional accepted positions are diagnostics, not independent samples.

Also draw 65,536 target-only categorical samples from that independently computed `q`, using the actual target categorical sampler. These are draws from cached real-model weights, not autoregressive completions. This separately catches a sampling/normalization discrepancy. Neither branch requires another full target forward per draw.

Before any audit draws, define eight binary token events from the frozen case:

1. The highest-probability AR target token, breaking equal-probability ties by token ID.
2. Membership among the top eight AR target tokens.
3. Membership among the top 32 AR target tokens.
4. Membership among the top 128 AR target tokens.
5. Membership in the tokenizer's exact EOS/eot terminal-token set.
6. Membership in the unique valid first-level proposed-child set.
7. Membership in a fixed vocabulary hash partition: the low bit of SHA256 of `[20261109, 0, token_id]`.
8. The corresponding independently fixed hash partition with salt 1.

Events may overlap; they need not be independent. Deriving top-token sets from the frozen exact reference distribution is allowed; selecting them from observed sampler outcomes is not. Save the event membership definitions and exact `q(event)` values with the capture hashes.

Allocate familywise alpha 0.025 to all 60 cases × eight events × two samplers: 960 tests. For each event use the fixed-sample Hoeffding interval, with simultaneous radius

`r = sqrt(log(2 * 960 / 0.025) / (2 * 65536)) = 0.009264...`.

Pass the declared conditional **event-probability** check only when every `abs(empirical_probability - q(event)) + r <= 0.02`. Otherwise report failed or inconclusive as appropriate; a non-significant goodness-of-fit p-value is insufficient. Under exact equality of every tested law, `2r < 0.02`, so the same concentration bound gives at least 97.5% joint probability of satisfying all these conservative event criteria. Conversely, falsely passing an event whose true error exceeds its margin is controlled by the simultaneous confidence construction. Counts, bounds, all failed cases and exact completed sample counts must be reported.

This is 3,932,160 actual native posterior calls and the same number of cached target categorical draws. It tests real full-vocabulary model outputs at selected conditional states. Eight event checks do **not** upper-bound full-vocabulary TV: an error can redistribute mass within the event intersections without changing any tested event. Report a full token histogram and exploratory discrepancies, but do not promote those diagnostics to an unregistered full-TV confidence claim.

## B. End-to-end native-generator distribution screen

Run the complete unchanged AR and speculative generators, including fresh initialization, cache reset/commit, subsequent trees and normal termination. For each method, each of the ten fixed prompts and each mode, collect exactly 1,024 independent continuations with a comparison cap of 32 tokens: 61,440 continuations overall, not millions. Salt PRNG seeds by method, mode, question and repetition, and use disjoint AR/spec seed streams. The existing reproduction runner intentionally pairs seeds across modes; reusing that coupling would require a different paired concentration bound and cannot silently use the independent-arm radius below. Preserve raw emitted tokens, exact comparison prefixes through the first EOS/eot inclusive then capped, overshoot, seeds, cache/reset identities, timing and source/model provenance.

The population for each method is the equal-weight mixture of these ten fixed prompts. Fixed balanced sampling permits Hoeffding bounds for independent bounded observations even when prompt-specific probabilities differ. Do not treat tokens within one continuation as independent observations. The sample unit is one complete continuation. This pooled population can conceal opposite prompt-specific errors; per-prompt descriptive results and section A remain necessary.

Predeclare 12 binary events on each comparison continuation:

- At positions 2, 8, 16 and 32, each of the two token hash partitions defined above: eight events. For a position after termination use a distinct absorbing END symbol with its own fixed hash; retain the terminating token at its actual position.
- Whether a terminal token occurred by positions 8, 16 and 32: three events.
- Whether any contiguous four-token sequence repeats within the first 32 comparison tokens: one event.

There are three methods × 12 events = 36 two-sample differences. Allocate the other familywise alpha 0.025. Each arm contributes `N = 10 * 1024 = 10240` independent continuations per method. The simultaneous bound for each AR/spec event-probability difference has radius

`r = sqrt(log(2 * 36 / 0.025) / 10240) = 0.027891...`.

Pass this intentionally coarse end-to-end screen only if every `abs(p_spec - p_AR) + r <= 0.06`. Under exact equality, `2r < 0.06`, giving at least 97.5% joint passing probability under the stated assumptions. Together the two confidence families control false equivalence conclusions for their declared event sets at at most 5%. This six-percentage-point screen detects large pipeline bias; it does not establish a two-percent full output-distribution tolerance. T=1 same-seed token equality is neither required nor sufficient.

Keep the existing T=0 exact comparison check over the complete reference panel. For this T=1 screen, also report invalid token IDs, missing/extra samples, nonfinite probabilities, residual mass failures, unexpected cache reuse, termination/cap violations and method-specific failures. Missing records, capture failures or sample-budget shortfalls cannot yield a pass.

## Cost and what the proposal saves

The conditional stage needs at most 60 real tree captures plus independent AR reference forwards, then cached posterior/categorical work. It saves millions of repeated target-model forward passes. It still contains nearly four million Python posterior calls with CUDA synchronization and full-vocabulary softmax work; runtime must be measured, not inferred from the four-token CPU test. At 0.1, 1 or 10 milliseconds per call, the posterior portion alone costs approximately 0.11, 1.09 or 10.92 GPU-hours, respectively. Eight independent workers can distribute captures/cases; loading and different method/runtime waves add overhead. CPU replay of copied logits is useful debugging but cannot stand in for actual CUDA sampling evidence.

The end-to-end screen has at most 1,966,080 comparison tokens, plus raw block overshoot, prefill and any necessary native final draft. The completed local [T=0 smoke report](../../results/engineering-2026-11-09/baseline-smoke-report.json), SHA256 `aea6b4e424ba73f960a080806309de119f131daf1a56ea18a7630005d40fee58`, measured approximately 32–39 AR and 77–91 speculative comparison tokens/second on A100s; these were short smoke observations, not measured T=1 rates. Using 35 AR and 80 speculative tokens/second only as a planning illustration gives approximately 7.8 AR plus 3.4 speculative GPU-hours before extra overhead, or about 1.4 ideal hours over eight GPUs. A practical reservation is two to four full-node hours, revised from an excluded timing-only calibration. This is a planning range, not an execution promise. EOS can reduce work; raw overshoot and repeated prefill can increase it. Report actual GPU-hours separately from allocated idle time.

A stronger five-point end-to-end event margin can use 16 fixed prompts × 1,024 repeats per arm, giving 98,304 continuations and up to 3,145,728 comparison tokens. That is an alternative registration, not permission to switch designs after inspecting outcomes. Strong conditional full-vocabulary identity testing is also possible with a dedicated statistically validated tester; it does not inherently require full AR completions because `q` is known from logits. Worst-case discrete identity testing has a vocabulary-dependent sample cost, and the high-confidence results are not the same task as estimating TV with the current four-token plug-in bound. See the primary [identity-testing result](https://arxiv.org/abs/1708.02728).

## Proof and admission boundary

Write the appendix proof separately for the actual fixed-child rule, including repeated tokens, invalid padding, exhausted residual support, arbitrary draft-side tree selection and every point at which randomness or target information can influence ordering. The usual draft-probability rejection rule has an established exact-distribution argument, but that does not automatically prove this different deterministic-candidate rule or an implementation with altered cache state. The original [speculative-decoding paper](https://proceedings.mlr.press/v202/leviathan23a.html) provides the relevant distinction between exact sampling arguments and speed measurements.

The proposed empirical pass means only: selected real-model conditional event probabilities and a specified short-continuation mixture passed predeclared margins with complete source-bound coverage. It does not establish correctness for every prefix, long sequence, merged/routed/pooled variant, batch size or serving backend. A new drafter/tree rule, target, dtype, runtime/kernel path or serving backend requires its own scoped evidence. The current native research gate must remain closed until the validator, proof, registration and precise admission interpretation are independently reviewed and explicitly integrated. No existing gate, scientific scope or claim is relaxed by this document.
