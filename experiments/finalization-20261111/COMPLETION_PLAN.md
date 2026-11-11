# Completion plan · 11 November 2026

This plan maps the [unchanged rebuild memo](../../TAPS_rebuild_plan.pdf) to evidence, dependencies and completion criteria. Its SHA256 is `357ba28e2e5e1aa658dae82c279253ac07457d9d406ac50cde0fbf513b150dae`. The live check was at 02:03 Beirut time on 11 November (00:03 UTC). It is a plan for completing the research programme, not a claim that the current infrastructure has completed it.

The 13:53 Beirut check found 307,392/540,107 generation inputs processed (56.9%), with eight active A100 80GB workers. The [remaining-token diagnostic](../../results/fp32-decision-2026-11-11/README.md) completed on eight additional GPUs as job 2215900: all eight captures were valid, the remaining disagreement was a near-tie argmax flip, and seven controls retained their top token. Follow-up [seed-policy job 2215909](../seed-policy-20261111/PREREGISTRATION.md) tests an observed upstream-versus-registered RNG difference. Neither experiment changes training admission. The [independent review packet](../verification/REVIEW_PACKET_20261111.md) pins the exact proof, collector, validator and statistical evidence. Earlier percentages and deadlines below retain their original snapshot times; allocation rollover remains to be observed.

## Current position

We are still in **§5.1, pipeline and reproduction**. No full generalist or specialist training result exists. The engineering trainer completed two updates on eight ranks and same-engine checkpoint recovery; that does not establish training quality or recovery after a process restart.

| Item | Checked state | Consequence |
|---|---|---|
| GPU allocation | Eight A100 80GB GPUs on deep-chungus-10; no second running GPU allocation | All eight run target-response generation; pending jobs reserve no GPUs |
| Target regeneration | 133,120 / 540,107 source inputs processed, 24.65%; no completion marker | These are processed inputs, not retained training examples; censored outputs and native token filtering still matter |
| Original released-baseline panel | 22,400 measured generations; 19/22 acceptance cells within 5%; strict gate failed | Preserve this failed report unchanged |
| New independent replications | Both completed; 20/22 acceptance cells within 5%; 9/22 strict cells passed | Diagnostic evidence only; neither the original gate nor training admission is replaced |
| FP32 greedy control | Same three failures in each of two 1,120-pair panels | Two context-stop length differences and one unresolved token disagreement |
| Temperature-one checks | Conditional and end-to-end screens passed their registered statistical checks | Bounded evidence; independent validator review and overall admission remain incomplete |
| Training queue | Data continuation → CPU admission → eight-GPU 200-update scratch profile | This is not a queued forty-epoch reproduction or the RQ1 pilot |

The read-only [diagnostic receipt](../../results/status-2026-11-11/diagnostic-audit.json) verifies completed worker/report hashes and extracts the new results. The [cluster receipt](../../results/status-2026-11-11/cluster-snapshot.json) records the live queue. Earlier snapshots remain historical records.

## 1. Close reproduction and correctness (§§5.1, 5.5)

Do this while target generation continues. Diagnose the smallest informative cases before spending GPUs on another full panel.

1. Register a common output budget from the prompt length, cache capacity and reserved verifier nodes **before decoding**. In the repeated CNN/DM cases, AR is an exact prefix of SD: 438 versus 440 tokens and 423 versus 424. Both hit the same effective context limit, with AR overshooting by one token and SD by three or two. Retain the original outputs and raw stopping metadata. A post-hoc cut to whichever output happened to be shorter is not an acceptable correction.
2. The MT-Bench question 146, turn 1 logit capture is complete: the same-prefix token-pair gap changes from +0.00004387 in AR to −0.000003815 in the tree, reversing the top token. Both replays reproduce the original decision, and seven controls retain their top tokens. Review this numerical characterization and perform clean-prefix/kernel/KV follow-ups if required; the exact kernel-level cause is not isolated.
3. Resolve the EAGLE-3 temperature-one acceptance outliers against pinned paper/code/checkpoint, prompt, stopping and reduction definitions: GSM8K is 5.514 versus 4.59 (+20.14%); MT-Bench is 4.565 versus 4.24 (+7.68%). The source audit found upstream seed zero per question differs from registered per-question/turn/repeat hash seeding; the first-turn GSM8K/Alpaca control now tests this difference. It does not resolve multi-turn RNG handling. The registered statistic is the **question mean**. Pooled and equal-turn reductions remain diagnostics; choosing a reduction after seeing which cells pass would change the experiment.
4. Version and review the corrected numerical/stopping protocol, then run its complete registered control and any necessary reference rerun. Report mismatch rate and classified causes as the memo requests. Do not silently require mathematical bitwise equality under every arithmetic implementation, or silently waive an unexplained failure.
5. Complete independent review of the full-model sampling validator, its power/coverage and source identities. Existing passing T=1 screens do not certify a future pooled backend, new target or serving runtime.

**Done when:** the required published acceptance comparisons meet the registered criterion or a documented source discrepancy has been independently resolved; the numerical policy and remaining mismatch causes are reviewed; a new source-bound admission report passes. The current guard pins the original failed baseline and will remain closed until an explicit versioned admission change is justified. Do not overwrite the failed report or its hashes.

## 2. Finish real data and reproduce training (§5.1)

The source corpus contains 529,031 train / 5,539 dev / 5,537 test conversations. Only 284,115 have complete original answers. Keep original-answer and regenerated comparisons on a registered common eligible population; report exclusions and domain/length distributions so missing answers and censoring cannot change the comparison unnoticed.

Finish all eight generation journals, then use CPU Slurm work for completion validation, deduplication, leakage checks, native tokenization, the 2,048-token training filter, vocabulary construction and an immutable corpus manifest. Generation currently permits a 4,096-token context and up to 2,048 new tokens; processed-input counts and generated-token counters are not retained training-token counts. Do not bulk-export target hidden features when native online generation is the intended training path.

Before a long run, demonstrate a real process restart restoring model, optimizer, scheduler, mixed-precision scaler where applicable, RNG, sampler and data cursor, including a mid-epoch checkpoint. Then run the queued 200-successful-update real-data profile, measure valid tokens/second, peak memory and GPU-hours, and estimate full-recipe cost. Train the registered full ShareGPT/UltraChat generalist and evaluate it against the released drafter under the same held-out settings.

**Done when:** the corpus passes QA, restart recovery is demonstrated, the full recipe completes with provenance, and the trained generalist has a reviewed held-out comparison against the official checkpoint. A profile or falling training loss is insufficient.

## 3. Run the decision-making RQ1 pilot (§§5.2, 6, 8)

Start with one target and chat/math/code. One generalist plus three specialists × target-regenerated/original responses × three seeds gives **24 core drafter trainings**. Train a matched seed cohort on the same physical hardware when GPU-seconds are the compute measure. Evaluate every drafter on every held-out domain and on a mixed stream; preserve paired prompts.

Register sources, common eligible examples, splits, response policy, training-token and compute budgets, evaluation traffic, primary contrasts and the oracle definition before launch. Match supervised tokens and measured compute, including target costs, rather than equating update counts. Keep target tokenizer/vocabulary and architecture contracts fixed within a comparison.

There are two useful compute questions: a specialist versus an equally trained generalist, and an entire specialist pool versus a generalist given the pool's total training budget. The proposed extra control is **six pool-budget generalists** (two response policies × three seeds), taking the pilot to 30 runs if adopted. This is a proposed registration choice, not a currently queued experiment or a replacement for the 24-run core.

Estimate the oracle-specialist advantage over the matched generalist with paired prompt intervals and seed variation. Define the memo's approximate 10% decision threshold as a relative improvement, with uncertainty and the primary workload fixed in advance. If the gap is small, complete the explanation, mixture and capacity study and report bounded composition controls. If it is substantial, prioritize the pooled tree and shared experts. A temperature-zero pooled prototype may progress alongside the admitted pilot, as §8 suggests, without being presented as validated scientific evidence.

**Done when:** all registered core runs and cross-domain evaluations are complete and a reviewed, uncertainty-aware branch decision exists.

## 4. Expand RQ1 without an uncontrolled Cartesian sweep (§5.2)

Cover chat, mathematics, code, long reasoning, and a fifth domain. Retrieval/summarization is the proposed fifth-domain default; multilingual remains the memo's alternative. Fix this choice in the experiment registration before implementation.

Cover Llama-3.1-8B, Llama-3.3-70B, Qwen3-8B, Qwen3-32B and a MoE such as Qwen3-30B-A3B; the dense Qwen3 targets require thinking-on/off conditions. Study one/two/four drafter layers, 10k/100k/1M examples per domain, and three-domain mixture weights. Fit mixture predictions and cheap workload-KL selection diagnostics on development data and test their predictions on untouched traffic.

These are substantial engineering extensions. The current native adapter is a fixed Llama 8B architecture, and the validated stock path is one-layer EAGLE-3. It does not already support Qwen, 70B targets, MoE or two/four-layer export and serving. The 2k training context is also inadequate as an unexamined long-reasoning policy. Profile tensor-parallel target inference, long-context memory, native losses, checkpoint/export compatibility and serving loads in separate pinned environments before committing a large grid.

As a planning illustration, five targets × six drafters × two response policies × three capacities × three data scales × three seeds is **1,620 trainings** before mixture sweeps. Treating the two dense Qwen thinking modes separately gives seven target conditions and 2,268 trainings. These are factorial cost illustrations, not a claim that the memo mandates every Cartesian cell. Register a core cross-domain grid and focused ablations covering each scientific factor; record any scope reduction explicitly.

SpecForge is a candidate for a separate scaling feasibility spike: its current documentation describes Qwen recipes, disaggregated target capture and resumable offline training. Its EAGLE-3 strategy remains one-layer, so it does not by itself resolve the capacity sweep. Require loss/gradient, checkpoint and exported-model parity before adoption; preserve the working runtime. [Official documentation](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md).

**Done when:** the registered target/domain coverage and explanatory capacity/data/mixture contrasts are complete, with measured resource cost and held-out predictive tests.

## 5. Complete pooled and shared-expert studies (§§5.3–5.5)

| Study | Required implementation and evidence | Completion criterion |
|---|---|---|
| Fixed-budget pool, RQ2 | Distinct trained experts; held-out node-level verification labels; cross-drafter calibration; identical token-path deduplication; prefix closure; deterministic ties; one consistently counted budget including the shared root | K=1–7 and budget sweeps, calibration/dedup/cost-aware/concurrent-drafting ablations, paired comparisons with the strong generalist and selectors |
| Verification proof | Fixed candidates before verification randomness; sequential residual rule; duplicate tokens, zero residuals, padding, sibling order and fresh RNG handled explicitly | Independently reviewed proof and native T=0/T=1 checks for this backend; exact-probability optimality is not claimed for estimated calibration scores |
| Weight composition, RQ3 | Generalist ancestor → fine-tuned specialists; aligned vocabulary/parameters; linear interpolation, averaging, task arithmetic, TIES and DARE | Compare with multi-task training, plot interpolation curves and measure actual decoding performance |
| Shared experts, RQ3 | Native shared body plus output heads/LoRA and a trained token gate; export and KV integration | Scientific quality and measured drafting cost versus independent experts; a common body is not treated as free work for every tree node |

Existing prototypes demonstrate engineering paths, not these trained comparisons. Keep full-tree merging at approximately twice the verification budget as an explicit cost control, separate from the fixed-budget pool.

## 6. Finish baselines and production serving (§§3, 5.6, 5.7)

Refresh the independent prior-work and availability audit before making novelty claims. Implement or integrate the actual classifier, no-regret, MetaSD/BanditSpec, online adaptation, SpecInfer-style full merge and retrieval baselines. Record supported targets, source revisions, weights, data and costs; do not relabel the current pooled adapter as these methods.

Osprey now has public code and downloadable checkpoint folders, so the memo's conditional baseline is eligible for evaluation. It also studies target-regenerated, cross-domain drafting, which must inform the novelty assessment. Availability is not local installation or successful cluster execution. [Official code](https://github.com/LeanModels/Osprey), [released files](https://huggingface.co/BobbieBieee/Osprey-Speculative-Decoding/tree/main).

Integrate the actual pooled/shared methods into a pinned SGLang or vLLM runtime with continuous batching, attention/KV correctness and scheduler instrumentation. Stock serving's 344 HTTP smoke requests do not establish this integration. Measure realized scheduler batch occupancy at 1/8/32/128, TPOT, per-request throughput, total system throughput, drafting/verification time and incremental drafter memory. Use a clean target-only process without an unused resident drafter for the AR memory baseline.

Evaluate held-out LMSYS-Chat-1M, WildChat, shifting controlled mixtures and Spec-Bench. Repeat timings at least five times with warmup excluded, pairing comparisons on the same hardware. A100 execution is verified; healthy H100 execution remains to be demonstrated after the failed health probe. B200 is conditional on access. Never derive wall-clock speedup from acceptance length. If a method helps only at batch one, report latency and its crossover rather than a general throughput claim.

**Done when:** every claimed baseline and custom method has comparable validated serving measurements across the registered workloads and hardware, or a clear access limitation is disclosed.

## 7. Close the paper and release (§§2, 5.8, 7, 8)

Build a factual original-paper discrepancy dossier from original logs/scripts and independently checked tables/figures. Distinguish the memo's allegations, verified arithmetic inconsistencies and facts about how values were produced. A correction or withdrawal proposal goes to the coauthors; this plan does not authorize sending it or changing arXiv.

Generate all new tables and plots from raw records, keep at least three training seeds for core drafters, use paired prompt bootstrap intervals/tests and development-only tuning, and report negative results and incomplete coverage. Have a person who did not run the experiments regenerate the main tables. Release code, checkpoint identities, permitted checkpoints/data manifests and logs with provenance. Write a new title, abstract and limitations around whichever §6 branch the evidence supports.

**Done when:** an independent reproduction of the reported tables succeeds, claims match measured results and the release is complete. A running cluster queue is not the completion criterion.

## Resource and scheduling plan

- Preserve deep10's live holder, its successor and the frozen legacy controller. The holder has about 12.6 hours remaining at the snapshot; corpus generation is likely to cross that deadline. Validate actual journal recovery at rollover. The continuation policy has been checked, but successful real rollover has not yet been observed.
- Current generation uses all eight GPUs. Its slowest worker suggests roughly 35 more hours at its average rate so far, **before** filtering and allowing for rollover or throughput changes. This is a rough extrapolation, not a completion commitment.
- The 18-GPU high-priority allowance permits two whole eight-GPU nodes, or 16 GPUs in practice, leaving two unused when workloads require full nodes. The second node is currently occupied by other users. Request concrete validated work when capacity is available; use suitable idle normal-priority whole nodes as authorized. Do not reserve GPUs for undefined training.
- Use Slurm CPU jobs for corpus QA and report generation. Keep source, configuration, artifact and hardware identities frozen for running work. Schedule known successors with dependencies, but preserve correctness/research admission between stages. The existing admission job remains blocked and the pending profile is not a substitute for full training.
- After the real profile, publish tokens/second, GPU-hours and memory by workload, then estimate the pilot and expansion. Run independent matched-seed cohorts across nodes, with all eight GPUs doing useful work on each node. Reuse only artifacts with compatible target/tokenizer/data identities.

The next decisive deliverables are a reviewed reproduction diagnosis, the completed corpus, restart-safe full training, and the 24-run RQ1 decision. There is no defensible end-to-end completion date yet: larger-target/multi-layer support and custom serving are still unvalidated, and the measured training profile has not run.
