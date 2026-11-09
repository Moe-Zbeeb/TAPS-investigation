# Implementation and evidence boundaries

The rebuild memo defines a research programme, not a single runnable benchmark. The source implements preparation, official reference execution, data processing, a shared-expert research prototype, pooled candidate generation/verification, weight composition, native EAGLE-3 integration and stock serving tools. Execution receipts determine which paths are verified.

The eight-GPU engineering checks passed on 9 November 2026, including native two-update training with explicit model/optimizer checkpoint recovery and real stock serving requests; see `experiments/engineering-20261109/RESULTS.md`. Published baseline reproduction, generalist/specialist training and scientific result sweeps are separate stages. Failed attempts remain in the raw outputs.

## Remaining before the complete paper

- Complete the registered 22 published acceptance cells and full-model temperature-one distribution audit. Finite-vocabulary posterior checks are supplementary.
- Reproduce training on registered ShareGPT/UltraChat data and compare the resulting drafter against the original checkpoint.
- Register actual sources and held-out splits for five domains, two model families and sizes, a MoE target, target-written/original responses, capacity/data/mixture grids and three seeds.
- Enforce and audit matched training compute as well as token/optimizer budgets. The prototype's token counters do not establish matched FLOPs or GPU time.
- Export real node-level verification labels for calibration, then study distinct trained experts, K=1–7, budgets, deduplication, cost-aware expansion and concurrent drafting.
- Connect shared-body training checkpoints to native candidate generation and compare a shared-generalist-to-specialist chain, interpolation and composition against multi-task training.
- Implement classifier/no-regret/bandit/online selection, full-tree merge, retrieval and any eligible recent public baseline from their actual primary sources. They are not interchangeable names for the current pooled adapter.
- Integrate custom pooled/shared-expert methods into a serving engine, with scheduler, KV state, kernels and batched correctness tests. Stock engine command support alone does not do this.
- Register held-out LMSYS/WildChat and shifting traffic streams, measure server component costs and memory, and validate hardware beyond the tested A100s when available.
- Fit mixture predictions and workload-KL selection diagnostics on development data and test held-out traffic. Add final preregistered paired contrasts and independent table regeneration before release.

These items depend on the reproduction and pilot outcomes. No historical number is a new measurement, no paper correction is made automatically, and no artifact is uploaded to Hugging Face by these tests.
