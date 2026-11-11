# Training readiness and automatic handoff

The controlling stage checklist remains `TAPS_rebuild_plan.pdf`. These changes prepare §5.1 generalist reproduction; they do not complete the 24-run RQ1 pilot or the later pooled/shared/serving studies.

| Order | Work | Pass requirement |
|---|---|---|
| Alongside corpus generation | [Full fixed-budget greedy control](../stopping-control-20261111/PREREGISTRATION.md) | Preserve all 1,120 pairs, predetermined stopping budgets, raw outputs and every remaining mismatch |
| Engineering qualification | [Actual process restart](PREREGISTRATION.md) | Eight ranks recover complete native state and match an uninterrupted control's final state, RNG and batch sequence |
| Engineering qualification | `export.py` | The resulting two-update engineering checkpoint loads into the unchanged official EAGLE-3 inference architecture and produces valid AR/SD outputs on eight frozen cases covering all five benchmarks |
| Existing scientific/data admission | Frozen training queue | Reviewed full-model validator and required official reproduction evidence, completed corpus, native tokenizer/length audit |
| CPU data readiness | [Exact native vocabulary preparation](DATA_READINESS.md) | Native train/dev/test filters and a train-only 32,000-token vocabulary; bind the cache and audit receipt before any GPU launch |
| Existing profile | 200 successful scratch-native updates | All eight ranks finish the registered real-data profile; this is not forty-epoch training |
| Full preparation | `python -m taps_rebuild.training_handoff prepare` | Matching restart/export source qualifications, complete admitted profile, unchanged full scientific gates and frozen real data |
| Full training | `python -m taps_rebuild.training_handoff launch` | Original forty-epoch scratch-native recipe with periodic process checkpoints; subsequent held-out evaluation is still required |

The handoff config is [`configs/training-handoff-20261111.json`](../../configs/training-handoff-20261111.json). Preparation runs in CPU Slurm with `CUDA_VISIBLE_DEVICES=''`. Launch and explicit `resume` run inside one verified eight-A100 allocation. No command submits jobs itself. Operators freeze this package/configuration separately and schedule known successors using Slurm dependencies; no public snapshot is deployed over active data generation.

Normal native launch remains single-use. Recovery-enabled runs may use explicit `native_training.py resume --plan ...` or the handoff's `resume` action after a recorded stopped/failed attempt. Checkpoints bind the same plan, source inputs and physical GPU mapping. The implementation keeps two committed process checkpoints and retains their historical manifests. Actual qualification results are separate from source tests and remain pending until the GPU reports exist.

The frozen `continue_training.py` runs in a dependent CPU Slurm job. It resubmits the same full training only after accounting verifies an owned timeout, preemption or node failure and a committed checkpoint exists. Ordinary training failures stop the chain. It preserves interrupted receipts, never touches deep10, prevents duplicate continuations, rechecks the total high-priority GPU allowance and retains the same physical GPU mapping. The initial full run and each recovery have a queued CPU successor; completion stops further allocation. A source change or failed scientific gate cannot be bypassed by continuation.

The original failed baseline report and scientific validator are preserved. Engineering success does not open scientific admission. The current published-acceptance outliers and independent review requirement remain explicit; no queue dependency or claimed passing metadata waives them.
