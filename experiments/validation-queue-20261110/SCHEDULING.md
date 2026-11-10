# Scheduling addendum · 10 November 2026

## Excluded startup checks and independent replication

Each node first runs an explicitly labeled engineering smoke with 16 conditional draws and two end-to-end continuations per prompt and arm. These startup outcomes are excluded from the registered full-budget analyses and cannot alter their inputs, events, sample budgets or margins. A smoke does not pass the statistical gate at these budgets; operational execution is its only purpose.

After the A100 end-to-end screen completes operationally, run a separately labeled full EAGLE-3 reference replication on the same 80GB node using the existing frozen reference runner. Preserve all five panels, both temperatures, five measured repeats, warmups, FP16 weights and generation settings. Change only the study identifier and base seed to 20261110. This independent run diagnoses seed/hardware sensitivity of the original EAGLE-3 outliers; it neither replaces the original report nor changes its registered reduction. The generated replication configuration has SHA256 `73ef7b80ce2288814d41f830e0c8af017136a49dff9f4dae9e6cf337fcdd35ff`.

The operational `chain.py` and this scheduling addendum are recorded separately from the already frozen validation-input registration. The scientific driver and sample/event definitions are unchanged. Batch dependency success refers to operational completion; it does not automatically admit research training.

## Submitted execution and recovery

CPU input preparation job 2214861 completed on deep-chungus-6. The original frozen driver SHA256 was `9abf0bf66749062c43c5fae7d4ca4023a3b10486fa90e868f7d16667273e6a80`. Its excluded startup smoke exposed an inference-tensor cache reset outside PyTorch inference mode. No full-budget phase was entered. The failed deep10 step and superseded second-node job 2214866 are retained. Only that second-node workload allocation was stopped; the protected deep10 holder and successor were unchanged.

Revision v2 moves the independent AR cache reset inside inference mode and stops operationally failed worker waves promptly. The scientific driver SHA256 is `d6ea2b5216b0052aa68282646b343143c6523ed14e75f8b2b838f99bb6458934`. The input migration changes only the driver SHA; selected prompts/prefixes, source/model identities, seeds, event definitions, budgets and margins are unchanged. Eight CPU Torch regression tests passed inside Slurm, including the actual inference-tensor reproduction. The lightweight local suite passed 333 tests with nine Torch-dependent skips.

| Execution | Resources | Automatic sequence |
|---|---|---|
| Protected holder 2212576, detached v2 step | deep-chungus-10, eight verified A100 80GB GPUs | Excluded startup checks → full-panel FP32 greedy control → full conditional T=1 audit → summaries |
| Batch job 2214872 | high-priority/spec-priority, one full eight-GPU 80GB-class A100 node; currently deep-chungus-11 | Excluded startup checks → full end-to-end T=1 screen → independent full EAGLE-3 reference replication → summaries |
| H100 health job 2214867 | h100/background, all six configured GPUs on deep-h-1 | Failed CUDA initialization before matrix execution; dependent job 2214868 was cancelled automatically |

Both A100 nodes passed full-node CUDA matrix checks and reported 80GB-class devices. The two allocations use 16 of the user's 18 high-priority GPUs. The remaining two do not fit another complete GPU node. No compatible complete normal-priority A100 node was idle at submission. Deep-h-1 could not initialize CUDA in its granted environment; no reset, driver change or cluster reconfiguration was attempted.

Run artifacts are under `outputs/validation-pipelines/primary-2212576-validation-queue-20261110-v2-*` and `outputs/validation-pipelines/screen-2214872-validation-queue-20261110-v2-*`. Frozen sources, source inventories, migrated input identities and exact launch commands remain under the corresponding `outputs/preparation/`, `outputs/validation-inputs/` and `outputs/launchers/` directories. These allocations and paths describe this execution, not a reusable allocation guarantee.

The controllers are detached from the local SSH client. Each successor starts from a verified operational completion; scientific failures remain visible and never launch training. The H100 dependency demonstrated the failure stop: its full workload did not start.
