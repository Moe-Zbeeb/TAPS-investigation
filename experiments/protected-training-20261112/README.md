# Protected-node training calibration

The original teacher run completed on 12 November at 06:41 UTC. It processed all 540,107 inputs and retained 477,805 train, 5,009 dev and 5,009 test conversations. Full native QA passed; the unchanged upstream length filter retains 256,496 train, 2,659 dev and 2,627 test examples. The train-only native vocabulary has 32,000 entries.

The previously reviewed two-node generation migration was not deployed: there was no unfinished teacher work after SSH access returned. Its source and approval remain available for future workloads; they are not executed-migration evidence.

The original primary profile, job 2216033, began on deep11 while live state was being checked. A proposed move to the idle protected allocation refused the changed state before creating files or mutating jobs. The original profile and its automatic full-training chain remain intact.

## Parallel calibration

Use the idle protected deep10 allocation for a separate exact-recipe, 200-update real-data profile. This measures training throughput and sampled memory on both A100 80 GB nodes and checks the protected-holder launch path with real data. It is the same seed and recipe, not another scientific seed or a trained-quality result. It does not launch another forty-epoch run.

| Work | Job | Dependency |
|---|---|---|
| Separate profile preparation | 2216594 | CPU admission and the completed original corpus |
| Bind original audited vocabulary | 2216595 | Successful preparation |
| Attach all eight protected GPUs | 2216596 | Successful binding |

`profile.py` verifies every file in the frozen bundle, profile admission, exact native-plan content and the live protected-holder authority before requesting an exclusive eight-GPU workload step. It requires eight hours remaining on the allocation and bounds the step to 7 hours 50 minutes. It does not signal or release either holder job.

The [independent source review](../verification/reviews/protected-profile-approval-20261112.json) found no blocking wrapper issue. That review does not establish successful queued execution or replace the original scientific training admission.

The calibration bundle is `outputs/preparation/protected-calibration-20261112-v1`. It was copied from and checked against `pipeline-training-20261111-v1`, with fresh run/bundle paths and unchanged approved scientific helpers. The run is `sharegpt-ultrachat-protected-calibration-20261112-v1`. CPU preparation and binding precede GPU attachment; pending calibration does not imply active GPU computation.

The primary path remains job 2216033 → full preparation 2216034 → full training 2216035 → conditional continuation 2216036. Total high-priority allocation is sixteen GPUs, including the protected eight, within the eighteen-GPU allowance. All computation and dependencies run remotely through Slurm.

CPU comparison job **2216602** depends on both profiles succeeding. Its frozen `compare_profiles.py` verifies their native plans, eight-rank 200-update completions, physical mapping and token counters before writing `outputs/training-calibration/20261112-v1/comparison.json`. It submits no training jobs.

## Measurement interpretation

Record startup separately when interpreting cost. The existing profile wall time includes source/model hashing, native preprocessing, model loading and actual updates. Its linear full-recipe projection is deliberately rough; do not treat multiplied startup as a precise completion estimate. Compare token counts, elapsed training phases, per-GPU telemetry and physical identities before choosing later placement or experiment budgets.

The comparison isolates the recorded native-launcher interval from the rest of each profile. That interval still includes model initialization and dataset work. Its alternative full-recipe scenario includes one outside-launcher startup cost and repeated native-launcher cost; it is not a steady-state training-time estimate.

The first primary startup exposes an important scaling cost: the unchanged upstream program calls `build_dataset_rank` in every distributed rank after the launcher has already performed its separate native audit. The eight interleaved map progress streams are preprocessing, not eight optimizer progress streams. Later runs should qualify a source-bound prepared-dataset loader before a large pilot: compare complete ordered token/mask rows against the pinned upstream function, preserve formatting, collation, shuffling and RNG behavior, then repeat loss/update, restart and export checks and obtain a new exact-source review. That optimization is not implemented or admitted here, and active training bundles remain unchanged.

The GPU telemetry includes a known stale driver context on deep11. Whole-device memory samples are not exact framework peak-memory measurements. A Slurm RUNNING profile can still be performing CPU preprocessing; report actual utilization and optimizer progress separately from allocated GPU counts.
