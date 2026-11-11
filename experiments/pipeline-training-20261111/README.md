# Admitted native generalist pipeline

The user authorized moving the PDF project forward without matching published paper numbers. The [v2 policy](../verification/PIPELINE_VALIDATION_V2.md), [independent evidence approval](../verification/reviews/evidence-approval-20261111-v3.md) and [production replay](../../results/readiness-2026-11-11/pipeline-v3-replay.json) preserve historical failures and require source-bound correctness evidence. Production gate job 2216030 passed on the unchanged reviewed certificate and review artifact.

The source-matched eight-rank restart and stock export/inference qualifications passed as 2216015 and 2216016. These qualify execution on a synthetic two-update fixture. Real-data training follows the separate corpus, native vocabulary and profile gates below.

| Stage | Job | Resources | Dependency |
|---|---|---|---|
| Target-response completion | 2214983 | Eight workers on protected deep10 allocation | Existing generation plan |
| Full data/native vocabulary QA | 2215985 | 48 CPU cores | Successful generation |
| Admitted profile preparation | 2216031 | 16 CPU cores | Successful generation and production admission |
| Bind audited vocabulary | 2216032 | 16 CPU cores | Successful QA and profile preparation |
| Scratch-native profile | 2216033 | Eight A100 80GB GPUs | Successful vocabulary binding; 200 successful updates |
| Full generalist preparation | 2216034 | 16 CPU cores | Successful profile, with source-matched restart/export and data evidence |
| Full native generalist | 2216035 | Eight A100 80GB GPUs | Successful full preparation; forty native epochs |
| Conditional continuation | 2216036 | CPU coordinator | After the full allocation terminates |

The [submission record](../../results/readiness-2026-11-11/training-submission.json) contains the actual commands and verified Slurm records. Pending requests do not reserve GPUs. High-priority work uses account `grad-students` and QoS `spec-priority`; running high-priority GPU use remains within 18. The protected deep10 holder and its holding successor are unchanged.

The active deployed bundle is `outputs/preparation/pipeline-training-20261111-v1` under the cluster project. Its full source/configuration hashes are in [bundle.json](frozen/bundle.json). The small generated controllers, schedule and configurations are archived in [frozen/](frozen/) with original hashes. They are records of this deployment; fresh experiments require new run IDs and fresh frozen bundles. Do not resubmit an active single-use plan or edit deployed sources. Keep the native runtime separate from PRIME and upstream lockfiles.

The profile wrapper records sampled whole-device utilization/memory/power and actual native token counters, wall time and GPU-hours. It emits `profile-cost.json` with an explicitly rough full-recipe projection. Telemetry failures are recorded without stopping healthy training. Device memory samples can include the separately recorded stale driver context; they are not exact framework peak memory. Acceptance length is never converted to speedup.

Full training checkpoints every 1,000 successful updates, retains two committed process checkpoints, and restores model/optimizer/scaler/scheduler/RNG/data position using the qualified helper. The CPU continuation resumes only an owned timeout, preemption or node failure with a committed checkpoint and the required physical mapping. Ordinary failures stop. Success ends with `training_complete_evaluation_required`; held-out trained-drafter validation, quality/speed comparison and the broader PDF experiments still follow.

All computation and dependency transitions run through remote Slurm. Closing the local app or SSH connection does not stop the admitted queue, generation or protected holder.
