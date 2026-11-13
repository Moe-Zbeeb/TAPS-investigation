# 13 November execution evidence

At the [14:57 UTC snapshot](live-progress-latest.json), all eight A100 80GB GPUs on deep11 were training the corrected generalist: 7,107 successful updates per rank and a committed 7,000-update checkpoint. The protected deep10 successor remained pending. No fixed-budget pilot has completed.

The fresh pilot passed its 208-update regression on all eight ranks, all 12 standard qualification phases, all 14 independent replay checks and separate actual evidence review. Its [resource handoff](pilot-reviewed-resource-handoff-submission.json) is running in Slurm and waits for the verified owned high-priority allocation before rehash, integration and training.

- [Stage status](../../experiments/STATUS_20261113.md) maps observed progress to the PDF requirements.
- [Native admission](native-v4-admission.json), [fresh generalist coordinator](native-v4-science-submission.json) and [same-node continuation](corrected-generalist-continuation-submission.json) retain the exact source/checkpoint and scheduling boundaries.
- [Teacher diagnostics](teacher-diagnostics/) retain the failure reproduction, buffer intervention and production-helper proof with all eight rank records and original completion hashes.
- [Independent V9 replay](pilot-v9-independent-replay-submission.json) and [14-check report compilation](pilot-v9-qualification-compilation-submission.json) completed successfully. Their [actual replay](../../experiments/verification/reviews/direct-pilot-v3-full-v9.json), [compiled report](../../experiments/verification/reviews/direct-pilot-v3-qualification-report.json) and [separate scientific approval](../../experiments/verification/reviews/pilot-v3-scientific-evidence-approval-20261113.json) preserve the complete evidence bindings.
- [Replacement preparation and qualification chain](pilot-v3-guard-v2-full-queue.json) preserves the sealed original data, vocabulary, Q and all 96 schedule/mask files. Failed earlier preparations and the old 207-update scientific attempt remain recorded.

Source review, completed engineering, scientific approval, actual training and held-out results are separate states. The full forty-epoch generalist, 24 completed pilots, evaluated quality/speed and later PDF studies remain outstanding. Queued jobs are not completed results.
