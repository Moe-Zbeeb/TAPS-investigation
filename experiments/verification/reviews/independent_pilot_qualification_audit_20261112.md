# Independent CPU qualification analyzer

Run only inside a CPU Slurm job with `CUDA_VISIBLE_DEVICES=''`:

```sh
python -B independent_pilot_qualification_audit_20261112.py --study /project/outputs/pilot-study/STUDY --qualification /project/outputs/pilot-qualifications/STUDY --output /project/outputs/pilot-qualification-audit/REPORT.json
```

The project root is inferred from the study manifest. `--root` can be supplied explicitly. The code imports no project implementation helpers. Torch is imported only for CPU inspection of retained tensor evidence; the schedule/state-trace analysis otherwise uses the standard library.

This version reconstructs every schedule and sampled mask, the Boolean cache/gate intersections, source/model/data inventories, all seven FP64 CE and scaled FP16-logit gradient references, full-row sparse outside-mask support, and retained raw-native/update and 4-versus-2+2 state traces. It checks that observed positive-LR updates actually changed owned FP32 masters. It records all consumed file hashes and rehashes their bytes at the end.

It reports unavailable/partial coverage explicitly. In particular, the initial version does not certify the alternate-pad comparison or stock export compatibility and does not independently reconstruct raw full ZeRO Adam/scaler states from their serialized checkpoints. Hash-state trace equality is labelled as such. It never writes scientific approval or admission artifacts.

For a partial/selected input set, `--manifest REQUEST.json --root /project` accepts schema version1 with optional `study_manifest`, `gradient_cases` and `state_comparisons`. Every artifact descriptor is `{"path":"...","sha256":"..."}`; SHA256 is optional only when recording a new evidence identity for subsequent inspection. A state comparison has `kind` equal to `native_step_parity` or `restart_parity`, `name`, optional `group`, and `left`/`right` each containing `rank` and `segments`. Each segment supplies `initial`, `journal` and `complete` descriptors. Native parity uses one four-step segment per side, with one actual native-reference side. Restore parity uses one four-step left segment and two right segments covering0–2 and2–4. An eight-rank group must share one plan/source binding.

Gradient evidence follows the exact `Guard.evidence` fields. It requires FP32 returned full-vocabulary teacher logits, FP16 draft-head logits and scaled logit gradients, static scale128 and the declared head weights/normalization. New full-row sparse gradient captures permit independent outside-mask support verification. Older selected-only captures cannot pass that check. Signal-bearing qualification coverage must include every head in chat/math/code under original/target contexts.

The default report contains all supported and unsupported check statuses. `passed_requested_checks` is deliberately scoped to the named checks and is never a scientific certificate. `partial_evidence` is a successful analysis with missing evidence, not an approval. `failed` includes a fixed error code and consumed-artifact inventory without printing private input text or vectors.
