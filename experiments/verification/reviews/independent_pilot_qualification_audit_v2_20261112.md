# Independent pilot qualification analyzer v2

V1 remains unchanged at SHA256 `32f142ee0142b54d6dd2cd4f0650656ff06d3fba609f55b40ab7ad3f87bce85b`. V2 loads only that exact adjacent reviewer file and imports no project implementation helpers. Run substantial work only in CPU Slurm, with explicitly empty `CUDA_VISIBLE_DEVICES`, using the pinned native environment.

```sh
python -B independent_pilot_qualification_audit_v2_20261112.py --study STUDYDIR --qualification QUALDIR --output OUTPUT.json
```

It automatically looks for `QUALDIR/runtime/pre.json`, `post.json`, and `interval-accounting.json`. These can instead be supplied using `--runtime-pre`, `--runtime-post` and `--execution-envelope`.

V2 reruns all v1 checks, then checks the following actual retained evidence:

- Both response policies, eight ranks and four steps of alternate-future-pad versus uninterrupted captures. Real input prefixes must match, suffixes must actually differ for each domain/policy, guarded target/head logits and scaled logit gradients must be tensor-equal, and normalized CE differences must be at most `5e-5`. Full pre-step partition identities, model/optimizer/scaler/scheduler/RNG traces and update norms must match exactly. Signed zeros compare equal in tensor comparisons. This strict same-shape invariance test does not reuse the decoder near-tie exception.
- Raw model, all eight optimizer partitions and raw rank checkpoint files at step4 for uninterrupted, raw-native-reference, split/restore and alternate-pad runs, plus the split checkpoint at step2. It independently computes semantic model/Adam/scaler/scheduler/RNG identities, verifies counters and static-scale/optimizer recipe, reassembles FP32 master partitions, and checks their rounded FP16 values against every saved trainable. It checks that fresh Adam moments are nonzero. The raw step2 state must match the uninterrupted step1 trace, and all four step4 variants must match. Frozen target/embedding weights are intentionally absent from these trainable checkpoints; their prepared assets and source bindings remain separate.
- Exact exported trainable/vocabulary identity against the final native state, all sixteen stock compatibility receipts, unchanged inference source files and genuine nonempty continuations. Greedy token equality is recorded, not required: this is export compatibility, not a new losslessness or quality result.
- Reviewed CPU pre/post runtime observations, unchanged packages/source/shared libraries and preexisting fused-Adam artifacts, with newly compiled files explicitly distinguished. GPU backend observations remain separate. Temporal qualification coverage must have successful Slurm accounting and dependency evidence.

The interval receipt has top-level `pre_job`, `qualification_job` and `post_job`. Each contains string `job_id`, `state: "COMPLETED"`, `exit_code: "0:0"`, timezone-aware ISO `start_utc`/`end_utc`, and string-list `afterok_jobs`. The qualification job must depend on the pre job; the post job must depend on the qualification job. Include the raw `TZ=UTC` sacct/dependency evidence in the receipt or its separately hashed evidence inventory for the agent's final review.

For a standalone validation of the old committed checkpoint format:

```sh
python -B independent_pilot_qualification_audit_v2_20261112.py --root PROJECTROOT --raw-checkpoint CHECKPOINTDIR --manifest-sha256 EXPECTED_SHA256 --output OUTPUT.json
```

This old-format probe reconstructs raw hashes and FP32-to-FP16 parameter consistency. It is not new pilot evidence. The earlier metadata probe omitted the optimizer-root hash because its conservative type whitelist did not support DeepSpeed's integer-subclass `ZeroStageEnum`. V2 encodes the actual enum type tag with its JSON numeric value, as specified by the native semantic hash format; it does not waive a mismatch.

Every consumed artifact is hashed and rehashed after analysis. State file deserialization is restricted operationally to hash-verified, trusted, committed checkpoints from the reviewed producer; raw pickle files are never downloaded. Reports contain counts, metadata and hashes rather than input text or tensor values. The analyzer never writes an admission/review approval. A complete result is named `all_checked_evidence_passed_requires_independent_review`; incomplete and failed checks remain explicit.
