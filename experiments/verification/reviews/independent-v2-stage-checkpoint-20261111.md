# Independent v2 stage-scope checkpoint

Decision: **changes required for full-stage integrity; evidence approval remains pending**. No production review or training authorization is issued.

The latest candidates repair the previously reported numerical population/provenance binding, valid full-row offset, hardware schema/mapping, prepared source-manifest binding and cache geometry checks. The conditional TV diagnostic now clearly distinguishes CPU FP64 reconstruction from the retained CUDA scalar without changing registered event tests. The new training contract concretely constrains target, runtime, source recipe, eight ranks and the 200-update profile ceiling.

I found the following remaining integrity gaps in the newly added full-stage checks. These concern new stage logic, rather than the already repaired numerical collector.

1. `src/taps_rebuild/admission_v2.py:252-259` verifies profile-plan file hashes but not the plan's canonical `plan_identity_sha256`, and accepts rank receipts based only on rank/status/200 successful steps. It does not require each rank's plan identity, world size and Slurm job to match that qualified profile. Thus unrelated completed 200-update rank receipts can be mixed with an otherwise passing profile plan. Require the actual profile plan identity and a pinned launch receipt, then compare all rank identities and the eight-rank job binding. The existing native worker receipts already contain these fields.

2. `admission_v2.py:260-262` verifies qualification `native_training_sha256` but omits `native_recovery_sha256`. The adjacent `training_handoff.qualifications` helper checks both. The independent full-stage verifier should also bind both helper identities to prevent accepting a restart qualification for another recovery implementation.

3. `admission_v2.py:263-264` checks only QA's train hash and cache-key presence, not the dev/validation hash, actual cache hash or complete QA identity. More directly, the actual full-run train/dev inventories are not checked against the qualified profile plan: `src/taps_rebuild/native_training.py:338-343` accepts any fresh disjoint train/dev data after the config-only gate passes. An otherwise valid full-stage configuration can therefore be supplied with changed data while still claiming the qualified profile/QA. Bind the actual prepared train and validation hashes to the profiled/QA dataset identity, and preserve that relationship at plan load/launch. The intended handoff uses the same frozen paths, but the source-level gate should enforce the claimed constraint.

The numerical observer remains suitable for collecting the registered confirmation, pending actual vector and CPU replay results. Focused tests for v2 admission, native training and handoff ran 38 tests successfully. They do not substitute for retained-vector evidence or the repeated source-matched eight-GPU qualification.

Exact candidate identities at this checkpoint:

| Material | SHA256 |
|---|---|
| `src/taps_rebuild/admission_v2.py` | `86836ae6bfa6d315f1df333bacbf95191fd5b124ab69ed61eb79c6116d5f729c` |
| `src/taps_rebuild/numerical_validation.py` | `cb0546c268c09d73e15f62bde6d55a9a3437d26cce24ef464c14a9138d84629a` |
| `src/taps_rebuild/native_training.py` | `ea69b4dbabc66408cabfdef87cf0f99fa776bf6f8a60af10b1053d7bc3e07662` |
| `src/taps_rebuild/training_handoff.py` | `432c1bba790dd848591dfa7f0e65a6a1bd4b8f48a1f1b04f824f75e1c7e2e0ab` |

Source changes during continuing implementation require a new final hash-bound review; this checkpoint does not approve future bytes.
