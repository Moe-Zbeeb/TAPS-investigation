# Independent v2 implementation checkpoint

Decision at this checkpoint: **changes required; evidence approval pending**. This document is an implementation-review record only. It is not an approved production review JSON and does not authorize scientific training.

I reviewed `admission_v2.py`, `numerical_validation.py`, the retained-vector collector/protocol, the v2 policy and the native-training branch independently of their author. Source changed while the review was running; the exact initial examined identities are below. This checkpoint must not be treated as approval of later bytes.

## Supported design

The new report separates CPU reconstruction from independent evidence admission and pins the review to the exact evidence identity plus complete artifact inventory. Admission rehashes content rather than relying on file stats. The policy honestly preserves historical reproduction failures, explicitly removes only the paper-agreement prerequisite, discloses retrospective prompt reconstruction, and confines the statistical and numerical claims to their evidence scopes.

The numerical observer captures both repeated native route vectors, a distinct clean full-prefix cache, actual native selections, common prefixes, tree candidates/retrieval, masks, positions and cache metadata. The selected-path mask/position checks are structurally appropriate: each selected tree node must see exactly its ancestors and use the corresponding absolute position. Pinned EAGLE's fresh-cache allocator initializes data and logical lengths to zero. The retained inherited arena remains alive during clean allocation, so the pointer-disjointness check is meaningful. Repeated complete vectors, isolated top-two support, an epsilon fixed before confirmation, and a clean-reference decision check provide materially stronger evidence than the former top-five summaries. The exact kernel cause may remain unresolved and must stay explicit.

## Concrete issues communicated

- Initial `numerical_validation.py:73-106` self-bound the selected eight cases without pinning the original selection or numerical source/model/provenance/hardware identities. Current revisions add the original cases SHA, compare selected metadata with registered EAGLE-3 and bind row provenance/hardware/runtime. These repairs need final-source verification and real CPU replay.
- Initial `numerical_validation.py:46` incorrectly rejected offset equal to candidate-row length, although an all-accepted row can legitimately supply the next-root distribution from its last logit. Current revisions permit equality while retaining complete path checks.
- Initial numerical replay expected `hardware.tokens` and later `hardware.slurm_job_id`, but the collector uses `pool.allocation`, which supplies `device_bindings` and `job_id`. Current revisions correct those keys. GPU lookup should use each binding's UUID/index rather than assume device-binding and sorted inventory order coincide. Require eight unique physical GPU bindings.
- Initial `admission_v2.py:100-111` did not compare the loaded run manifest/config to the fixed prepared provenance, permitting a changed manifest to supply new authored-source hashes. Current revisions add the necessary prepared-provenance digest checks.
- `admission_v2.py:47-50` initially requires a CPU-FP16 softmax-derived TV scalar to equal the original CUDA-FP16 scalar within 1e-12. Cross-backend softmax rounding is not guaranteed equal. Resolve this before interpreting replay: preserve the original CUDA diagnostic and define any new CPU FP64 diagnostic explicitly, or justify a frozen comparison rule. The diagnostic must not silently redefine the registered event reference law or its confidence bounds.
- The native v2 branch initially binds implementation bytes but does not enforce the actual training target/world-size/runtime/recipe or profile-versus-full stage. A scope string is insufficient: the same reviewed certificate can otherwise authorize a changed configuration or a direct full run outside the guarded 200-update profile and qualified handoff. Encode and enforce the allowed configuration/stage, and keep source-matched restart/export and data/vocabulary QA requirements concrete.
- Strengthen numerical cache validation to require the exact single-GPU target arena dimensions and integer 64-entry logical-length tensors, including clean post-forward lengths. Metadata list lengths and dtype strings alone do not validate the claimed arena shape. Bind retained tensor hashes consistently to the bytes consumed by replay.

No source changes were made by this reviewer. No large-vector evidence was available for this checkpoint, no CPU Slurm reconstruction was independently executed here, and no job state was inspected or changed. The author's reported capture jobs are not evidence that classification passed. Focused local source tests (`tests/test_admission_v2.py` and `tests/test_native_training.py`) ran 35 tests successfully; those tests do not exercise retained numerical vectors on this Torch-free local interpreter.

## Initial examined identities

| Material | SHA256 |
|---|---|
| `src/taps_rebuild/admission_v2.py` | `96563dd58a7b49270b908170efe93c257b9e9ab789b389f0dbc81a7c2d286a37` |
| `src/taps_rebuild/numerical_validation.py` | `dd449e6036bfd281e1220e5280ca640c93689c85c4ae746e36fadda34367d6dd` |
| `experiments/numerical-confirmation-20261111/driver.py` | `7a18bfc90886c4133dc0641b2c1bd42c69eae65d629efe8a829d8c374131f89c` |
| `experiments/numerical-confirmation-20261111/PREREGISTRATION.md` | `522f98f6ac1b5ae118998d3bdfd077ea12241f61f7b309c05783e034ed769e1c` |
| `experiments/verification/PIPELINE_VALIDATION_V2.md` | `9040d672eaea77648837b188f68347fa3548e467abf84741ac03505bb2c7731f` |
| `src/taps_rebuild/native_training.py` | `c6afd124a566b2b11d3973d34c99284f940a45340f2df24e62de06265a527b07` |

The fresh-cache allocation source was inspected at the pinned primary source: [EAGLE kv_cache.py](https://raw.githubusercontent.com/SafeAILab/EAGLE/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/kv_cache.py). This source confirms allocation behavior; source-bound actual capture and CPU replay still determine whether the experiment satisfied it.
