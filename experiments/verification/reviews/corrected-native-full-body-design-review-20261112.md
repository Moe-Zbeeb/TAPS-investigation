# Corrected native full-body wrapper design review

Decision: the isolated wrapper approach is feasible. Preparation and bounded diagnostics can proceed after the actual wrapper is reviewed. This is design advice, not source approval or scientific training admission. The unadapted legacy path remains blocked.

## Reviewed sources

- `native_training.py`: `2aad19f7aefddfb70dd02b78d2eaf9709c90834d2308d055a836be83de5fec51`.
- `native_recovery.py`: `4293c07e350a4bafbb18520050888daf7a7d84a7760c5da9e1e7db7649a34c10`.
- `training_handoff.py`: `3e3ea9b80c6c81bd91e8fcecf8e46006cdb08f4f6b2ca9a5e1495f427f211a9e`.
- `native_gradients.py`: `594ee0ee196f9deceecdfe7af7a2bda6b8ee728f2aa9c222748689755c4f4d1a`.
- `admission_v2.py`: `3a14cc3365513de02f6543ccae6299144cf343fad4b32b3d7fadf619703f2b49`.
- Native `main.py`, `cnets.py` and `ds_config.json` were inspected read-only on the cluster and checked against the exact hashes in the legacy helper's `UPSTREAM` map. No active files or jobs were changed.

## Plan and launch integration

`native_training.load_plan`, lines394–425, verifies the plan self-hash, original run location, every frozen file, model inventory, storage marker, configuration and research gate. It does not reconstruct the command. A new plan can therefore point its torchrun worker script at `inputs/native_full_body.py`, provided it also binds that file and all imported dependencies, updates `models/.../native-plan.json` to the new plan identity, and creates all tokenization/vocabulary receipts against the final identity. Never alter the already-running or retired plans.

Changing the command alone is insufficient. `native_training.launch`, line986, starts a separate audit subprocess through `inputs/native_training.py`. A wrapper-only research-gate override in the parent does not propagate to that child. Route this exact audit invocation to the wrapper's audit entry, or provide a separately source-bound launcher implementation. A narrow exact-argument subprocess router is possible; it must reject unexpected native-audit commands and leave all other execution behavior intact. Qualify that route, rather than treating a successful parent audit as evidence the launch path is correct.

Keep `run_native_main`, its `__main__` binding and `expose_spawn_collator` unchanged. Their expected path is the frozen `inputs/native_training.py`; multiprocessing children must still expose the exact upstream collator AST without executing training setup. The new torchrun wrapper can call the loaded legacy worker while the legacy helper retains this internal identity.

The old v2 stage verifier authorizes exactly200 profile updates or a full stage backed by the old contract. It cannot express a four-step scratch qualification, and its former training interpretation is insufficient after the gradient-path finding. Add a distinct mandatory full-body-v3 gate in the wrapper. Keep the existing pipeline certificate/review as independently verified inference/data prerequisites. Use new corrected-qualification/profile/full stage names so an accidental direct invocation of the old native worker rejects the new plan. The wrapper's gate must bind exact wrapper/adapter/legacy/recovery/upstream/runtime/data/vocabulary identities and enforce the registered stage ceiling. Do not return an old training-approval result as the new approval.

The existing `training_handoff.py` similarly lacks wrapper, policy and full-body evidence bindings. Reuse its inventory and export logic where helpful, but a new wrapper-aware handoff must require corrected source-matched qualifications and a corrected profile; old restart/export/profile success cannot satisfy this new gate.

## Minimal execution hooks

Load the exact frozen legacy helper as a module. Wrap its `install_model_adapter`: call the original adapter, then subclass the resulting native Model. After `scandata` returns, prime and freeze the RoPE cache before DeepSpeed conversion. Override `train(mode)` to call the parent and then keep the frozen target in evaluation mode. Assert target/embedding remain frozen and all13 intended trainables are present. Assert target evaluation mode at each online target forward.

Install explicit nonreentrant checkpoint routing before the native model is constructed. The direct reference must call the same checkpointed callable with identical native arguments and the same priming, preserving the native `gradient_checkpointing` branch. Priming is a new numerical recipe, not equality to the old extension-from-FP16 behavior.

Wrap the legacy step-initialization adapter externally to install named Tensor hooks. Preserve its recovery object, native loader/sampler, AST loop transform and data cursor. Do not replace the trainer or silently modify loss normalization, DeepSpeed configuration or the native per-batch `model.zero_grad` behavior. Every new initial run must have fresh empty model storage, no old checkpoint discovery and no inherited resume environment. Resume is allowed only from checkpoints of this exact corrected plan.

## Native-recipe qualification differences

The exact native DeepSpeed configuration uses gradient accumulation2, dynamic loss scaling starting at2^14, warmup12000 and total800000. Native main enables TF32 and uses full native assistant masks with variable padded shapes. These differ from the pilot's GAS1/static128/fixed2048/selected-context recipe.

Do not call the pilot coverage `finish()` unconditionally before native step. It would reject overflow gradients before the dynamic scaler can record a legitimate skipped update and lower its scale. Observe named callbacks for every backward and retain each accumulation window. Require all13 participation; require finite nonzero coverage on successful optimizer windows. Retain nonfinite flags, overflow/skipped counters, scale changes and all attempted work for skipped windows. Reject silent missing-body callbacks even when a step is skipped. Put a finite attempt ceiling on engineering qualifications so a request for four successful updates cannot run indefinitely.

At native warmup learning rates near1e-9, a small early FP32 update to a norm initialized at1 may round away. Require all13 nonzero saved moments in the short qualification and meaningful per-parameter FP32 changes by the corrected200-update profile, or in a separately registered positive-learning-rate diagnostic. Do not apply the pilot's step2 master-change criterion to this different recipe without checking its numerical power.

## Minimum fresh evidence before forty epochs

1. On all eight ranks, use the exact native scratch initialization, frozen real-data vocabulary, native full masks, native dynamic scaler/GAS2 and corrected adapter. Compare nonreentrant execution to the direct-function reference on matched real native batches for at least four successful updates, with all13 callbacks and named raw Adam coverage. Compare pre-step gradient partitions, losses, raw masters/Adam/scaler/scheduler, RNG and actual counters; retain failures without widening bounds after observing them. Include a native maximum-length2048 case, or a separate bounded maximum-shape probe, to exercise the2054 RoPE capacity and peak memory.
2. Run a fresh process4-vs2+2 restore under this same recipe on all eight ranks. Compare the complete attempted batch sequence, accumulation state, global/successful/skipped counters, dynamic scaler fields, all raw parameter/Adam/master states and CPU/CUDA/NumPy/Python RNG. A real-data engineering subset can test an epoch boundary separately; label that subset honestly. Neither same-process roundtrip nor matching successful counters is sufficient.
3. Independently verify the corrected native export's complete trainable/vocabulary identity and genuine stock inference. The corrected pilot export does not replace this native-wrapper export test.
4. Complete a fresh corrected200-successful-update profile on the actual full corpus with the final wrapper bytes. Bind data/tokenization/vocabulary inventories, physical GPU mapping, per-parameter learning/update evidence, actual memory and time, overflow/skips and source/runtime identity. The original head-only profile's timing and memory are invalid for this purpose.
5. Independently inspect these actual retained reports and bind the new qualification/profile report identities and full artifact inventory in the new gate. Then admit one fresh scratch forty-epoch run with the corrected recipe. Pilot admission, input QA and paper matching remain separate scopes.

The corrected pilot can supply evidence about the shared adapter and selected-context numerical machinery. It cannot establish the native full-mask dynamic-scaler/GAS2 path, the new launch/spawn integration, or the full native recovery/export behavior. No production gate or admission override was authored by this reviewer.
