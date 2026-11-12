# Scientific trainability blocker: nested reentrant checkpoint output

Decision: the legacy default-checkpoint path must not be admitted as training the intended full drafter until the actual diagnostic is resolved. This finding does not alter any running job, old report, source or prior receipt. It is independent of paper agreement. The earlier pilot source decision authorized bounded diagnostics only; it did not admit scientific training.

## Exact reviewed sources

- `cnets.py`: SHA256 `71295b56d8bd5648a1c630af0995f6a23a13671c8e60fe92636fffd9ee786a9c`, in `/mnt/nfs/home/mohamadzbib/projects/taps-rebuild/repos/EAGLE/eagle/traineagle3`.
- Pinned Torch2.6 `torch/utils/checkpoint.py`: SHA256 `7a05799e678085df1b81d512bed679c35053fb6225ea90c96661306f93140f6c`.
- Approved historical native helper: SHA256 `2aad19f7aefddfb70dd02b78d2eaf9709c90834d2308d055a836be83de5fec51`.

## Source mechanism

`LlamaDecoderLayeremb.forward` sets `outputs=(hidden_states, return_hidden)` at `cnets.py:441`, then returns `outputs, latest_hidden_cache` at line444. The outer return therefore consists of a nested tuple and nested cache lists, rather than a top-level Tensor.

`Model.forward` sends this function to `torch.utils.checkpoint.checkpoint` at lines806–813 without a `use_reentrant` argument, then consumes `layer_outputs[0]` at line827. The installed Torch implementation assigns `use_reentrant=True` when omitted (`checkpoint.py:464–474`) and calls `CheckpointFunction.apply` at line489. Its reentrant forward executes the function inside `torch.no_grad` at lines263–265; the pinned implementation explicitly documents at lines409–411 that nested tensors do not participate in reentrant autograd.

Consequently the nested hidden output can lose its graph into `fc` and the entire `midlayer`. The outer `norm` and `lm_head` still construct a trainable graph after the checkpointed output, so scalar losses, their logit gradients, optimizer steps and genuine inference can all look valid while most intended parameters receive no gradient.

This is not an intentional parameter filter: the native constructor creates `midlayer`, `fc`, `norm` and `lm_head` as ordinary trainable modules. It explicitly freezes the separate target and embedding. The native helper's `training_view` requires `gradient_checkpoint=True` and exposes `gradient_checkpointing=True`. The committed raw checkpoint has all13 intended parameters in its optimizer shape map. The existing raw format probe observed zero moments in the first four ZeRO partitions and nonzero moments in the later four; that pattern is consistent with detached midlayer/fc and live outer norm/head, but a named-state diagnostic is needed rather than inferring every tensor from a coarse partition flag.

## Read-only diagnostics prepared

`checkpoint_nested_gradient_probe_20261112.py`, SHA256 `4f9b3c6aa137ea25d71cdc6992659a30cc2c8b3fadf2d2b12cf8e6b18f7a2627`, executes the exact pinned decoder-forward AST with tiny substitute modules, after importing the pinned DeepSpeed runtime. Seven-step runs compare checkpoint-off, default, explicit reentrant and explicit nonreentrant modes with identical weights. It records actual per-parameter gradient participation and returns a factual reproduced/not-reproduced result. This is a CPU mechanism diagnostic, not a full8B execution.

`raw_named_adam_coverage_20261112.py`, SHA256 `7258421e55b15f4e1f09a54d5937db89699bf4c6572fe4f80dd9f0f0f3caca7b`, uses actual raw optimizer fragment maps and first/second moments to report all13 named tensors' nonzero coverage, after verifying all checkpoint bytes before and after. It is a raw saved-state observation, not a new backward execution.

## Repair and release requirements

Use a new isolated, source-bound adapter with explicit nonreentrant checkpointing, or an explicit checkpoint-off path, and qualify it against the pinned checkpoint-off native forward. Preserve the old failed evidence and active source snapshots. Tokens and standalone target-only gate scans are unaffected by this training-graph repair and can remain bound to their original identities.

The corrected production path must show backward-hook participation for every intended parameter, not post-backward `.grad` presence, since ZeRO clears/partitions those gradients. Require named raw Adam coverage and meaningful positive updates across real original/target cases. Compare corrected checkpointing to checkpoint-off for all intended gradients and optimizer updates, then rerun actual eight-rank process restart and export qualification on the exact corrected bytes. Two executions of the same broken default path cannot establish full-gradient correctness by agreeing with each other.

Do not use this source finding alone to claim what every past runtime did. Inspect the actual CPU mechanism and named checkpoint results, then retain any bounded full-model diagnostic needed to confirm the deployed path. No stop/cancel/source-change action was performed by this reviewer.
