# Two-node teacher generation

This migration accelerates the unfinished ShareGPT/UltraChat target-response generation. It does not change the scientific training admission policy or turn engineering checks into benchmark results.

Use two full nodes, each with eight A100 80 GB GPUs: the protected deep10 holder and a new high-priority/spec-priority deep11 allocation. Sixteen running GPUs fit the user's eighteen-GPU allowance. Slurm handles background preemption on deep11. Never cancel either protected holder job, its continuation, or another user's jobs.

## Frozen response policy

The original population contains 540,107 conversations. Preserve every committed original outcome, including exclusions. Keep the original pinned teacher model, tokenizer, source scripts, decoding settings and per-example seeds. New scheduling can change FP16 near ties, so this is a new source-bound generation plan rather than a claim of byte-identical continuation.

Before stopping the original workload, each secondary GPU must perform actual teacher generation on one frozen 64-conversation tail batch, produce positive generated-token counts and at least one complete response, and retain all 64 outcomes. The 512 qualification outcomes are part of the original source population. If the original worker already committed an overlapping conversation, its outcome wins regardless of response quality; retain the second result as qualification evidence.

Persist the handoff intent before canceling the verified owned CPU coordinator and its exact named GPU workload step. Hold both original writer locks, verify stopped Slurm state, copy raw journals, record incomplete tails without modifying them, and seal the new plan. A disk-backed index checks identities and exact population coverage. No completed responses are deleted or regenerated.

## Shared queue and recovery

Both nodes use per-chunk NFS locks with atomic, fsynced, no-clobber result publication. Prove cross-node lock exclusion with all sixteen distinct GPU UUIDs before processing the remaining queue. A busy queue is not a completed queue. Interrupted unfinished chunks can be claimed again; completed chunks cannot be overwritten.

The protected-node CPU supervisor resolves the live holder authority again after allocation expiry. The second-node recovery schedule is hash-bound in the bootstrap and only retries infrastructure terminal states: timeout, node failure, preemption or boot failure. Ordinary application failures, user cancellation and out-of-memory stop automatic admission. Recovery checks the high-priority allowance, adopts a previously submitted job after a controller interruption, and preserves immutable hardware-attempt and history evidence. Supervisors have bounded deadlines and retry counts.

The final CPU merge validates input, model, runtime, hardware and source provenance; proves every source ordinal occurs exactly once; streams sorted canonical/native splits; and atomically publishes the completed corpus. Native tokenization and train-only vocabulary construction remain a separate required QA step.

## Automatic successors

`deploy.py` is scoped to this recorded CSAIL migration. It refuses a reused destination, requires an independently approved exact adapter hash, rechecks owned live jobs, and freezes a separate bundle. It holds migration until the replacement successor DAG is complete and only the seven verified old pending TAPS successors have been retired.

The new sequence is:

1. CPU bootstrap and eight-GPU secondary qualification.
2. Durable original-workload handoff, then sixteen-GPU generation.
3. Exact-coverage merge and native data/vocabulary QA.
4. Independently admitted 200-update scratch-native training profile.
5. Full forty-epoch generalist training and conditional checkpoint continuation.

The existing source-matched restart/export qualifications, numerical classification, scientific evidence review and pipeline certificate stay in force. Subsequent PDF stages still require trained-model evaluation, ablations, specialization and final analysis. Neither data generation nor the profile alone establishes a trained-quality result.

Execution receipts live under `outputs/preparation/parallel-generation-20261111-v1`; generation artifacts live under `outputs/teacher-generation/sharegpt-ultrachat-parallel-20261111-v1`. Do not redeploy changed code into either active directory. Compare measured generation throughput before and after the handoff instead of assuming a twofold speedup.
