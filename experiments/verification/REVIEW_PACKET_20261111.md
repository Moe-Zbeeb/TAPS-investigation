# Independent validation review packet

This packet makes the existing review requirement concrete. It supplies no approval and changes no training gate. Review of the validator, diagnosis of numerical differences and successful baseline reproduction are separate requirements.

## Frozen materials to review

Paths below are relative to `/mnt/nfs/home/mohamadzbib/projects/taps-rebuild`. Their hashes were checked on 11 November 2026. Review these frozen copies, not an unverified replacement from a newer checkout.

| Material | Cluster path | SHA256 |
|---|---|---|
| Full-model validator | `outputs/preparation/training-queue-20261110-v6/src/taps_rebuild/admission.py` | `028747152f4c4bbbeca49120aaa72100b6d5293439411840d9546815a6ec3c0b` |
| GPU collection driver | `outputs/preparation/validation-queue-20261110-v2/driver.py` | `d6ea2b5216b0052aa68282646b343143c6523ed14e75f8b2b838f99bb6458934` |
| Registered tokenized cases | `outputs/validation-inputs/registered-20261110-v2/cases.json` | `d3f236a73ab18940bd28f76112210449a2f6ef797061d50a1ab98fc059e0a865` |
| Verification proof | `outputs/preparation/training-queue-20261110-v6/experiments/verification/LOSSLESS_RULE.md` | `94de731a8918f84f2126cb8fc9dffd92298c25184595ad04ea8f7e3820a4113d` |
| Sampling proposal | `outputs/preparation/validation-queue-20261110-v2/TEMPERATURE_ONE_AUDIT_PLAN.md` | `dcba7076095d4ffac978bc139889d9d205fcaa7f898bc9b7dbe2c1381d15638e` |

The readable source counterparts are [the validator](../../src/taps_rebuild/admission.py), [collector](../validation-queue-20261110/driver.py), [proof](LOSSLESS_RULE.md) and [sampling proposal](TEMPERATURE_ONE_AUDIT_PLAN.md). The proposal's initial status text predates implementation and execution. Its frozen definitions and the actual source-bound receipts determine what ran.

## Completed evidence and its limits

- Conditional audit: 60 fixed real-model states, 65,536 posterior draws and 65,536 independent target-reference draws per state; 960 registered event comparisons passed. There are 7,864,320 draws across the two arms. The event checks do not bound full-vocabulary total variation.
- End-to-end screen: 61,440 continuations, 20,480 per method; all 36 registered mixture-event comparisons passed. The continuation is the sample unit. The 6-point margin and the short fixed-prompt mixture do not establish general long-sequence equivalence.
- Greedy controls: two complete FP32 panels each retain three failures among 1,120 pairs. Two are explained by context stopping; the remaining token decision is under separate [diagnostic investigation](../remaining-greedy-20261111/PREREGISTRATION.md).
- CPU raw replay job `2214995` recomputed all registered phases. Conditional and sampling checks passed; greedy and independent review remained false. Its report is correctly blocked. The [retained replay](../../results/data-2026-11-10/full-raw-recheck.json) preserves its exact artifact inventory and hashes.
- The original 22-cell baseline failed and remains authoritative for its registration. Later replications are separately recorded diagnostics and do not replace its admission identity.

## Review questions

1. Does the fixed-child residual proof cover the actual implementation's duplicate handling, padding, child order, probability-one/exhausted residual cases, fallback sampling and target-sampled root? Are all candidate-selection decisions fixed before fresh verification randomness?
2. Are conditional AR distributions recomputed independently at the identical prefix with tree masks disabled and cache lengths reset? Check that tree logits are not reused as the reference law.
3. Do the native CUDA posterior calls, residual draws and independently seeded target draws implement the registered sample units and budgets? Does the collector preserve complete histograms, event masks, source identities and coverage failures?
4. Do both simultaneous confidence families use the registered event definitions, independent-arm assumptions, population weights, alpha allocations and fixed margins? Confirm that no full-TV or universal losslessness claim follows from these event screens.
5. Does the CPU validator reconstruct each claim from raw records and tensor captures, reject incomplete/duplicate coverage and bind source, model, prompt, seed and artifact identities? Check the certificate's later immutability verification.
6. What numerical and stopping interpretation is appropriate for the observed greedy differences? The current strict validator rejects all three. Any proposed policy change needs a new registration, validator identity, complete evidence and review; classifying a failure does not rewrite it as a pass.
7. Is the proposed training-admission scope justified for the specific target/dtype/runtime/tree implementation? Identify which changes require new validation. A future pooled/shared or serving backend is outside this audit's scope.

## Decision record

The existing validator expects a nonempty reviewer identity and an explicit decision bound to the validator, collector, proof and proposal hashes. No review file existed at `outputs/training-queue/sharegpt-ultrachat-native-20261110/review.json` at the live check. This packet deliberately does not create one or infer an approval from tests.

Record findings and requested corrections first. Approving the validator alone cannot pass failed greedy/reproduction checks. After any corrections, freeze the reviewed artifacts and regenerate the full source-bound report; do not manually edit an old report or certificate.
