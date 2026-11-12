# Decision: changes required before scientific admission

The 24-run design is feasible as a bounded, common-ancestor fine-tuning pilot. Preparing its immutable data and dependency graph can proceed. The proposed strict token/compute claim needs the definitions and safeguards below before training is admitted. This review does not approve implementation or execution. No jobs or running sources were changed.

The inspected upstream sources are `/mnt/nfs/home/mohamadzbib/projects/taps-rebuild/repos/EAGLE/eagle/traineagle3/cnets.py`, SHA256 `71295b56d8bd5648a1c630af0995f6a23a13671c8e60fe92636fffd9ee786a9c`, and sibling `main.py`, SHA256 `439c4d8de398c8876f72806d7ba7e6436b5127a4751f3491505a3f51412a4eeb`. The existing native helper `2aad19f7aefddfb70dd02b78d2eaf9709c90834d2308d055a836be83de5fec51` and validator `3a14cc3365513de02f6543ccae6299144cf343fad4b32b3d7fadf619703f2b49` remain the approved, separate generalist contract.

## Seven-step alignment and suffix padding

At `cnets.py:723–724`, teacher logits and input IDs shift left once while the mask remains in place. At lines 862–864, IDs, logits and mask shift again after each step. For a selected original mask index `s`, step `k=0..6` therefore contributes at grid position `t=s-k` and uses the teacher logits at original input index `s+1`. Those logits predict the distribution of the token after that context. Do not reinterpret the selected index as the predicted token index or silently repair the official alignment.

Require every selected `s` to satisfy `s >= 6`, `s + 1 < L_real`, and the frozen native assistant-mask eligibility predicate. If the intended supervision additionally requires the teacher context itself to be inside an assistant span, explicitly require the native mask at `s+1`; that is an additional selection rule. Selecting terminal/pad-reaching indices changes the target context, and selecting early indices loses some later-step contributions.

With those bounds, right-appended EOS tokens with attention ones are mathematically safe for the retained contributions under this source. The base cache has causal attention (`cnets.py:280–284`); subsequent recurrence caches are accessed at the same grid index (`286–305`). At step `k`, retained grid `s-k` depends on original input indices no later than `s+1`. Future suffix tokens cannot reach that loss or its gradients. This is a mathematical dependency argument, not a promise of identical finite-precision kernels or RNG consumption.

The loss at `cnets.py:855` averages over all `B*T` positions, including zeros. At fixed `T=2048`, its value and gradients differ by `L_real/2048` from an unpadded single-example calculation with the same selected mask. Padding parity must compare per-position losses, or rescale the aggregate denominator explicitly. The proposed sparse mask also changes the ordinary full-assistant objective; identify it as the pilot objective. Require finite tensors across the complete padded computation because zero times NaN does not remove NaN.

## Counts that are and are not matched

The actual position mask is `native_mask * t2d[argmax(teacher_logits)]` (`cnets.py:833–855`). A base mask with 64 ones does not guarantee 64 nonzero effective losses. The gate can vary with domain and response regime. The seven shifted heads use the same original teacher-logit index for a selected `s`, so the above boundary guard permits exact headwise accounting.

Choose and freeze one honest policy:

1. Prefer the minimally selected population if the primary question is response/domain adaptation: keep 64 guarded raw mask positions, report exact effective counts and weighted loss mass, and call the comparison raw-mask-position/dense-slot matched. Do not claim exact effective-token matching.
2. If exact effective matching is required, select 64 distinct guarded, gate-eligible contexts in each response of each paired ID. Freeze that eligibility before training; verify the gate/count invariant online. Use common relative-quantile variates within each pair. Report that this conditions the estimand on vocabulary-covered, sufficiently long responses. Store only eligibility/argmax summaries, not full teacher vectors. This does not make real nonpadding context-token totals equal.

For `N=12288`, raw base-mask positions total 786432, dense input slots total 25165824, and seven-step raw contributions total 5505024 only when all boundary guards hold. Log real nonpadding tokens, raw mask counts, per-step effective gate counts and `sum(0.8**k * effective_count[k])` separately. The source weights are at `main.py:284–286`. Equal tensor shapes control dense operation counts; measured GPU time is a separate empirical result.

Exactly 1536 successful updates correspond to exactly 12288 processed examples only if every attempted update succeeds with microbatch 1, world size 8 and GAS 1. Any overflow/skipped step otherwise breaks the claim. Prespecify invalidation on a skip for the strict design, retain the failed attempt, and do not silently spend extra examples/compute to reach 1536 successes. A retry policy would require a different exposure definition. Freeze optimizer, LR schedule and warmup against the pilot budget, not an inherited 40-epoch total.

## Population, order and feasibility

Count paired native-length and mask eligibility before freezing `N`. The code domain may not have 12288 distinct paired training IDs with 64 eligible contexts. A defensible default is one common `N <= 12288`, divisible by 24, derived by a preregistered rule from the minimum eligible training count. Alternatively lower the common mask size before outcomes are inspected. Repetition is possible only if explicitly registered with unique-ID counts and exposure multiplicities. Never borrow dev/test rows to meet the budget.

Use identical paired IDs within each response-policy comparison. Make the union's equal domain allocations nested subsets of the corresponding specialist population. Freeze all splits and disclose paired-selection survival by source, response length and regime. Under a fixed total budget, the specialist has more in-domain exposure than the union; that is the allocation contrast being estimated. It does not isolate domain mixture from in-domain exposure.

Three seeds provide only three rows of an eight-condition Latin square, not complete position balance. Use an accurately named preregistered randomized seed-block order with partial counterbalancing and response-policy pairs spread across early/late positions. Do not choose order after viewing results. State whether seed changes initialization/shuffle/dropout only or also data/mask sampling. A common released ancestor with one fixed vocabulary supports later common-ancestor merging, but these results concern fine-tuning from that ancestor, not scratch training or the active generalist checkpoint.

Cache token IDs and native masks once with source/tokenizer/template/vocabulary identities; freeze separate seed-specific mask/index artifacts. Estimate wall cost from the new fixed-shape/GAS-1 qualification and separate measured update time from initialization, checkpoint, evaluation and export time. Do not extrapolate a different recipe's whole profile wall time. All 24 runs must remain sequential on protected deep10, with real restart/holder-boundary handling and no changes to the active generalist or protected holder services.

Define the +/-10% compute criterion in advance, including its reference, aggregation, initialization and failure/restart charges. Record end-to-end allocated GPU-seconds and training-phase measurements separately. Offline teacher regeneration cost must also be reported; matching online target-forward training cost is not matching complete response-policy cost. If the tolerance fails, retain the results and withdraw the compute-matched claim instead of adding sleep or training work. Shared-storage contention and cold caches remain measurement limitations.

## Required new admission evidence

Freeze the revised protocol, paired populations, vocabulary/ancestor and mask artifacts before training. Qualify native-mask/alignment reproduction; guarded padded/unpadded teacher features, all seven losses and trainable gradients after the known denominator scaling; one optimizer update; effective-gate counting; and finite padded computation. Include short, multiturn and boundary cases that would fail without the guard. Pin production precision/backend and prescribe training-specific numerical tolerances; the earlier decoder near-tie policy does not supply these tolerances.

Run fresh source-matched eight-rank recovery and inference-export checks for the new initialization, data/cache, optimizer schedule and mask implementation. Bind all rank batch/order/RNG/budget receipts. The old pipeline certificate can support its original scoped decoder evidence; it does not approve a different training source or override its existing training contract.

Register fixed final-checkpoint selection, common held-out evaluation traffic, baselines, contrasts and uncertainty before outcomes. Preserve all three seeds and failures. Queue the 24 scientific jobs with real preparation/qualification/review dependencies; keep them held until their new source-bound admission passes. No pilot training or merge approval is issued by this protocol review.
