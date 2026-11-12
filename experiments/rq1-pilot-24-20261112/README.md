# RQ1: 24 common-ancestor pilot trainings

The PDF's core pilot is four drafters × two response-context policies × three seeds. Chat, math and code specialists are compared with a balanced union generalist. Every run fine-tunes the same complete released EAGLE3 ancestor with the same released 32,000-token vocabulary. The separate ongoing 40-epoch scratch generalist continues independently.

Both policies use online, frozen Llama-3.1-8B-Instruct soft distributions. “Original” and “target” refer to the response text in the training context, not different hard-label sources. No teacher hidden-state feature cache is used.

The frozen recipe is [rq1-pilot-24-20261112.json](../../configs/rq1-pilot-24-20261112.json). Register the recipe, full schedules, source inventories and qualification tolerances before any pilot updates. These are bounded fine-tuning pilots; they do not reproduce a scratch model or establish the PDF's later scale, composition, routing or serving results.

## Matched logical budgets

Each run commits 1,536 successful GAS1 updates on eight A100 80GB GPUs: 12,288 example exposures and 25,165,824 dense token slots. The generalist receives 512 exposures per domain on each rank. Specialists allocate the same total exposure to one domain. Cycle permutations and repeat occurrences are deterministic and recorded; exposures are not claimed to be unique examples.

Choose Q=64 before training if every domain/rank has at least 128 paired eligible training groups, otherwise Q=32 under the same rule. If neither works, block training. Original and regenerated responses share IDs, splits, rank assignments, repeat occurrences and sampling variates. Each policy must have Q distinct eligible positions in every example. Selection uses disjoint integer strata, so marginal context inclusion need not be uniform when stratum sizes differ.

Eligibility retains the pinned native assistant mask, requires base index s≥6 and s+1 before the real sequence end, and requires the full-vocabulary teacher argmax at original index s+1 to lie in the released draft vocabulary. This permits all seven native shifted heads to receive exactly Q effective contexts. Each native loss mean is multiplied by 2048/Q before weighting by 0.8**head. The future suffix is padded to 2,048 tokens without truncating real text; its causal invariance requires qualification.

Any overflow or skipped update invalidates the strict run. Logical committed work and all physical attempts, replayed work and GPU cost are reported separately. Checkpoints retain optimizer, scaler, scheduler, RNG, exact masks and cursor; two complete checkpoints are retained, with older manifests archived. Weights and checkpoints stay under project `models/`.

## Automatic stage order

1. Verify pinned source corpora, target generation plans and canonical output bytes.
2. Prepare paired native token/mask caches on a CPU Slurm node. Native preprocessing parity covers both response policies in all three domains.
3. Scan eligibility on all eight GPUs of the protected deep10 allocation. Preserve the separate generalist's eight GPUs on deep11.
4. Select Q and freeze all 24 plans, complete schedules and repeat/coverage statistics.
5. Qualify the new source: independent FP64 loss/gradient reconstruction, native optimizer-step parity, alternate-future-suffix invariance, fresh eight-rank 4 versus 2+2 recovery and stock inference of the exported checkpoint.
6. Independently review actual evidence and integrate a source-bound admission receipt. Design approval alone cannot release training.
7. Run the randomized seed-block order sequentially on the same eight physical GPUs. Retain the protected holder and its successor. Report actual utilization separately from allocation counts.

CPU preparation and eligibility scans are not pilot training. A prepared plan or queued coordinator is not a completed run. Scientific inference comparisons require their own checkpoint and decoding checks before wall-clock results are reported.

## Evaluation fixed before outcomes

Use 80 original held-out prompts per domain, independent of whether target regeneration succeeded. Evaluate the released ancestor and all final-step pilots on identical traffic at temperatures 0 and 1, with five timing repeats and 128 generated tokens. Measure autoregressive and speculative wall time directly; report domain, equal-mixture and dev-selected oracle traffic. Publish all three seed outcomes and paired prompt uncertainty, with the limited seed count explicit.

The compute criterion is that every run in a seed block lies within ±10% of that block's median measured training-loop GPU-seconds; publish max/min too. If the criterion fails, withdraw the compute-matched claim. Account for regeneration, eligibility, qualification, checkpoint/export and all attempts separately in total study cost.
