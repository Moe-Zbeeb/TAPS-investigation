# Baseline follow-up registration

This follow-up is registered after observing the original failed baseline panel and three FP32 greedy discrepancies. It is an exploratory hardware replication, not a replacement for the original registration or a training admission.

After job `2214872` completes operationally, use all eight A100 80GB GPUs on deep-chungus-11 for these sequential workloads:

1. Repeat the complete 1,120-pair FP32 greedy control with the unchanged driver, selected inputs, seeds, precision and comparison rule. Include every EAGLE-3, EAGLE-2 and HASS case; select no favourable subset. Write a new validation directory.
2. Repeat the complete EAGLE-2/HASS reference panel with base seed 20261110, matching the independent EAGLE-3 replication seed. Preserve 80 questions, both temperatures, five timing repetitions, excluded warmup and paired target-only measurements. This produces 12,800 measured generations in a separate run directory.

The frozen driver SHA256 is `d6ea2b5216b0052aa68282646b343143c6523ed14e75f8b2b838f99bb6458934`. The prepared case SHA256 is `d3f236a73ab18940bd28f76112210449a2f6ef797061d50a1ab98fc059e0a865`. Reuse the existing frozen replication configuration and deployed source without modification.

Node11 and node10 have different driver versions; retain hardware identities and avoid attributing every difference to precision or model choice. Compare each method against its own co-located target-only measurements. These follow-ups do not amend the original failed report, waive any gate, or authorize training. Their purpose is to test reproducibility of the rare greedy discrepancies and complete an independent baseline panel on the same eight-GPU node.

The successor requests high-priority/spec-priority, account grad-students, one full eight-GPU node, and depends on successful completion of the current node11 job. Together with the protected eight-GPU deep10 holder, it keeps running high-priority use at 16 GPUs. No extra allocation is reserved while the predecessor runs.
