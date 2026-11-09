# References and input provenance

The research requirements are preserved in the unchanged [TAPS rebuild plan](../TAPS_rebuild_plan.pdf). Its SHA-256 is:

```text
357ba28e2e5e1aa658dae82c279253ac07457d9d406ac50cde0fbf513b150dae
```

The [input manifest](inputs.json) records this identity and the URLs, versions, sizes, page counts and SHA-256 hashes of four external paper PDFs used during preparation. Those downloaded PDFs are omitted from this public source snapshot. Their `references/papers/` paths describe where separately materialized copies belong; they are not bundled files.

## Primary papers and official source

| Work | Pinned paper version | Official repository |
|---|---|---|
| TAPS: Task Aware Proposal Distributions for Speculative Sampling | [arXiv:2603.27027v1](https://arxiv.org/abs/2603.27027v1) | [Moe-Zbeeb/TAPS](https://github.com/Moe-Zbeeb/TAPS) |
| EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test | [arXiv:2503.01840v3](https://arxiv.org/abs/2503.01840v3) | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) |
| Learning Harmonized Representations for Speculative Sampling | [arXiv:2408.15766v3](https://arxiv.org/abs/2408.15766v3) | [HArmonizedSS/HASS](https://github.com/HArmonizedSS/HASS) |
| EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees | [arXiv:2406.16858v1](https://arxiv.org/abs/2406.16858v1) | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) |

The original TAPS paper and repository are historical references. Their reported results are not measurements from this investigation.

## Reproducible source identities

The [source registry](../configs/sources.json) pins the official EAGLE and HASS revisions and benchmark-file hashes. The [asset registry](../configs/assets.json) records target/drafter and tokenizer identities. Fetched upstream source belongs in the excluded `repos/` directory, and weights belong in the excluded `models/` directory. Downloading source or verifying a file does not establish successful execution.

The [reproduction protocol](../experiments/official-reproduction-20261109/PREREGISTRATION.md) defines how published reference cells are compared with new measurements. The full bibliography and proposed research scope remain in the rebuild plan; implementation and execution boundaries are recorded in the [implementation status](../experiments/IMPLEMENTATION_STATUS.md).
