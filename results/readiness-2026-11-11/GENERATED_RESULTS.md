# Generated execution evidence

These tables describe the retained source-bound controls. They are not trained-model quality, full-PDF completion or universal losslessness. Published values are nonblocking diagnostics under the user’s 11 November instruction.

| Fixed-budget FP32 method | Pairs | Exact pairs | Mismatches |
|---|---:|---:|---:|
| eagle2 | 320 | 320 | 0 |
| eagle3 | 480 | 479 | 1 |
| hass | 320 | 320 | 0 |

Overall mismatch rate: 1/1120 = 0.089286%. Divergent suffixes remain unequal.

| Original CLI diagnostic | Questions | Turns | Official-counter τ | Published τ | Relative difference |
|---|---:|---:|---:|---:|---:|
| gsm8k | 80 | 80 | 5.515150 | 4.59 | +20.16% |
| mt_bench | 80 | 160 | 4.525161 | 4.24 | +6.73% |

Official-counter and raw-token counter equivalence has not been established. The unmodified CLI uses native multi-turn history and its own RNG procedure.

| Engineering control | Passed ranks | Scope |
|---|---:|---|
| Fresh-process restart | 8/8 | Two successful updates; exact state/RNG/batch order after a mid-epoch restart |
| Native export/inference | 8/8 | Stock inference of the engineering checkpoint; no quality or speed claim |

The CPU vocabulary fixture passed with 32 training/eight dev examples and a 16-token draft vocabulary. It does not qualify the full corpus or the native 32,000-token draft vocabulary.
## Final scoped replay

| Independent raw reconstruction | Population | Maximum event bound |
|---|---:|---:|
| Conditional registered events | 7,864,320 draws / 960 checks | 0.014536679 |
| End-to-end registered events | 61,440 continuations / 36 checks | 0.039414033 |

All 48 retained vectors passed the source-bound checks. Known-site maximum pairwise full-vector delta: 5.36441802979e-05; largest control delta: 8.225440979e-05; frozen ceiling: 0.0001.

The clean reference selects the native AR winner at the known site. The exact kernel cause remains unresolved. The CPU replay and separate evidence review retain their own identities; finite-event checks and a classified point do not establish universal native losslessness.

## Source report SHA256

- [stopping-report.json](stopping-report.json): `dfdafda192ea5462226ac390211a6b57083f1720be201f654595aea5ac0e5b44`
- [stopping-cpu-replay.json](stopping-cpu-replay.json): `3e6761850f2cf848ac59151812f0ba1bd49524117b661cd2eb5d99ed3767ad1f`
- [upstream-cli-report.json](upstream-cli-report.json): `49ef2c545a270c8d05b2871845e40d1410fa62034a6f802fc414f6f20d3bb649`
- [restart-v4-report.json](restart-v4-report.json): `8b30ec44b67e9daada835ba6d4f4f7839d4de219185de0bea6336ac9a707d905`
- [export-v3-report.json](export-v3-report.json): `c85f398f81221a595f884a801e5e0127b53e5390f7a973f457aa6f9c25b4bfc3`
- [vocabulary-engineering.json](vocabulary-engineering.json): `3bf24f19c4d2acab46ac363aac3e63851c9faccaad8e42abc3863ac00f051e50`
- [pipeline-v3-replay.json](pipeline-v3-replay.json): `687ec4ac53f1ba3160db9891e592dce66cb384e69f516551d2e7745be7426295`
- [../../experiments/verification/reviews/independent-raw-replay.json](../../experiments/verification/reviews/independent-raw-replay.json): `0a4faa9ef49b698d2ea9f4260ebe5f8e3d2868663bdcb9a99d0e3680af90d12b`
