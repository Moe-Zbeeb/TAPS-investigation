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

## Source report SHA256

- [stopping-report.json](stopping-report.json): `dfdafda192ea5462226ac390211a6b57083f1720be201f654595aea5ac0e5b44`
- [stopping-cpu-replay.json](stopping-cpu-replay.json): `3e6761850f2cf848ac59151812f0ba1bd49524117b661cd2eb5d99ed3767ad1f`
- [upstream-cli-report.json](upstream-cli-report.json): `49ef2c545a270c8d05b2871845e40d1410fa62034a6f802fc414f6f20d3bb649`
- [restart-report.json](restart-report.json): `e3f383f12a031d9712e577ec40be726ac8e000168d23e71a3235649a347c1e69`
- [export-report.json](export-report.json): `8e366d23eec47380d081584dcf22ebca77bf62f445ba851dc104e808e07aedec`
- [vocabulary-engineering.json](vocabulary-engineering.json): `3bf24f19c4d2acab46ac363aac3e63851c9faccaad8e42abc3863ac00f051e50`
