| Benchmark | RNG policy | Question-mean τ | Published reference | Relative difference |
|---|---|---:|---:|---:|
| alpaca | upstream_seed_zero | 5.336986 | 5.56 | -4.01% |
| alpaca | registered_hash | 5.595749 | 5.56 | +0.64% |
| gsm8k | upstream_seed_zero | 5.396826 | 4.59 | +17.58% |
| gsm8k | registered_hash | 5.514285 | 4.59 | +20.14% |

| Benchmark | Seed-zero minus hashed τ | Paired question bootstrap 95% interval |
|---|---:|---:|
| alpaca | -0.258763 | [-0.457659, -0.073186] |
| gsm8k | -0.117459 | [-0.281659, +0.038188] |

Generated from report SHA256 `686dfe7f3fdd6cdfc1ccfe9931f9ef1ddf1164b3e9c5dba758ea7aca3bf9cbaf`.
