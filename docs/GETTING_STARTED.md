# Getting started

This repository is a source snapshot of an experimental research pipeline. Start with the CPU checks below, then read the protocols before preparing a GPU environment. The Python module remains `taps_rebuild`.

## Clone and inspect

Use Python 3.10 or newer. Run these commands from a source checkout:

```bash
git clone https://github.com/Moe-Zbeeb/TAPS-investigation.git
cd TAPS-investigation
PYTHONPATH=src python3 -B -m taps_rebuild --help
PYTHONPATH=src python3 -B -Werror -m unittest discover -s tests
```

The core command inventory and lightweight source suite can be inspected without installing a CUDA environment. Torch-dependent tests skip when Torch is unavailable. The publication check ran 324 tests with eight such skips on macOS; that result does not establish GPU execution.

Keep the checkout available when working with the pipeline: configurations, protocols and the rebuild PDF are repository inputs. Python package metadata alone does not provision GPU dependencies, model weights, upstream checkouts or preparation receipts.

## Find the relevant command

The CLI dispatches to separate workflows. Inspect an individual command with `PYTHONPATH=src python3 -B -m taps_rebuild <command> --help`.

| Workflow | Commands | Boundary |
|---|---|---|
| Preparation | `prepare-assets`, `prepare-sources`, `prepare-native`, `prepare-serving`, `doctor` | Readiness and provenance are distinct from successful model execution |
| Official evaluation | `reproduce`, `reference-group`, `report`, `verify-sampler` | Smoke results cannot pass the full published-reference gate |
| Data | `data`, `data-gpu` | Authored fixtures exercise integration; they are not research training data |
| Training | `train`, `native-train` | The shared-body prototype and native EAGLE-3 adapter have different data and checkpoint contracts |
| Composition | `compose`, `specialization` | Research comparisons require registered data, matched budgets and eligible evidence |
| Engineering and serving | `gpu-smoke`, `pool`, `serve`, `serving-smoke` | Stock serving support does not implement custom pooled-tree or shared-expert serving |

## Prepare for GPU work

The checked-in profiles describe the validated CSAIL deployment. They contain absolute project/runtime paths, Slurm account and partition choices, and checks against a protected allocation holder. Porting them requires an explicit site configuration and compatible environments. The [runbook](../experiments/RUNBOOK.md) retains the original deployment instructions; it is not a portable installation script.

Before launching a workload:

1. Read the [reproduction protocol](../experiments/official-reproduction-20261109/PREREGISTRATION.md), [source registry](../configs/sources.json) and [asset registry](../configs/assets.json). Verify the pinned source revisions, weights, tokenizer files and benchmark identities.
2. Review the site paths and scheduler assumptions in [the reproduction profile](../configs/reproduction.json) and the experiment launchers. Validate current allocation ownership and capacity rather than treating historical operational records as live state.
3. Prepare separate compatible runtimes for official EAGLE/HASS evaluation, native EAGLE-3 training and stock vLLM serving. Preserve their recorded dependency identities and preparation receipts. The validated deployment uses explicit interpreters with `uv run --no-sync --no-project`.
4. Run substantial preparation, builds and analysis in Slurm; run GPU work only inside a verified allocation. Assign every GPU in the selected full-node allocation. The native-training and serving engineering profiles require eight A100s. Follow the holder-preservation instructions in the original operational documentation.
5. Use fresh run directories and retain frozen inputs and raw records. Resume only when the launcher verifies compatible source, configuration, assets and hardware identities.

The native-runtime preparation performs real CUDA metadata checks inside the GPU allocation. The runtime receipt, successful optimizer execution and checkpoint-recovery evidence are separate checks. See the [engineering report](../experiments/engineering-20261109/RESULTS.md) for what was actually executed.

## Understand the research gates

The complete reference panel has 22 method/benchmark/temperature cells. Every cell requires all 80 questions, all turns, five measured repetitions and paired target-only generations, with warmup excluded. Acceptance must agree with the registered published reference within 5% relative error. The pinned EAGLE panels are distinct from full benchmark datasets such as full GSM8K.

Numeric agreement alone does not establish losslessness. The [verification rule](../experiments/verification/LOSSLESS_RULE.md) states the fixed-tree argument; the [temperature-one audit plan](../experiments/verification/TEMPERATURE_ONE_AUDIT_PLAN.md) describes the remaining full-model sampling check. The finite-vocabulary audits and 48 matched greedy smoke pairs are supporting engineering evidence.

Scientific native training remains gated on reviewed full reproduction and full-model temperature-one validation. The next planned pilot has chat, math and code, a generalist plus three specialists, two response policies and three seeds. Actual datasets, held-out splits and equal token/compute budgets must be registered before those 24 core runs. The [implementation status](../experiments/IMPLEMENTATION_STATUS.md) lists the remaining research work.

## Locate inputs and evidence

| Location | Contents | Included in this snapshot |
|---|---|---|
| `src/`, `configs/`, `tests/` | Implementation, profiles and source checks | Yes |
| `experiments/` | Protocols, engineering summaries and historical operational context | Yes |
| `results/` | Selected machine-readable engineering receipts and provenance | Yes |
| `references/` | Canonical links and source/hash manifest | Yes; external paper PDFs are materialized separately |
| `data/`, `models/`, `repos/` | Datasets, weights and fetched upstream source | No; external artifacts |
| `outputs/` | Raw logs, frozen run inputs, preparation receipts and generated reports | No; external artifacts |

Paths to excluded artifacts in the experiment documents describe the original evidence layout. They are not public download links, and their presence in a document does not establish that an artifact is available in a fresh clone. Begin with the retained [evidence index](../results/README.md) and [engineering summary](../experiments/engineering-20261109/RESULTS.md), then inspect the corresponding raw records when those artifacts are supplied.
