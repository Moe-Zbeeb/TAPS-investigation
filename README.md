# TAPS investigation

Research code for studying specialist drafters under mixed traffic, following the [TAPS rebuild plan](TAPS_rebuild_plan.pdf). The project connects official-baseline reproduction, controlled specialization experiments, fixed-budget candidate pooling, shared-body experts and serving evaluation. The Python module is `taps_rebuild`.

[Getting started](docs/GETTING_STARTED.md) · [Reproduction protocol](experiments/official-reproduction-20261109/PREREGISTRATION.md) · [Implementation status](experiments/IMPLEMENTATION_STATUS.md) · [Current completion plan](experiments/finalization-20261111/COMPLETION_PLAN.md) · [References](references/README.md)

The [13 November status](experiments/STATUS_20261113.md) supersedes earlier live snapshots. The corrected generalist is training on eight A100 80GB GPUs with a committed 7,000-update checkpoint. Deep10's allocation reached its deadline; its protected high-priority successor remains queued. The pilot's pristine-teacher-buffer fix passed its fresh 208-update failure-case test and all 12 GPU qualification phases; all 14 independent replay checks and actual evidence review passed. The automatic handoff waits for deep10 before final integration and training. No fixed-budget pilot or forty-epoch generalist has completed.

```mermaid
flowchart LR
    inputs["Pinned inputs"] --> frozen["Frozen run"]
    frozen --> workers["AR / speculative workers"]
    workers --> records["Cycle records"]
    records --> reports["Paired reports"]
```

## Evidence and current scope

This snapshot contains research infrastructure, full baseline measurements and recorded engineering evidence. The full **22-cell released-baseline panel completed 22,400 generations; its reproduction gate failed**. See the [historical full results](results/baselines-2026-11-10/README.md) and [current execution status](experiments/STATUS_20261113.md). The user's 11 November instruction makes matching published paper numbers nonblocking. The [versioned pipeline policy](experiments/verification/PIPELINE_VALIDATION_V2.md) retains correctness, independent evidence review, data QA and source-matched engineering gates. Scientific generalist/specialist results and custom pooled-tree serving remain outstanding.

Engineering records dated 9 November 2026 document checks on eight A100 80GB GPUs:

| Check | Recorded outcome | Scope |
|---|---|---|
| Official EAGLE-3, EAGLE-2 and HASS smoke | 48/48 paired temperature-zero completions matched target-only tokens | Eight MT-Bench questions, two turns, short output cap |
| Native EAGLE-3 training | Two optimizer updates on every rank; model and optimizer hashes recovered after explicit checkpoint perturbations | Authored engineering data; same-engine recovery |
| Stock vLLM serving | 344 completed HTTP requests across eight workers | Startup and concurrent-request integration |

The [evidence index](results/README.md) contains selected machine-readable receipts, and the [engineering report](experiments/engineering-20261109/RESULTS.md) explains their limits. Large raw logs, preparation artifacts, weights and datasets are excluded from this Git snapshot. Paths to these artifacts in experiment documents identify external records. These small checks do not establish drafter quality, published-number reproduction or production speedup.

The initial publication source check ran 368 tests with nine Torch-dependent tests skipped on macOS. CPU source checks and CUDA execution are separate evidence. CI runs the CPU suite on Python 3.10 and 3.12.

The [historical replication check](results/status-2026-11-11/README.md) found 20/22 acceptance values within 5%, while 9/22 strict cells passed. Both FP32 controls retained three mismatches among 1,120 pairs. Two are explained by context-stop lengths; the [eight-GPU logit diagnostic](results/fp32-decision-2026-11-11/README.md) captures the remaining decision as a same-prefix near-tie argmax flip, with seven matched controls. The [seed-policy experiment](results/seed-policy-2026-11-11/README.md) found that upstream seed zero does not resolve the GSM8K outlier. These historical failures remain recorded. The subsequent [pipeline replay](results/readiness-2026-11-11/pipeline-v3-replay.json), source-matched restart/export checks and independent review support the separate v2 training admission; matching published numbers is nonblocking.

ShareGPT and UltraChat [source preparation](experiments/corpus-20261110/README.md) contains 540,107 grouped conversations, including 284,115 with complete original answers. Chat and math/code target generation completed. Earlier native profiles and the stopped scratch run used a defective decoder-gradient path and cannot qualify corrected full-body training. The paired pilot cache and eligibility scan completed; its new budget, objective and recovery checks remain mandatory.

## Start on CPU

From the repository root, using Python 3.10 or newer:

```bash
PYTHONPATH=src python3 -B -m taps_rebuild --help
PYTHONPATH=src python3 -B -Werror -m unittest discover -s tests
```

Torch-dependent tests skip when Torch is unavailable. See [getting started](docs/GETTING_STARTED.md) for the command groups, environment boundaries and GPU prerequisites. The supplied GPU profiles were validated at CSAIL and contain site-specific paths, Slurm settings and allocation-holder checks; a fresh clone does not provision those environments.

## Research plan

The unchanged rebuild memo defines the following requirements. This is a map of the programme, rather than a list of completed stages.

| Plan section | Research requirement |
|---|---|
| 5.1 · Pipeline and reproduction | Reproduce released EAGLE-3, HASS and EAGLE-2, then reproduce drafter training on ShareGPT/UltraChat |
| 5.2 · Specialization, RQ1 | Separate prompt domain, response style, data scale and capacity under matched tokens and compute |
| 5.3 · Pooled budget tree, RQ2 | Merge duplicate candidate paths, calibrate across drafters and retain a fixed verification budget |
| 5.4 · Shared-body experts, RQ3 | Compare domain experts and weight composition from a shared generalist initialization |
| 5.5 · Verification | Check greedy token equivalence and full-model sampling distributions under an explicit verification rule |
| 5.6 · Serving, RQ4 | Measure latency, throughput and memory under continuous batching and mixed traffic |
| 5.7 · Baselines | Compare with matched-compute generalists, selectors and other independently implemented baselines |
| 5.8 · Statistics and reporting | Use three training seeds, paired comparisons, prompt-level intervals and repeated timings without warmup |

The [registered core pilot](experiments/rq1-pilot-24-20261112/README.md) has one union generalist and three specialists, original and target-generated response contexts, and three seeds: **24 drafter runs**. All fine-tune the same released EAGLE3 ancestor. The separate scratch generalist requires a corrected restart. The fixed-budget plans and guarded Slurm chain are prepared; their source-matched qualification and independent admission precede training. The measured specialization gap will guide composition experiments. See the [roadmap](experiments/ROADMAP.md) and [pilot analysis design](experiments/rq1/SPECIALIZATION_ANALYSIS.md).

## Repository layout

```text
TAPS_rebuild_plan.pdf    Original research requirements
src/taps_rebuild/        Preparation, execution, training and analysis tools
configs/                Pinned registries and experiment profiles
tests/                  Source, protocol and integration checks
experiments/            Protocols, engineering summaries and operational records
results/                Selected engineering receipts and their provenance
docs/                   Public getting-started guide
references/             Paper links and input provenance
```

Materialized inputs belong in `data/`, weights in `models/`, fetched source in `repos/` and run artifacts in `outputs/`; these directories are excluded from Git. The [reference manifest](references/inputs.json) preserves the rebuild-plan hash and external-paper identities.

## Measurement discipline

Published and historical values are reference targets, never new measurements. Reports are generated from per-prompt and per-cycle records with source, checkpoint and configuration identities. Acceptance length is reported separately from measured wall-clock speedup. Missing cases and unexplained greedy mismatches prevent the reproduction gate from passing.

The [reproduction protocol](experiments/official-reproduction-20261109/PREREGISTRATION.md) defines the reference panel, 5% acceptance tolerance and reduction rules. The [verification argument](experiments/verification/LOSSLESS_RULE.md) and [temperature-one audit plan](experiments/verification/TEMPERATURE_ONE_AUDIT_PLAN.md) distinguish mathematical guarantees, finite-vocabulary checks and the remaining full-model validation. The [runbook](experiments/RUNBOOK.md) preserves the original site's operational procedure.
