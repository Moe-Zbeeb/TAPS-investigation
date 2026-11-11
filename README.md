# TAPS investigation

Research code for studying specialist drafters under mixed traffic, following the [TAPS rebuild plan](TAPS_rebuild_plan.pdf). The project connects official-baseline reproduction, controlled specialization experiments, fixed-budget candidate pooling, shared-body experts and serving evaluation. The Python module is `taps_rebuild`.

[Getting started](docs/GETTING_STARTED.md) · [Reproduction protocol](experiments/official-reproduction-20261109/PREREGISTRATION.md) · [Implementation status](experiments/IMPLEMENTATION_STATUS.md) · [Current completion plan](experiments/finalization-20261111/COMPLETION_PLAN.md) · [References](references/README.md)

```mermaid
flowchart LR
    inputs["Pinned inputs"] --> frozen["Frozen run"]
    frozen --> workers["AR / speculative workers"]
    workers --> records["Cycle records"]
    records --> reports["Paired reports"]
```

## Evidence and current scope

This snapshot contains research infrastructure, full baseline measurements and recorded engineering evidence. The full **22-cell released-baseline panel completed 22,400 generations; its reproduction gate failed**. Acceptance length met tolerance in 19/22 cells, but greedy token disagreements and three sampling outliers require investigation. See the [full results](results/baselines-2026-11-10/README.md). A [precision control](results/greedy-audit-2026-11-10/README.md) reproduced 24 selected FP16 failures and observed FP32 agreement at each disputed token. Scientific generalist/specialist training, independent validator review and custom pooled-tree serving remain outstanding.

Engineering records dated 9 November 2026 document checks on eight A100 80GB GPUs:

| Check | Recorded outcome | Scope |
|---|---|---|
| Official EAGLE-3, EAGLE-2 and HASS smoke | 48/48 paired temperature-zero completions matched target-only tokens | Eight MT-Bench questions, two turns, short output cap |
| Native EAGLE-3 training | Two optimizer updates on every rank; model and optimizer hashes recovered after explicit checkpoint perturbations | Authored engineering data; same-engine recovery |
| Stock vLLM serving | 344 completed HTTP requests across eight workers | Startup and concurrent-request integration |

The [evidence index](results/README.md) contains selected machine-readable receipts, and the [engineering report](experiments/engineering-20261109/RESULTS.md) explains their limits. Large raw logs, preparation artifacts, weights and datasets are excluded from this Git snapshot. Paths to these artifacts in experiment documents identify external records. These small checks do not establish drafter quality, published-number reproduction or production speedup.

The publication source check ran 333 tests with nine Torch-dependent tests skipped on macOS. CPU source checks and CUDA execution are separate evidence. CI runs the CPU suite on Python 3.10 and 3.12.

The [11 November check](results/status-2026-11-11/README.md) confirms both independent replication jobs completed: 20/22 acceptance values were within 5%, while only 9/22 strict cells passed. Both full FP32 controls retained the same three mismatches among 1,120 pairs; two are explained by context-stop lengths and one token disagreement remains unresolved. The conditional audit and 61,440-continuation screen passed their registered checks. Scientific training remains blocked by reproduction and validator review. Eight A100 80GB GPUs are generating target responses; 133,120/540,107 source inputs were processed at the latest audit. The [completion plan](experiments/finalization-20261111/COMPLETION_PLAN.md) maps the remaining work to every section of the memo.

ShareGPT and UltraChat [source preparation](experiments/corpus-20261110/README.md) now contains 540,107 grouped conversations, including 284,115 with complete original answers. Target-response generation and a dependent eight-GPU native training profile have separate frozen launchers. Training remains blocked by the failed scientific checks and required validator review.

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

After reviewed reproduction and full-model temperature-one validation, the next scientific experiment is a single-target chat/math/code pilot: one generalist and three specialists, target-generated and original responses, and three seeds, for **24 core drafter runs**. Data sources, held-out splits and matched token/compute budgets must be registered first. The measured specialization gap determines which composition experiments follow. See the [roadmap](experiments/ROADMAP.md) and [pilot analysis design](experiments/rq1/SPECIALIZATION_ANALYSIS.md).

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
