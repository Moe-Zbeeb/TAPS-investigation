# Running the rebuild

The cluster project is `/mnt/nfs/home/mohamadzbib/projects/taps-rebuild`. Run GPU work only inside an owned full-node allocation. The default reproduction launcher attaches to the verified current deep10 holder; it does not replace or release that holder.

## Check preparation and results

```bash
cd /mnt/nfs/home/mohamadzbib/projects/taps-rebuild
source experiments/environment.sh eagle
taps doctor --root "$TAPS_PROJECT_ROOT"
```

`doctor` checks recorded readiness. A completed GPU receipt, rather than an installed package or available model file, establishes successful execution. The engineering evidence and its limits are recorded in `experiments/engineering-20261109/RESULTS.md`.

## Reproduce the official reference panel

```bash
taps reproduce --root "$TAPS_PROJECT_ROOT" --config configs/reproduction.json --run-id official-reference-my-run --expected-gpus 8 --detach
```

This runs all 22 registered cells with 80 questions, both temperatures and five measured repetitions. Each speculative completion has a paired target-only measurement. The run directory is `outputs/runs/official-reference-my-run/`; `pool.json` records progress, `raw/` contains completed cases and `report.json` is produced after complete coverage. The launcher prints its persistent receipt and log paths.

For a short engineering check, add `--smoke` and choose a different run ID. It cannot pass the full reference gate. To continue an interrupted compatible run, repeat the original command with `--resume`. Change the run ID when changing inputs. The launcher verifies frozen identities and refuses conflicting writers. A step is capped at 12 hours and at the current holder's remaining lifetime minus a safety buffer.

The source configuration fixes the context policy. Before generation, every prompt is checked against the real tree/cache budget. Long prompts can reduce the output cap, and those limits are recorded. Prompts are not silently truncated.

## Data and training

```bash
taps data-gpu --help
taps native-train --help
taps train --help
taps compose --help
taps specialization --help
```

`data-gpu run` distributes authored engineering examples across all eight GPUs for target regeneration and aligned feature export. Its completed cache contains a merged manifest accepted by the reference trainer. `train` is the shared-body prototype. `native-train` runs the separately pinned official EAGLE-3 architecture and recipe adapter; it computes target features online and uses a different data contract. A prototype checkpoint is not a native EAGLE-3 checkpoint.

Use fresh fixture, plan and run paths for engineering training. CPU preparation and substantive analysis also run in Slurm, with CUDA hidden. The engineering fixtures are authored test data and cannot be relabelled as research datasets. Scientific training requires complete baseline and source-bound full-model validation, including the independent review required by the controlling proposal. The current gate remains closed. The real ShareGPT/UltraChat preparation and dependent training profile are documented in [the corpus runbook](corpus-20261110/README.md).

For a new native engineering check, run this preparation inside a CPU Slurm step with CUDA hidden:

```bash
export ROOT=/mnt/nfs/home/mohamadzbib/projects/taps-rebuild
export NATIVE_RUN=native-my-fresh-check
export RUN="$ROOT/outputs/native-training/$NATIVE_RUN"
export NATIVE_PYTHON="$(python3 -c 'from pathlib import Path; import os; print(Path(os.environ["ROOT"]).joinpath("envs/native-eagle3").resolve() / "bin/python")')"
export CUDA_VISIBLE_DEVICES=''
/usr/bin/python3 -B "$ROOT/src/taps_rebuild/native_training.py" fixture --root "$ROOT" --output-dir "$ROOT/data/engineering/$NATIVE_RUN" --train-rows 32 --dev-rows 8
/usr/bin/python3 -B "$ROOT/src/taps_rebuild/native_training.py" prepare --root "$ROOT" --config "$ROOT/configs/native-eagle3-training.json" --train "$ROOT/data/engineering/$NATIVE_RUN/train.jsonl" --validation "$ROOT/data/engineering/$NATIVE_RUN/dev.jsonl" --run-dir "$RUN"
"$NATIVE_PYTHON" -B "$RUN/inputs/native_training.py" audit --plan "$RUN/plan.json"
```

Then, inside the verified eight-A100 allocation with all GPUs free of other workloads:

```bash
env -u CUDA_VISIBLE_DEVICES MAX_JOBS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 "$NATIVE_PYTHON" -B "$RUN/inputs/native_training.py" launch --plan "$RUN/plan.json"
```

The launcher verifies the complete allocation, starts eight ranks and stops after two successful optimizer updates. It tests same-engine model/optimizer recovery by saving, perturbing one trainable weight and one owned optimizer moment, then requiring both complete hashes to return exactly after loading. This does not establish restart from a mid-epoch dataset position. Run one GPU workflow per node at a time; different run IDs do not provide separate GPUs.

## Serving

```bash
taps serve --help
taps serving-smoke --help
```

The serving smoke starts four target-only and four original EAGLE-3 stock vLLM servers, one per GPU, then sends real warmup and measured concurrent requests. Its receipt covers model startup, actual usage and latency, and cleanup of its own processes. It is not a custom pooled-tree serving implementation or an eligible production speedup benchmark.

The prepared native and serving environments are separate from the preserved official evaluation environments. Use their recorded launchers without dependency synchronization. Weights and trained checkpoints belong under `models/`; raw results belong under `outputs/`. The Hugging Face collection is the eventual publication destination; these commands do not upload artifacts.

The next experiment order and scaling decisions are in `experiments/FAST_RESULTS_STRATEGY.md`. The remaining research implementation is listed in `experiments/IMPLEMENTATION_STATUS.md`.
