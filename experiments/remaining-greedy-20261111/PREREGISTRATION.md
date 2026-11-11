# Remaining FP32 token disagreement · 11 November 2026

Status: version 1 passed real-artifact CPU preparation; job 2215881 stopped before model loading because existing GPU processes were observed. Version 2 added a bounded scheduler-cleanup wait. Version 3 handles the specifically verified stale driver context below. The selected cases, precision, seeds and probes are unchanged. See the adjacent execution README for the latest submission and outcome. This diagnostic advances rebuild-plan §§5.1 and 5.5 without admitting training or changing any existing result.

The two complete FP32 panels retained identical failures: MT-Bench question 146, turn 1 differs at output offset 200, while CNN/DM questions 295 and 288 have identical shared prefixes and unequal context-stop lengths. The [prior CPU audit](../../results/status-2026-11-11/README.md) already records the two stopping cases. This experiment investigates the remaining token decision.

Select the registered MT-Bench failure from the source-bound primary FP32 panel. Add seven distinct FP32-matched controls, each with at least 37 comparison tokens. Sort eligible controls by the canonical SHA256 of their full registered case, then select round-robin over Alpaca, CNN/DM, GSM8K, HumanEval and MT-Bench until seven are selected. This is descriptive case selection, not a prevalence estimate or a hypothesis-test population.

Use the original target, released EAGLE-3 drafter, source, tree budget, depth, width, seed and input tokens. Load FP32 weights. The failed case probes offset 200; controls probe offset 32. The requested generation cap is `min(original_cap, probe_offset + 4)`, with the original 2048-token cache setting. Shortening the requested output must not be presented as a full-panel stopping-policy correction.

Reuse the existing observational worker from `experiments/greedy-audit-20261110/greedy_audit.py`, copied byte-for-byte as `audit_worker.py` into a frozen new preparation bundle. It runs each mode twice, observes the unchanged native decision and captures the top-five target logits, top-two margin, original token-pair logit difference and AR-versus-tree logit discrepancy. Record exact prefix identity and whether both replays reproduce the original FP32 prefix through the probe. An invalid/missing prefix capture is inconclusive, never evidence of an arithmetic cause.

Run one task on each of eight GPUs of an owned full A100 80GB node. Preserve physical device mapping, allocation ownership, driver/runtime, source, registration, cases and worker receipt hashes. Check that the node has no existing active GPU workload before starting these workers. After scheduler preemption, allow at most 300 seconds for existing processes to disappear, polling read-only every five seconds and recording each observation. The observed scheduler uses REQUEUE, a 30-second KillWait and 60-second unkillable-step timeout; the high-priority partition has a 120-second grace period, while background has zero. Never signal another workload or load models while any active GPU process remains. Leave deep10 generation, its holder and successor alone. Request only a concrete full-node diagnostic within the 18-GPU high-priority allowance; use authorized suitable idle normal-priority capacity when appropriate. Never launch from a historical job ID without live verification.

Inside the owned node11 allocation, the driver reported PID 2107512 with name `[Not Found]` and 10,034 MiB on GPU UUID `GPU-8700e8c0-2ab7-4c31-3e0a-bcfc35597de8`. The PID did not exist in host `/proc`; the proc mount had no `hidepid` restriction, the process namespace was not nested, PID 1 was systemd, and processes of multiple other UIDs were visible. That GPU still reported 71,113 MiB free. This is a stale driver allocation, not evidence of a currently running process.

Version 3 binds the [site-specific context policy](../../configs/remaining-greedy-node11-context-20261111.json) into preparation. It permits only this exact UUID/PID/name entry, at no more than 11,000 MiB, and only while the PID remains absent and host process visibility is reverified. Every other entry, including a recycled PID that now exists, blocks startup. Require at least 51,200 MiB free on every owned GPU before loading models. Retain the full context/memory observation and policy hash. No GPU reset, process signal, freed-memory claim or clean timing benchmark is permitted by this exception. The remaining-token case is assigned to logical GPU zero; the affected GPU seven runs a matched control.

Preparation must verify raw-worker hashes, complete unique 480-case EAGLE-3 coverage, the original three failures, registered prepared-case identity, original validation-driver identity and original run provenance. Existing raw data and deployed code are read-only. Any failed identity check stops the diagnostic.

Report all eight tasks, missing captures and replay instability. A small logit margin with a sign change is a numerical observation; it is not by itself proof of the underlying cause or permission to waive a gate. If needed, follow with separately registered clean full-prefix AR forwards or KV-state tests. No sampling margins, published acceptance reductions, original gates or training-admission rules change here.

## Submission after reconnecting

Copy `driver.py`, this registration, the exact old observational worker and the site policy (as `stale-context.json`) into a fresh `outputs/preparation/remaining-greedy-20261111-v3` bundle. Verify the copied hashes, live owned high-priority GPU usage, node11's configured GPU count and 80GB compatibility, then submit the full-node batch script. If a compatible idle normal-priority node is available, record the scheduling override; do not mutate frozen runtime inputs. All eight tasks are FP32 and require the tested 80GB class.

The bundled `run.sh` performs CPU-only preparation inside the new Slurm job, then starts eight GPU workers. It deliberately contains no `sbatch` call and allocates nothing on its own. The diagnostic is not queued until an actual submitted job ID and its effective Slurm fields have been verified. A complete diagnostic report is still not training admission.

Proposed high-priority submission, only after the live checks above:

```bash
sbatch --parsable --account=grad-students --partition=high-priority --qos=spec-priority \
  --nodes=1 --exclusive --nodelist=deep-chungus-11 --gres=gpu:a100:8 \
  --ntasks=1 --cpus-per-task=64 --mem=500G --time=02:00:00 --no-requeue \
  --job-name=taps-fp32-decision \
  --chdir=/mnt/nfs/home/mohamadzbib/projects/taps-rebuild \
  --output=/mnt/nfs/home/mohamadzbib/projects/taps-rebuild/outputs/launchers/fp32-decision-%j.log \
  --wrap='/bin/bash /mnt/nfs/home/mohamadzbib/projects/taps-rebuild/outputs/preparation/remaining-greedy-20261111-v3/run.sh'
```

Node11 is proposed because the previous replica verified its eight 80GB A100s. This does not establish that it is currently available. Do not submit if the new live topology/ownership/quota check conflicts with this command, and do not attach this workload to busy deep10.

Local source validation also covers cleanup becoming idle and a timeout that never permits model loading. Selection tests reject duplicate/missing panel cases, altered inputs, shifted/additional failures and missing paired modes, and verify deterministic control selection across all five benchmarks. Shell syntax and CLI checks are separate from GPU execution. See the execution README for the final test count and real outcomes.
