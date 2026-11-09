# Full reference execution group

The accelerated panel uses disjoint methods on complete nodes. The original `configs/reproduction.json`, its 22 references, all seeds, five timing repetitions, prompt files and model/source identities remain common. A run's strict collector must establish complete paired coverage before the group can be reported.

| Run ID | Method | Intended node | GPUs | Priority |
|---|---|---|---:|---|
| official-eagle3-20261109-v2 | EAGLE-3, 10 cells | deep-chungus-3 | 9 | high-priority / spec-priority |
| official-eagle2-hass-20261109-v1 | EAGLE-2 and HASS, 12 cells in separate waves | preserved deep-chungus-10 holder | 8 | high-priority / spec-priority |

Both runs were admitted on 9 November: EAGLE-3 batch job 2212933 and the EAGLE-2/HASS step within holder 2212576. They allocate 17 high-priority GPUs, within the 18-GPU allowance. Deep11 had become partly occupied, so its prepared normal-priority HASS template was not launched alongside the combined run. Live Slurm state remains authoritative.

The initial EAGLE-3 job 2212926 failed before GPU inventory or model loading because its batch PATH omitted `/usr/local/bin`, where the compute-node Slurm commands live. The v2 operational retry changes only that PATH and the run/log names; it preserves the exact frozen scientific inputs and both failed and successful submission receipts. See PATH_RECOVERY.md. The active cluster deployment stays frozen at commit a9855371b638fdd13a1f8125a0041759958aefbf.

The held-node waves were launched with:

```bash
taps reproduce --root "$TAPS_PROJECT_ROOT" --config configs/reproduction.json --run-id official-eagle2-hass-20261109-v1 --expected-gpus 8 --methods eagle2,hass --detach
```

Each additional batch job pins its node, owns the full node, requests 12 hours and permits requeue. A requeued job can reuse only compatible frozen inputs and the same GPU inventory/topology. Committed raw cases are retained; incomplete tails are archived before continuation. Normal-priority work can be preempted immediately. The deep10 holder and its pending continuation are preserved.

Shard count changes assignment rather than scientific seeds. Paired target-only and speculative measurements for a cell stay on the same GPU. Hardware, driver and priority are reported separately per method; raw throughput across different nodes is not a controlled cross-method ranking. If that ranking becomes a central result, run a same-node crossover.

After every run completes, execute the source-bound group collector in a CPU Slurm step:

```bash
taps reference-group --root "$TAPS_PROJECT_ROOT" --run-id official-eagle3-20261109-v2 --run-id official-eagle2-hass-20261109-v1 --output-dir outputs/reference-groups/official-20261109-v1
```

The group collector requires the exact disjoint union of all 22 configured cells, verifies the completed constituent runs and regenerates the aggregate table from their raw cases. A complete run that disagrees with a published number produces a failed numerical gate; it is not silently discarded or relabelled. The full-model temperature-one correctness check remains separate and cannot be replaced by this reproduction table.
