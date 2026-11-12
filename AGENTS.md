# Working conventions

Do not add comments to authored code. Do not implement unclear research requirements. Preserve upstream source, lockfiles and pinned artifact identities.

Keep code in `src/`, configurations in `configs/`, experiment specifications in `experiments/`, inputs in `data/`, weights and trained checkpoints in `models/`, and execution artifacts in `outputs/`. Never commit credentials, model weights, datasets, environments, caches or large raw logs. Small reviewed result receipts may live in `results/` with their original hashes and scope.

The default configurations and historical launch helpers describe a validated CSAIL deployment. Read the getting-started guide before GPU work. Use a verified owned allocation for GPU work and Slurm for substantial cluster CPU work. Respect site-specific resource limits and preserve any active holder and continuation. Never infer an allocation from a saved job ID alone.

Keep a running experiment's deployed source, configuration, preparation receipts and hardware mapping frozen. Stage changes separately until that experiment completes. Do not redeploy a publication snapshot into an active run.

Schedule known runnable successors in advance using verified Slurm dependencies or a frozen sequential controller. Prefer full-node high-priority work within the user's allowance, use authorized idle normal-priority capacity when available, and retain correctness and research-admission gates between dependent scientific stages. Do not submit placeholder training to keep GPUs busy.

Distinguish fetched code, installed packages, engineering execution and scientific results. The user's 11 November instruction makes matching paper comparisons nonblocking; retain those historical failures unchanged. Complete source-bound full-model pipeline validation and independent evidence review before scientific training is admitted. The versioned v2 policy preserves correctness, data QA and engineering qualification gates.

Measure wall time directly. Never derive speedup from acceptance length. Generate tables from raw records, retain per-prompt and per-cycle provenance, and report incomplete coverage and failed comparisons honestly. Historical TAPS values and engineering smoke measurements are not new full-panel research results.

Run lightweight source checks with `PYTHONPATH=src python3 -B -Werror -m unittest discover -s tests`. Torch-dependent local skips must be distinguished from recorded CPU and GPU execution evidence.

The legacy native reentrant checkpoint path was shown to disconnect all11 fc/decoder trainables. Never resume the retired generalist or its cancelled continuation. Corrected checkpointing and CPU FP32 RoPE priming require fresh full-body evidence. On the retained deep11 allocation, old CPU coordinator PID2801275 is frozen solely to preserve allocation lifetime: never resume or terminate it. Read outputs/allocation-holders/deep11/current.json and verify live Slurm state before attaching work. Preserve the protected deep10 holder, frozen legacy PID3596188 and holding successor unchanged.
