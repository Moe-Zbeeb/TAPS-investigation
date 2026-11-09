# Batch PATH recovery

The active reference deployment is commit `a9855371b638fdd13a1f8125a0041759958aefbf`, source-identity SHA256 `6375933ffe6e688eb4eb032360eabe519efadd01e13233a56869a022bf6f2caa`. Its tracked files remain unchanged while the reference jobs run.

Initial EAGLE-3 job 2212926 failed in one second at the first `scontrol` lookup. It created no run manifest, worker output or model execution. Its batch PATH was `/usr/bin:/bin`; the verified compute-node Slurm directory is `/usr/local/bin`.

The independently reviewed `operational_retry_v2.py` reconstructs the original payload and allows only a fresh v2 run/log ID and PATH `/usr/local/bin:/usr/bin:/bin`. It binds the predecessor receipt and logs, verifies terminal accounting and absence of prior scientific output, rechecks the unchanged source, holder, continuation, full idle node and GPU allowance, and writes durable submission evidence. It never cancels or automatically retries a job. The explicit retry was accepted as job 2212933.

The packet and exact argument differences are in `outputs/resources/eagle3-v2-operational-submission-20261109.json`; the actual new receipt is `outputs/submissions/official-eagle3-20261109-v2.json`. These dated operational artifacts document this attempt and are not a general resubmission command.

The local follow-up snapshot repairs PATH in the regular registered submission helper and its launch configuration, with a regression check. Its launch-plan SHA256 is `a9d8d2a2f78daaa9ada7eb9a9c2bb2645dd0382456f4163dd284f9dd4628761d`. This follow-up is staged locally and must be deployed after the active reference group finishes. Do not bulk-copy its changed tracked files over the frozen active project: compatible requeue validates the original deployment. The general `taps reproduce` command uses the inherited site environment and was unaffected.

Use RUN_GROUP.md for the actual two-run aggregation command. Model, source, prompt, sampling, seed, node-budget and repetition settings did not change in this recovery.
