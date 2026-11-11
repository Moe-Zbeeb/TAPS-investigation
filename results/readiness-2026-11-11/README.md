# Training-readiness evidence · 11 November 2026

The [execution status](../../experiments/STATUS_20261111.md) maps the prepared queue to the PDF. [`scheduler-chain.json`](scheduler-chain.json) preserves the submission records, later pending-job replacements, live queue and unchanged active source identity. A queued GPU job is not an active GPU allocation or a passing scientific gate.

The [generated results](GENERATED_RESULTS.md) reconstruct the common-budget comparisons, unmodified CLI diagnostics, source-matched restart/export qualifications and final raw replay. Rebuild them with `python3 experiments/finalization-20261111/render_readiness.py --directory results/readiness-2026-11-11`. The [final pipeline replay](pipeline-v3-replay.json) has six passing checks and 468 content-bound artifacts. Its separate independent evidence review determines admission under the [v2 policy](../../experiments/verification/PIPELINE_VALIDATION_V2.md). The historical paper comparisons remain unchanged and nonblocking.

## Released-weight provenance

[`released-weight-history.json`](released-weight-history.json) records public Hugging Face commit/tree responses, their SHA256 values and the exact URLs fetched on 11 November. The project's pinned EAGLE-3 checkpoint at revision `ada412b672e293d682423de84a095447bf38a637` uses `pytorch_model.bin` SHA256 `16d5bf9524108d59f68c7af196edd69018c29316cb6a2fb491cfd2a393ee1cf4`, 849,768,795 bytes. That is the same file identity in the [first public weight release](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B/tree/607d0d5b7871cd4b89395b6af288c070cfa0a168) and the [currently pinned release](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B/tree/ada412b672e293d682423de84a095447bf38a637). The native `config.json` Git object identity also matches across the checked revisions.

This is a metadata/content-identity audit. It did not download another checkpoint, train a model, resolve the published acceptance outliers or establish the paper's unpublished experiment artifacts. It rules out the hypothesis that selecting that first public `pytorch_model.bin` would change the weights used by the current run.

## Evidence boundaries

Fixed-budget validation, actual process-restart qualification, native-export inference and direct upstream CLI execution have separate registrations and raw output directories. Reports are added only after their jobs execute. Existing failed panels, the source-bound scientific gate and the independent-review requirement are retained.
