# Validation queue · 10 November 2026

This registration preserves the failed original baseline panel. It authorizes validation measurements only and cannot admit scientific training. The frozen conditional and end-to-end definitions in `experiments/verification/TEMPERATURE_ONE_AUDIT_PLAN.md` remain controlling. No sample budget, prompt, margin or event is selected from the outcomes of these jobs.

## Full-panel greedy numerical control

Use all original repeat-zero paired prefixes: 480 EAGLE-3, 320 EAGLE-2 and 320 HASS question-turn pairs. Load the original released weights in FP32, retaining the original source, native generators, 60-node tree, depth, width, 512-token requested cap, 2048 cache argument, seeds and tokenized input. Compare the two FP32 outputs through the same termination/cap rule. MT-Bench turn-two inputs retain the original FP16 speculative history: this tests conditional equivalence on the frozen panel, not a newly generated FP32 conversation distribution. Report all mismatches and actual timings; changed precision and one repetition cannot replace the original published-number benchmark or its five-repeat timing protocol.

## Temperature-one controls

Implement proposal A with the native FP16 posterior, all 60 registered captures, 65,536 draws per sampler, eight registered events, familywise alpha 0.025 and event margin 0.02. Preserve the full-vocabulary histogram and independent target reference forward. A missing third cycle remains a coverage failure.

Implement proposal B with ten registered prompts per method, 1,024 continuations per prompt and arm, a 32-token comparison cap and the twelve registered events. Use independent mode-specific seed streams. The equal-weight prompt mixture, 36 simultaneous comparisons, familywise alpha 0.025 and 0.06 margin are unchanged. This screen cannot establish full-vocabulary or long-sequence distributional equivalence.

## Scheduling and admission

Deep10 retains its protected allocation and successor. A detached full-node step runs greedy validation followed by conditional sampling and report generation. A second full A100 80GB node runs the independent end-to-end screen. Deep-h-1 may run a separately labeled hardware replication after all six Slurm-configured GPUs pass the bounded CUDA check; it does not silently replace the registered A100 results. Keep total high-priority-partition usage at or below 18 GPUs. Additional jobs use verified normal/background settings only when appropriate capacity exists.

Freeze the driver, registration, selected inputs, source/model identities and hardware inventory before draws. Worker failures stop their controller. Statistical failures are preserved in summaries and do not authorize training. Chaining independent validation stages after an operationally complete phase does not waive any scientific gate. No scientific training job is submitted by this driver. Existing original source and run directories remain unchanged.
