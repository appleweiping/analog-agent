# Top-K Planner Baseline

`top_k_baseline` is the simplest planner-side research baseline in `analog-agent`.

It keeps the shared experiment pipeline, real `ngspice` verification path, and unified stats/export contracts, but removes the sharper planner mechanisms that distinguish the full system:

- no fidelity escalation
- no phase-aware updates
- no calibration-driven replanning
- no rollout guidance

Instead, it performs a straightforward ranked `top-k` simulation policy over the currently evaluated frontier and sends all selected candidates through `quick_truth`.

This baseline exists to answer one paper-facing question directly:

`Is the full planner better than a simple top-k simulator-selection policy under the same design task and the same real-SPICE backend?`

Because it is implemented as an `ExperimentMode`, not as a standalone script, it participates in:

- the unified benchmark runner
- the methodology comparison pipeline
- the paper evidence export pipeline
- the shared stats/export contracts
