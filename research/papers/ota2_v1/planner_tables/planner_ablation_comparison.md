# Planner Ablation Comparison

Primary planner-ablation table for the current vertical slice.

| Mode | Avg Sim Calls | Feasible Hit Rate | Efficiency Score | Avg Convergence Step | Focused Truth Ratio | Phase Change Rate | Calib Replan Rate | Rollout Guidance Rate | Avg Rollout Guidance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_system | 3.0 | 1.0 | 0.333333 | 0.0 | 1.0 | 0.666667 | 1.0 | 1.0 | 0.457235 |
| top_k_baseline | 6.0 | 1.0 | 0.166667 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| no_fidelity_escalation | 3.0 | 1.0 | 0.333333 | 0.0 | 0.0 | 0.666667 | 1.0 | 1.0 | 0.457235 |
| no_phase_updates | 3.0 | 1.0 | 0.333333 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.457235 |
| no_calibration_replanning | 3.0 | 1.0 | 0.333333 | 0.0 | 1.0 | 0.333333 | 0.0 | 1.0 | 0.457235 |
| no_rollout_planning | 3.0 | 1.0 | 0.333333 | 0.0 | 1.0 | 0.666667 | 1.0 | 0.0 | 0.0 |
