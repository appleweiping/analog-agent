# Planner Delta Against Simple Top-K Baseline

Primary paper-facing delta table against the simple top-K planner baseline.

| Mode | Delta Sim Calls | Delta Feasible Hit | Delta Efficiency | Delta Focused Ratio | Delta Phase Change | Delta Calib Replan |
| --- | --- | --- | --- | --- | --- | --- |
| full_system | -3.0 | 0.0 | 0.0 | 1.0 | 0.666667 | 1.0 |
| no_fidelity_escalation | -3.0 | 0.0 | 0.0 | 0.0 | 0.666667 | 1.0 |
| no_phase_updates | -3.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| no_calibration_replanning | -3.0 | 0.0 | 0.0 | 1.0 | 0.666667 | 0.0 |
| no_rollout_planning | -3.0 | 0.0 | 0.0 | 1.0 | 0.666667 | 1.0 |
