# World Model Utility Comparison

Primary methodology comparison table for the current vertical slice.

| Mode | Avg Sim Calls | Feasible Hit Rate | Avg Convergence Step | Focused Truth Ratio | power_w Gap | temperature_coefficient_ppm_per_c Gap | line_regulation_mv_per_v Gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_system | 1.0 | 0.0 | 3.0 | 1.0 | 0.000159 | 292.186125 | 381403101293.55286 |
| no_world_model | 3.0 | 0.0 | 3.0 | 1.0 | 0.000807 | 312.312088 | 5804782791150.791 |
| no_calibration | 3.0 | 0.0 | 3.0 | 1.0 | 0.000157 | 299.348286 | 127134367098.49329 |
| no_fidelity_escalation | 1.0 | 0.0 | 3.0 | 0.0 | 0.000159 | 292.186125 | 0.0 |
