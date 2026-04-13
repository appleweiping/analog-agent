# BayesOpt Baseline

`bayesopt_baseline` is the second research-grade baseline integrated into the main experiment runner.

It is intentionally lightweight and deterministic:

- it does **not** call the project world model
- it does **not** apply calibration updates
- it does **not** use fidelity escalation
- it does use real `ngspice` verification for all selected candidates

The baseline builds a simple acquisition surrogate from previously verified candidates:

- parameter vectors are normalized from `DesignTask.design_space`
- observed truth metrics are aggregated with a kernel-weighted surrogate
- uncertainty is derived from distance to known samples
- acquisition combines surrogate utility and uncertainty

This keeps the baseline academically meaningful while remaining fully reproducible and compatible with the unified benchmark, stats, and export pipeline.
