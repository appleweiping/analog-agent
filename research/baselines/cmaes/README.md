# CMA-ES Baseline

This directory documents the formal `cmaes_baseline` used in the unified benchmark runner.

The implementation is intentionally lightweight and research-facing:

- it does not call the project world model;
- it does not consume calibration updates;
- it does not use focused-truth escalation;
- it does adapt its sampling center and spread from previously verified real observations.

Within the repository, this baseline is treated as a classical optimization-style comparator rather than an agentic method. It runs through the same `DesignTask`, real `ngspice`, stats/export, and benchmark-suite contracts as the full system, so comparisons remain structurally fair.
