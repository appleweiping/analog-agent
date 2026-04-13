# Random Search Baseline

`random_search_baseline` is the first research-grade non-agent baseline integrated into the main experiment pipeline.

It is intentionally implemented as a unified `ExperimentMode`, not as a standalone script. That means it:

- consumes the same `DesignTask`
- uses the same benchmark runners
- executes the same real `ngspice` verification path
- emits the same `ExperimentResult`, `ExperimentStatsRecord`, and exported stats summaries

Current behavior:

- samples candidates directly from `DesignTask.design_space`
- does not use the world model for ranking or screening
- does not apply calibration updates
- does not use fidelity escalation
- runs real verification through the same L5 simulation service as the full system

This baseline is meant to answer a research question:

Can the agentic closed-loop system outperform a simple surrogate-free search process under the same simulation budget and benchmark protocol?
