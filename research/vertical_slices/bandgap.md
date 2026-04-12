# Bandgap v1

`bandgap_v1` is the fourth runnable vertical slice and the first frozen reference-family path.

It fixes the full object chain:

`DesignTask -> CandidateRecord -> SimulationBundle -> VerificationResult -> EpisodeMemoryRecord`

## Frozen Scope

- Family: `bandgap`
- Topology mode: `fixed`
- Physical validity: `demonstrator_truth`
- Default backend: `ngspice`
- Default fidelity: `quick_truth`
- Promoted fidelity: `focused_truth`

## Frozen Variables

- `area_ratio`: PTAT-to-CTAT device area ratio proxy
- `r1`: PTAT branch resistance
- `r2`: CTAT summation resistance
- `w_core`: core bias transistor width proxy
- `l_core`: core bias transistor length proxy
- `ibias`: startup and core bias current proxy

## Measurement Contract

Primary reporting metrics:

- `power_w`
- `temperature_coefficient_ppm_per_c`
- `line_regulation_mv_per_v`

## Fidelity Policy

- `quick_truth`: `op`
- `focused_truth`: `op + tran`

Escalation is intended for near-feasible candidates, verification anomalies, or planner-requested higher-confidence regulation review.

## Model Binding

- Default model type: `builtin`
- Default truth level: `demonstrator_truth`
- External model card integration remains supported through the general model-binding configuration path

## Entry Points

- Acceptance: `libs.vertical_slices.bandgap.run_bandgap_acceptance()`
- Experiment suite: `libs.vertical_slices.bandgap.run_bandgap_experiment_suite()`

This slice is intended as the paper-secondary reference benchmark and should remain semantically stable as `bandgap_v1`.
