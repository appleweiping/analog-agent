# Folded Cascode OTA v1

`folded_cascode_v1` is the second runnable vertical slice after `ota2_v1`.

It fixes the full object chain:

`DesignTask -> CandidateRecord -> SimulationBundle -> VerificationResult -> EpisodeMemoryRecord`

## Frozen Scope

- Family: `folded_cascode_ota`
- Topology mode: `fixed`
- Physical validity: `demonstrator_truth`
- Default backend: `ngspice`
- Default fidelity: `quick_truth`
- Promoted fidelity: `focused_truth`

## Frozen Variables

- `w_in`: input pair width
- `l_in`: input pair length
- `w_cas`: folded branch width
- `l_cas`: folded branch length
- `ibias`: total bias current proxy
- `cc`: stability capacitor

## Measurement Contract

Primary reporting metrics:

- `dc_gain_db`
- `gbw_hz`
- `phase_margin_deg`
- `power_w`

Focused-truth diagnostic metric:

- `slew_rate_v_per_us`

## Fidelity Policy

- `quick_truth`: `op + ac`
- `focused_truth`: `op + ac + tran`

Escalation is intended for near-feasible candidates, measurement anomalies, or planner-requested higher-confidence verification.

## Model Binding

- Default model type: `builtin`
- Default truth level: `demonstrator_truth`
- External model card integration remains supported through the general model-binding configuration path

## Entry Points

- Acceptance: `libs.vertical_slices.folded_cascode.run_folded_cascode_acceptance()`
- Experiment suite: `libs.vertical_slices.folded_cascode.run_folded_cascode_experiment_suite()`

This slice is intended as the paper-secondary amplifier benchmark and should remain semantically stable as `folded_cascode_v1`.

