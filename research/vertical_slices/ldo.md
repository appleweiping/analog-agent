# LDO v1

`ldo_v1` is the third runnable vertical slice and the first regulator-family path.

It fixes the full object chain:

`DesignTask -> CandidateRecord -> SimulationBundle -> VerificationResult -> EpisodeMemoryRecord`

## Frozen Scope

- Family: `ldo`
- Topology mode: `fixed`
- Physical validity: `demonstrator_truth`
- Default backend: `ngspice`
- Default fidelity: `quick_truth`
- Promoted fidelity: `focused_truth`

## Frozen Variables

- `w_pass`: PMOS pass-device width
- `l_pass`: PMOS pass-device channel length
- `w_err`: error-amplifier width proxy
- `l_err`: error-amplifier length proxy
- `ibias`: quiescent bias current proxy
- `c_comp`: loop compensation capacitor

## Measurement Contract

Primary reporting metrics:

- `gbw_hz`
- `phase_margin_deg`
- `power_w`
- `output_swing_v`

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

- Acceptance: `libs.vertical_slices.ldo.run_ldo_acceptance()`
- Experiment suite: `libs.vertical_slices.ldo.run_ldo_experiment_suite()`

This slice is intended as the paper-secondary regulator benchmark and should remain semantically stable as `ldo_v1`.
