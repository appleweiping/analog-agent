# Multi-Task Benchmark Design

## Positioning

This document freezes the first conference-facing multi-task benchmark design for `analog-agent`. The goal is not to pretend that every task is already runnable at the same maturity as `ota2_v1`, but to formalize a benchmark suite that can support:

- a primary paper path with full closed-loop evidence;
- cross-family generalization claims that are honest about readiness;
- future expansion without rewriting benchmark semantics.

The benchmark suite is defined in:

- `configs/benchmarks/multi_task_suite_v1.yaml`
- `configs/benchmarks/ota2.yaml`
- `configs/benchmarks/folded_cascode.yaml`
- `configs/benchmarks/ldo.yaml`
- `configs/benchmarks/bandgap.yaml`

## Design Principles

The suite follows five principles.

- One stable primary path: `ota2_v1` remains the canonical paper-primary runnable benchmark.
- Cross-family coverage: the suite now spans amplifier, regulator, and reference tasks.
- Explicit readiness: every benchmark declares whether it is `frozen_runnable`, `spec_ready`, or `planned`.
- Shared reporting axes: simulation budget, feasible hit rate, prediction gap, truth level, fidelity usage, and failure modes are tracked across tasks.
- Honest paper claims: no benchmark is described as executable or physically stronger than its current implementation state.

## Suite Composition

### 1. OTA2 v1

- Benchmark id: `ota2_v1`
- Family: `two_stage_ota`
- Role: `paper_primary`
- Readiness: `frozen_runnable`
- Purpose: main closed-loop experiment path and submission-facing anchor

This task is the only benchmark that currently supports the full Day1-Day12 story end to end:

`DesignTask -> World Model -> Planning -> ngspice Verification -> Calibration -> Memory`

### 2. Folded Cascode v1

- Benchmark id: `folded_cascode_v1`
- Family: `folded_cascode_ota`
- Role: `paper_secondary`
- Readiness: `frozen_runnable`
- Purpose: same broad objective family as OTA, but with a different topology and headroom/stability profile

This task is now the second runnable vertical slice and tests whether the system generalizes beyond the specific `two_stage_ota` topology while staying in the amplifier regime.

### 3. LDO v1

- Benchmark id: `ldo_v1`
- Family: `ldo`
- Role: `generalization_probe`
- Readiness: `frozen_runnable`
- Purpose: move from amplifier synthesis to regulator synthesis and stress the benchmark stack with family-specific regulation metrics

This is the first runnable benchmark that meaningfully broadens the task distribution beyond OTA-family design.

### 4. Bandgap v1

- Benchmark id: `bandgap_v1`
- Family: `bandgap`
- Role: `future_extension`
- Readiness: `spec_ready`
- Purpose: establish the benchmark contract for reference-generation tasks before the full vertical slice is implemented

Bandgap is included now so that the benchmark suite already encodes the intended long-term paper scope, even though it is not yet a runnable primary path.

## Reporting Structure

The suite is organized around three reporting tiers.

### Tier A: Runnable paper evidence

- `ota2_v1`
- `folded_cascode_v1`

This tier supports:

- full acceptance
- full experiment runner
- baseline comparison
- methodology comparison
- statistics export

`ota2_v1` remains the paper-primary path, while `folded_cascode_v1` is the secondary runnable amplifier benchmark used for generalization evidence.

### Tier B: Runnable cross-family generalization

- `ldo_v1`

This tier supports:

- benchmark configuration
- fixed `DesignTask -> Candidate -> Simulation -> Verification -> Memory` object flow
- frozen fidelity/default truth declarations
- acceptance and experiment entry points
- benchmark/stats exports aligned with OTA and folded cascode

### Tier C: Future extension

- `bandgap_v1`

This tier exists to stabilize benchmark semantics early and avoid ad hoc design later.

## Shared Evaluation Axes

All tasks in the suite share these reporting axes:

- task family
- execution readiness
- truth level
- fidelity level
- simulation budget
- feasible hit rate
- prediction gap
- failure mode distribution

This ensures that even before every task is runnable, the experimental language remains consistent.

## Family-Specific Metric Contracts

### OTA2 / Folded Cascode

Primary metrics:

- `dc_gain_db`
- `gbw_hz`
- `phase_margin_deg`
- `power_w`

Auxiliary metrics:

- `output_swing_v`
- `noise_nv_per_sqrt_hz`
- `area_um2`

### LDO

Primary metrics:

- `gbw_hz`
- `phase_margin_deg`
- `power_w`
- `output_swing_v`

Auxiliary metrics:

- `slew_rate_v_per_us`
- `psrr_db`
- `line_regulation_mv_per_v`
- `load_regulation_mv_per_ma`

### Bandgap

Primary metrics:

- `power_w`

Auxiliary metrics:

- `temperature_coefficient_ppm_per_c`
- `line_regulation_mv_per_v`
- `noise_nv_per_sqrt_hz`

## What This Enables for the Paper

This benchmark design already supports a strong paper structure:

- main result on `ota2_v1`
- secondary generalization claim on folded cascode and LDO with runnable vertical slices
- honest statement that the suite is broader than the currently frozen executable path
- direct alignment between benchmark configs, stats aggregation, and future tables/figures

## What Is Still Missing

The benchmark suite is now formally designed, but not fully executable across every task. The following still needs implementation work:

- family-aware templates and measurement contracts in the simulation main path for bandgap
- cross-task baselines beyond OTA/folded-cascode/LDO

## Conference-Ready Interpretation

The benchmark suite should currently be described as:

- one paper-primary runnable benchmark (`ota2_v1`)
- one paper-secondary runnable benchmark (`folded_cascode_v1`)
- one runnable cross-family regulator benchmark (`ldo_v1`)
- one formal future extension (`bandgap_v1`)

This is strong enough to organize the next phase of top-conference experimentation without overclaiming current execution maturity.
