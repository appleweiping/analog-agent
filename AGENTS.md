# Agent Operating Contract

This repository is maintained as a harness-first analog design system. Agents
must preserve the six-layer implementation spine while keeping paper, benchmark,
and experiment claims inside explicit SPICE truth boundaries.

## Default Reading Set

Before starting non-trivial work, read or refresh these files:

- `README.md`
- `docs/configured_truth_user_action_boundary.md`
- `docs/repo-map.md`
- `docs/related_work_map.md`
- `docs/stop_conditions.md`
- `configs/default.yaml`
- `configs/benchmarks/multi_task_suite_v1.yaml`
- `configs/simulator/ngspice.yaml`
- `scripts/run_system_closure_report.py`

## Complex Task Rule

A task is complex when it crosses subsystems, touches more than one file group,
changes experiment or paper conclusions, alters public schemas/APIs, or changes
simulation/benchmark behavior. Complex tasks must use at least:

- one explorer-style pass to map current code and evidence;
- one reviewer-style pass to attack novelty, evaluation, and regression risks.

The main implementer remains responsible for integration, tests, and final
judgment. Multi-agent work is a harness requirement, not a substitute for review.

## Work Boundaries

- Keep apps thin; shared behavior belongs in `libs/`.
- Keep generated experiment outputs in `archive/`.
- Keep `.artifacts/` limited to transient simulator/cache output.
- Do not claim signoff, full-PDK, layout, PEX, yield, or universal analog-design
  closure unless the evidence is present and reproducible.
- Treat SPICE/configured-truth runs as the only paper-facing physical evidence.
- Keep lightweight internal baselines labeled as lightweight until external,
  tuned baselines are wired into the same budget and truth loop.

## Completion Report

Every completed complex task report must state:

- what changed;
- what was verified;
- what remains risky or incomplete;
- the next plan;
- whether the project should continue experiments or stop and write.

If the stop decision is unclear, run `scripts/review_harness_closeout.py` and
explain the blockers rather than proposing open-ended next steps.
