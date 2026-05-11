# Related Work Map

This map is a harness document, not a literature survey. It defines the external
anchors that future agents must compare against before making novelty or
readiness claims.

## Harness Engineering

- OpenAI Harness Engineering: agent-first development depends on tools,
  context, validation, and feedback loops around the model.
- Zhihu harness references: treat prompt text as only one part of the system;
  retrieval, task decomposition, permission boundaries, persistent state, and
  automated verification are the real engineering surface.
- Anthropic-style long-running harnesses: long tasks need observable state,
  scoped tools, durable checkpoints, and reviewable traces.

## Analog Design And Benchmark Anchors

- AnalogGym and AICircuit: benchmark breadth, fixed budgets, multiple tasks,
  reproducible simulator evidence, and fair baseline comparisons.
- AnalogCoder and AnalogCoder-Pro: LLM-assisted analog design workflows with
  reusable circuit knowledge, feedback loops, and simulation-grounded repair.
- AnalogGenie: topology generation is a separate high-standard contribution,
  not something to claim from sizing-only demos.
- AutoCkt and related RL sizing work: RL baselines must be budget-matched and
  tuned, not represented by a toy policy.
- OpenFASOC and ALIGN: layout/generator handoff, DRC/LVS/PEX, and open-source
  implementation flow define a higher bar than schematic-only closure.

## EDA LLM Evaluation Anchors

- ChipNeMo: domain adaptation and retrieval need explicit evaluation.
- VerilogEval: automatic, executable evaluation is part of the benchmark.
- RTLFixer and HaVen: compiler/simulator feedback and hallucination controls are
  first-class agent mechanisms.

## World Model And Planning Anchors

- I-JEPA and V-JEPA 2: world models predict latent structure rather than surface
  detail and are judged by prediction, planning, and surprise handling.
- TD-MPC2 and DreamerV3: learned models should support rollout, uncertainty, and
  budget-aware control, with horizon-error evidence.

## Required Positioning

Until stronger evidence exists, describe this repository as a calibrated,
SPICE-grounded analog sizing harness. Do not claim a general analog design
agent, full signoff flow, or top-tier world model unless the corresponding
evidence passes `scripts/review_harness_closeout.py`.
