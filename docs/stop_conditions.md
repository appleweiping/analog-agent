# Stop Conditions

The project should not drift into endless next steps. Use this document with
`scripts/review_harness_closeout.py` whenever a research milestone appears close
to completion.

## Continue Experiments

Continue implementation or experiments when any of these are true:

- paper-facing truth still depends on mock or demonstrator-only evidence for a
  claim that says configured truth, PDK realism, signoff, layout, PEX, or yield;
- benchmark results cover only smoke-test scale budgets, seeds, or task breadth;
- strong baselines are missing, untuned, or not run under the same simulator
  budget and truth policy;
- the world-model claim lacks train/eval split, calibration, uncertainty,
  OOD, or rollout-error evidence;
- LLM/agent claims lack tool traces, prompt/schema versions, repair-loop
  evidence, and LLM/no-LLM ablations;
- generated outputs cannot be reproduced from committed scripts, configs, seeds,
  and archived raw logs.

## Stop Experiments And Write

Move to writing when all of these are true:

- the closeout review returns `ready_for_writing`;
- remaining reviewer concerns are wording, framing, or optional breadth rather
  than missing evidence for the central claim;
- new experiments are unlikely to change the paper's claim boundary;
- benchmark scripts, raw outputs, plots, tables, configs, seeds, and model
  checkpoints are archived and reproducible;
- the abstract/title avoid claims not supported by configured evidence.

## Current Default Claim Boundary

The safe default claim is:

> A calibrated, surrogate-guided analog sizing loop under explicit SPICE truth
> boundaries.

Avoid broader claims until the closeout gate says they are supported.
