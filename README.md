# analog-agent

An initial scaffold for an analog circuit design agent that combines LLM planning, surrogate world models, simulator execution, memory, and benchmarking workflows.

## Goals

- Compile natural-language specifications into structured design intents.
- Search over analog topologies and parameterizations.
- Evaluate candidates with simulators and learned surrogates.
- Log trajectories, failures, and reflections for iterative improvement.
- Support benchmark-driven research loops for OTA, folded cascode, LDO, and bandgap designs.

## Repository Layout

- `configs/`: runtime configuration for models, simulators, surrogates, and benchmarks.
- `apps/`: service and worker entrypoints.
- `libs/`: shared schemas, planners, simulation helpers, memory utilities, and evaluation code.
- `research/`: experiment assets, datasets, baselines, and paper material.
- `tests/`: unit, integration, e2e, and regression coverage.
- `scripts/`: CLI utilities for data generation, training, benchmarking, and export.
- `infra/`: container, CI, and observability support files.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
uvicorn apps.api_server.main:app --reload
```

## Status

This repository is currently a skeleton. Most modules expose placeholder data structures or service stubs intended to be filled in as the architecture solidifies.
