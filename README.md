# analog-agent

`analog-agent` is a research-oriented framework for structured analog circuit design under a layered agent architecture. The project treats natural-language specification understanding, structured task compilation, model-based search, and simulator-backed verification as distinct system roles rather than collapsing them into a single generative loop.

This repository is being developed as an engineering-grade prototype for analog design automation. The public code emphasizes system interfaces, compile-time guarantees, and evaluation scaffolding. Implementation details that are tightly coupled to ongoing research assets are intentionally abstracted at a high level.

## Research Scope

The current repository is organized around four principles:

- specification-first interaction, where user requests are compiled into a strict intermediate representation;
- model-based decision support, where learned surrogates can be used for cheap lookahead and candidate ranking;
- simulator-grounded verification, where final decisions remain constrained by executable testbench logic;
- traceable evaluation, where validation, repair, and benchmark flows are treated as first-class artifacts.

The immediate benchmark focus is analog blocks such as OTA variants, LDOs, and bandgap references.

## Current Capabilities

At the current stage, the repository includes:

- a structured interaction layer that compiles natural-language requirements into a validated `DesignSpec`;
- deterministic normalization, validation, repair, and testbench-planning logic for the front-end compile pipeline;
- a lightweight API surface for compile and validation workflows;
- baseline project scaffolding for planning, simulation, memory, world-model, and evaluation modules;
- automated tests covering standard, underspecified, ambiguous, adversarial, and repair-oriented interaction cases.

This is still an active research codebase rather than a finished release. Some downstream modules remain intentionally minimal while interfaces stabilize.

## Repository Structure

- `apps/`: service entrypoints and worker-facing application modules.
- `configs/`: benchmark, simulator, model, and runtime configuration.
- `libs/`: core schemas, interaction logic, planners, simulation helpers, memory utilities, and evaluation code.
- `research/`: experiment assets, datasets, baselines, and paper-facing material.
- `scripts/`: command-line utilities for dataset, training, benchmarking, and export workflows.
- `tests/`: unit, integration, and regression coverage.
- `infra/`: container, CI, and observability placeholders.

## Getting Started

The repository targets Python `3.11+`.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
python -m unittest discover -s tests -p "test_*.py"
```

To launch the API locally:

```bash
uvicorn apps.api_server.main:app --reload
```

## Design Notes

This repository intentionally exposes stable system boundaries more than research-sensitive internals. In particular, the public implementation is meant to communicate:

- what the system consumes and produces;
- how front-end compilation is validated and repaired;
- how modules are separated for orchestration, planning, simulation, and evaluation;
- how the codebase can support reproducible engineering experiments.

It intentionally does not attempt to publish every modeling choice, prompt strategy, or research-specific heuristic in README form.

## Status

The project is under active construction. Interfaces in the interaction layer are substantially more concrete than the downstream optimization and simulation stack, which will continue to evolve as the research pipeline is expanded.
