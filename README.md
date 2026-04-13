# analog-agent

`analog-agent` is a research-oriented framework for structured analog circuit design under a layered agent architecture. The project treats natural-language specification understanding, structured task compilation, model-based search, and simulator-backed verification as distinct system roles rather than collapsing them into a single generative loop.

This repository is being developed as an engineering-grade prototype for analog design automation. The public code emphasizes system interfaces, compile-time guarantees, structured validation, and system-level evaluation scaffolding. Implementation details that are tightly coupled to ongoing research assets are intentionally abstracted at a high level.

## Research Scope

The current repository is organized around four principles:

- specification-first interaction, where user requests are compiled into a strict intermediate representation;
- model-based decision support, where learned surrogates can be used for cheap lookahead and candidate ranking;
- simulator-grounded verification, where final decisions remain constrained by executable testbench logic;
- traceable evaluation, where validation, repair, and benchmark flows are treated as first-class artifacts.

The immediate benchmark focus is analog blocks such as OTA variants, LDOs, and bandgap references, under a full six-layer agent architecture. At the current stage, the canonical paper-facing vertical slice is a frozen `two_stage_ota` path (`ota2_v1`) backed by real `ngspice` verification, with `folded_cascode_v1`, `ldo_v1`, and `bandgap_v1` now available as additional runnable generalization benchmarks.

## Current Capabilities

At the current stage, the repository includes:

- a structured interaction layer that compiles natural-language requirements into a validated `DesignSpec`;
- a task-formalization layer that compiles `DesignSpec` into a solver-facing `DesignTask`;
- a world-model layer that exposes structured state, action, transition, uncertainty, trust, ranking, rollout, and calibration contracts;
- a planning and optimization layer that maintains explicit search state, candidate lifecycle, budget control, and optimization traces;
- a ground-truth simulation and verification layer that realizes netlists, executes structured multi-analysis validation flows, adjudicates constraints, attributes failures, certifies robustness, and emits calibration/planner feedback;
- a memory and reflection layer that consolidates full episodes into evidence-backed knowledge objects, mines cross-episode patterns, emits governed strategy feedback, and controls long-term knowledge decay;
- deterministic normalization, validation, repair, testbench-planning, acceptance reporting, retrieval, governance, and API workflows across the implemented layers;
- automated unit, integration, and regression tests spanning interaction, tasking, world model, planning, simulation, and memory/reflection routes.
- a frozen OTA vertical slice (`ota2_v1`) with canonical config, netlist/testbench templates, fixed measurement targets, default fidelity policy, default demonstrator model binding, and CI-backed regression fixtures.
- a second runnable amplifier vertical slice (`folded_cascode_v1`) that reuses the same system contracts while changing topology, headroom behavior, and benchmark role.
- a runnable regulator vertical slice (`ldo_v1`) that extends the same contracts into output regulation, loop stability, and output-voltage verification.
- a runnable reference vertical slice (`bandgap_v1`) that extends the same contracts into low-power reference generation, temperature-stability proxies, and line-regulation verification.
- research-grade benchmark baselines that now include `random_search_baseline` and `bayesopt_baseline` inside the same experiment runner, stats pipeline, benchmark exports, and API routes as the full agent system.

This is still an active research codebase rather than a finished release. The first six system layers are now present in formal schema-and-service form, while research-tuned backend details and later experimental extensions remain under active iteration.

## Repository Structure

- `apps/`: service entrypoints and worker-facing application modules.
- `configs/`: benchmark, simulator, model, and runtime configuration.
- `libs/`: core schemas, interaction logic, task compilers, world-model services, planning services, simulation/verification services, memory utilities, and evaluation code.
- `templates/`: frozen family-aware netlist, testbench, and measurement-contract assets for canonical verification paths.
- `research/`: experiment assets, datasets, baselines, and paper-facing material.
- `scripts/`: command-line utilities for dataset, training, benchmarking, and export workflows.
- `tests/`: unit, integration, and regression coverage.
- `infra/`: container, CI, and observability placeholders.

## Project Tree

The repository now exposes a paper-facing system tree with four runnable benchmark slices, unified benchmark baselines, and frozen family-aware templates:

```text
analog-agent/
|-- configs/
|   |-- benchmarks/
|   |   |-- ota2.yaml
|   |   |-- folded_cascode.yaml
|   |   |-- ldo.yaml
|   |   |-- bandgap.yaml
|   |   `-- multi_task_suite_v1.yaml
|-- libs/
|   |-- eval/
|   |   |-- experiment_runner.py
|   |   |-- random_search.py
|   |   |-- bayesopt.py
|   |   `-- stats.py
|   |-- world_model/
|   |-- planner/
|   |-- simulation/
|   |-- memory/
|   `-- vertical_slices/
|       |-- ota2.py
|       |-- folded_cascode.py
|       |-- ldo.py
|       `-- bandgap.py
|-- templates/
|   |-- netlist/
|   |   |-- ota2/v1/
|   |   |-- folded_cascode/v1/
|   |   |-- ldo/v1/
|   |   `-- bandgap/v1/
|   `-- testbench/
|       |-- ota2/v1/
|       |-- folded_cascode/v1/
|       |-- ldo/v1/
|       `-- bandgap/v1/
|-- research/
|   |-- benchmarks/
|   |-- baselines/
|   |   |-- random_search/
|   |   `-- bayesopt/
|   |-- vertical_slices/
|   `-- papers/
|-- scripts/
|   |-- run_benchmark.py
|   |-- run_ota_experiment_suite.py
|   |-- run_folded_cascode_experiment_suite.py
|   |-- run_ldo_experiment_suite.py
|   `-- run_bandgap_experiment_suite.py
`-- tests/
    `-- integration/
        |-- test_random_search_baseline.py
        |-- test_bayesopt_baseline.py
        |-- test_ota_vertical_slice.py
        |-- test_folded_cascode_vertical_slice.py
        |-- test_ldo_vertical_slice.py
        `-- test_bandgap_vertical_slice.py
```

The full repository snapshot is stored in [`project_tree`](./project_tree) and updated alongside the runnable benchmark paths.

## Getting Started

The repository targets Python `3.11+`. On this codebase, the recommended Windows workflow is Python `3.12` inside a local `.venv`.

On Windows, prefer `py -3.12` instead of a bare `python` command. This avoids interpreter drift when multiple Python installations exist on the same machine, such as MSYS2 Python alongside the official CPython installer.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
python -m unittest discover -s tests -p "test_*.py"
```

On Windows PowerShell, use the following instead:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.12 -m pip install -U pip setuptools wheel
py -3.12 -m pip install -e ".[dev]"
py -3.12 -c "import fastapi, httpx, pytest; print('OK')"
py -3.12 scripts\run_test_suite.py
```

You can also bootstrap the same environment with:

```powershell
.\scripts\bootstrap_dev_env.ps1
```

On Windows, that bootstrap script temporarily clears `PIP_NO_INDEX` and common proxy environment variables for the script process only when they would otherwise block package installation. It does not permanently modify your global shell or system configuration.

Recommended Windows workflow:

```powershell
.\scripts\bootstrap_dev_env.ps1
.\scripts\run_test_suite.ps1
.\scripts\run_test_suite.ps1 -UseVenv -RequireApiDeps
```

This is the safest path on machines that have multiple Python installations or preconfigured proxy settings. The bootstrap script prints the launcher target, the active `.venv` interpreter, and the `pip` install target before dependency installation. It also verifies `fastapi`, `httpx`, and `pytest` before you move into API-inclusive testing.

## Testing

The repository uses two testing modes so local development is not blocked by missing web dependencies:

- `make test`: runs the full `unittest` discovery suite in the current environment. API tests may be skipped when `fastapi/httpx` are not installed.
- `make test-all`: runs the same suite from `.venv`, requires API-test dependencies to be present, and fails fast if the interpreter is not the repository `.venv`.
- `make test-api`: runs all API integration tests only from `.venv`, including interaction, task formalization, world model, planning, and simulation routes.

If `make` is unavailable on Windows, use the PowerShell wrappers or direct `py -3.12` commands instead.

Equivalent direct commands:

```bash
python scripts/run_test_suite.py
.venv\Scripts\python.exe scripts/run_test_suite.py --require-api-deps
```

Windows-specific equivalents:

```powershell
py -3.12 scripts\run_test_suite.py
.\.venv\Scripts\python.exe scripts\run_test_suite.py --require-api-deps
.\scripts\run_test_suite.ps1
.\scripts\run_test_suite.ps1 -UseVenv -RequireApiDeps
```

If your machine has multiple Python installations, do not use a bare `python` command for this repository on Windows. Use `py -3.12` consistently for environment creation, package installation, and test execution.

For a full local validation pass, prefer the `.venv`-backed path:

```powershell
.\scripts\run_test_suite.ps1 -UseVenv -RequireApiDeps
```

That command is the local equivalent of the CI expectation: API dependencies must be present, the interpreter must be the repository `.venv`, and no integration tests are silently skipped. If you still see skipped API tests, you are almost certainly on the lightweight path rather than the strict `.venv` path.

CI installs `.[dev]` and always runs the full API-inclusive suite, so `main` is protected even when local quick checks use the lighter path.

To launch the API locally:

```bash
uvicorn apps.api_server.main:app --reload
```

## Frozen Vertical Slices

The repository now ships with four frozen runnable slices:

- `ota2_v1`: the paper-primary physical closure path
- `folded_cascode_v1`: the paper-secondary runnable generalization path
- `ldo_v1`: the runnable regulator generalization path
- `bandgap_v1`: the runnable reference generalization path

Across these slices, the default benchmark-facing comparison stack now includes:

- `full_simulation_baseline`
- `random_search_baseline`
- `bayesopt_baseline`
- `no_world_model_baseline`
- `full_system`

The `ota2_v1` path fixes:

- the benchmark definition in `configs/benchmarks/ota2.yaml`;
- the canonical netlist and testbench templates under `templates/netlist/ota2/v1/` and `templates/testbench/ota2/v1/`;
- the core measurement contract for `dc_gain_db`, `gbw_hz`, `phase_margin_deg`, and `power_w`;
- the default fidelity ladder of `quick_truth -> focused_truth`;
- the default physical validity declaration of `builtin + demonstrator_truth`.

The standard OTA entrypoints are:

```bash
python scripts/run_ota_acceptance.py
python scripts/run_ota_experiment_suite.py
```

Windows equivalents:

```powershell
py -3.12 scripts\run_ota_acceptance.py
py -3.12 scripts\run_ota_experiment_suite.py
```

The corresponding folded-cascode entrypoints are:

```bash
python scripts/run_folded_cascode_acceptance.py
python scripts/run_folded_cascode_experiment_suite.py
```

Windows equivalents:

```powershell
py -3.12 scripts\run_folded_cascode_acceptance.py
py -3.12 scripts\run_folded_cascode_experiment_suite.py
```

The corresponding LDO entrypoints are:

```bash
python scripts/run_ldo_acceptance.py
python scripts/run_ldo_experiment_suite.py
```

Windows equivalents:

```powershell
py -3.12 scripts\run_ldo_acceptance.py
py -3.12 scripts\run_ldo_experiment_suite.py
```

The corresponding bandgap entrypoints are:

```bash
python scripts/run_bandgap_acceptance.py
python scripts/run_bandgap_experiment_suite.py
```

Windows equivalents:

```powershell
py -3.12 scripts\run_bandgap_acceptance.py
py -3.12 scripts\run_bandgap_experiment_suite.py
```

These entrypoints execute the main system contracts rather than test-only helpers, and they are protected by dedicated regression fixtures in CI.

## Layered System

The repository currently exposes a six-layer execution spine:

- Layer 1: specification understanding and validated `DesignSpec` compilation.
- Layer 2: task formalization into executable optimization problem instances (`DesignTask`).
- Layer 3: task-conditioned world-model services for prediction, rollout, trust, and calibration.
- Layer 4: budget-aware, uncertainty-aware planning over explicit search state and candidate records.
- Layer 5: ground-truth simulation, verification, robustness certification, and structured truth feedback.
- Layer 6: cross-episode memory, reflection, governed retrieval, and advisory strategy feedback.

The current most stable end-to-end paths across these six layers are `ota2_v1`, `folded_cascode_v1`, `ldo_v1`, and `bandgap_v1`, with `ota2_v1` retained as the primary submission-facing path.

This repository intentionally exposes stable system boundaries more than research-sensitive internals. In particular, the public implementation is meant to communicate:

- what the system consumes and produces;
- how front-end compilation is validated and repaired;
- how modules are separated for orchestration, planning, simulation, and evaluation;
- how the codebase can support reproducible engineering experiments.

It intentionally does not attempt to publish every modeling choice, prompt strategy, or research-specific heuristic in README form.

## Status

The project is under active construction, but it is no longer only a front-end scaffold. The repository now contains formal implementations for the first six layers of the system architecture, along with API routes, acceptance-oriented test coverage, and a frozen OTA `v1` vertical slice for reproducible end-to-end physical closure. What remains intentionally lightweight in the public tree are research-tuned backend details, deeper learned-backend integrations, broader circuit-family support, and stronger configured-truth simulator/model integrations that will continue to evolve as the research pipeline is expanded.
