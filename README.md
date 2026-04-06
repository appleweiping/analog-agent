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
- `make test-api`: runs all API integration tests only from `.venv`, including interaction, task formalization, and world model routes.

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

## Design Notes

This repository intentionally exposes stable system boundaries more than research-sensitive internals. In particular, the public implementation is meant to communicate:

- what the system consumes and produces;
- how front-end compilation is validated and repaired;
- how modules are separated for orchestration, planning, simulation, and evaluation;
- how the codebase can support reproducible engineering experiments.

It intentionally does not attempt to publish every modeling choice, prompt strategy, or research-specific heuristic in README form.

## Status

The project is under active construction. Interfaces in the interaction layer are substantially more concrete than the downstream optimization and simulation stack, which will continue to evolve as the research pipeline is expanded.
