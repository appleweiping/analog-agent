# Repository Map

`analog-agent` is organized as a harness-first analog circuit design research
system. The tracked repository should stay focused on source code, contracts,
configuration, templates, tests, and reproducible command-line entrypoints.

## Control Plane

- `apps/api_server/`: FastAPI routes for interaction, tasking, world modeling,
  planning, simulation, memory, experiments, and acceptance checks.
- `apps/orchestrator/`: cross-layer job and truth-loop orchestration.
- `apps/worker_llm/`: provider-neutral LLM worker adapters and role wrappers.
- `apps/worker_simulator/`: simulator bindings for ngspice, Xyce-compatible, and
  Spectre-compatible execution paths.
- `apps/worker_world_model/` and `apps/worker_memory/`: worker-facing adapters
  around core `libs/` services.

## Core Libraries

- `libs/schema/`: typed public contracts for specs, tasks, actions, world models,
  planning, simulation, memory, statistics, benchmark evidence, and submission
  packages.
- `libs/interaction/` and `libs/tasking/`: deterministic spec parsing,
  normalization, validation, repair, and task compilation.
- `libs/world_model/`: heuristic and learned-surrogate interfaces, calibration,
  state construction, feature projection, and validation.
- `libs/planner/`: candidate lifecycle, budgets, acquisition, rollout, MPC-style
  planning, and selection.
- `libs/simulation/`: testbench building, backend routing, netlist generation,
  measurement extraction, artifact registration, replay checks, and validation.
- `libs/memory/`: episodic memory, retrieval, pattern mining, governance, and
  strategy advice.
- `libs/eval/`: benchmark runners, statistics, baseline helpers, paper evidence,
  and submission-package builders.
- `libs/vertical_slices/`: frozen OTA, folded-cascode, LDO, and bandgap entrypoints.
- `libs/knowledge/`: compact cold-memory notes for architecture, related work,
  analog rules, equations, topology families, and failure modes.

## Configuration And Assets

- `configs/benchmarks/`: frozen benchmark suite and per-family task definitions.
- `configs/llm/`: role defaults for parser, planner, and critic workers.
- `configs/simulator/`: backend and truth-boundary configuration.
- `configs/world_model/`: surrogate and uncertainty configuration.
- `templates/netlist/` and `templates/testbench/`: frozen family-aware netlist
  and measurement contracts.

## Tests And Scripts

- `tests/unit/`: schema, service, evidence, harness, and review checks.
- `tests/integration/`: route, simulator, benchmark, memory, and evidence flows.
- `tests/regression/`: layer acceptance guards.
- `scripts/`: reproducible command-line entrypoints for tests, datasets,
  benchmarks, evidence generation, truth readiness, and review gates.

## Local Output Policy

- `archive/`: the only durable local sink for paper drafts, benchmark exports,
  experiment results, working notes, and historical outputs. It is ignored by git.
- `.artifacts/`: transient simulator/cache output only. It may be deleted or
  regenerated.
- `.pdk/`: local user-managed PDK staging area. It is ignored by git.

Do not reintroduce tracked tree snapshots such as `project_tree`; this file is
the repository map authority.
