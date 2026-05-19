# CLAUDE.md — Agent-AI4EDA

You are working on the Agent-AI4EDA research project: a layered AI agent for analog circuit design automation.

## Mandatory Read Order
1. `AGENTS.md` — authoritative engineering contract
2. `README.md` — system architecture and usage
3. `CONTEXT.md` — current project state snapshot
4. This file

## Quick Context
- GitHub: https://github.com/appleweiping/analog-agent
- Stage: Paper submission preparation (60-day plan completed)
- Core code: `libs/` (schema, interaction, planner, simulation, world_model, memory, eval)
- Apps: `apps/` (api_server, orchestrator, workers)
- Configs: `configs/` (benchmarks, simulator, LLM, PDK, world_model)
- Tests: `tests/` (unit, integration, regression)
- Paper materials: `../paper/` (layer specs, engineering reports — outside git)
- Planning: `../plan/` (upgrades plan, schedule, daily reports — outside git)
- Tools: `../tools/ngspice/` (ngspice 46 Windows x64 — outside git)
- Python: 3.11+ (venv at .venv/)

## Critical Rules
1. Never fabricate simulation results or claim untested circuits work
2. All SPICE verification must use real ngspice runs, not mocked
3. Follow the six-layer architecture: Spec → Task → WorldModel → Plan → Sim → Memory
4. Four frozen vertical slices: ota2_v1, folded_cascode_v1, ldo_v1, bandgap_v1
5. Evidence levels: L0 (design) → L6 (paper-ready with statistical significance)
6. Run tests before committing: `python -m pytest tests/ -x`

## Current Phase
Post-execution. The 60-day plan (Days 1-60) is fully completed. Remaining work:
- External PDK integration (sky130 real runs)
- Spectre backend support
- Final paper manuscript assembly
- Submission package finalization

## Agent Roles
- **Opus/Claude**: Architecture review, complex reasoning, paper writing
- **Codex**: Parallel execution, server commands, bulk implementation
- **OpenCode**: Implementation, testing, doc updates
