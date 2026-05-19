# CODEX.md — Agent-AI4EDA

Read `AGENTS.md` first. That is the authoritative contract.

## Your Role
You are the execution engine. Your job:
- Implement features in `libs/` and `apps/`
- Run tests: `python -m pytest tests/ -x`
- Build and verify: `make lint test`
- Generate daily reports in `../plan/reports/`

## Key Paths
- Source: `libs/` (schema, interaction, planner, simulation, world_model, memory, eval)
- Apps: `apps/` (api_server, orchestrator, workers)
- Tests: `tests/`
- Configs: `configs/`
- Templates: `templates/`

## Non-Negotiable
- Never skip SPICE verification
- Never commit to main without passing tests
- Never modify frozen vertical slice configs without explicit approval
- Always update `docs/repo-map.md` when adding new modules
