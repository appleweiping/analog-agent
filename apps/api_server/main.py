"""FastAPI entrypoint for the analog-agent control plane."""

from __future__ import annotations

from fastapi import FastAPI

from apps.api_server.routes.acceptance import router as acceptance_router
from apps.api_server.deps import get_runtime_summary
from apps.api_server.routes.experiments import router as experiments_router
from apps.api_server.routes.interaction import router as interaction_router
from apps.api_server.routes.memory_reflection import router as memory_reflection_router
from apps.api_server.routes.planning import router as planning_router
from apps.api_server.routes.simulation import router as simulation_router
from apps.api_server.routes.tasking import router as tasking_router
from apps.api_server.routes.world_modeling import router as world_modeling_router

app = FastAPI(title="analog-agent", version="0.1.0")
app.include_router(acceptance_router)
app.include_router(experiments_router)
app.include_router(interaction_router)
app.include_router(memory_reflection_router)
app.include_router(planning_router)
app.include_router(simulation_router)
app.include_router(tasking_router)
app.include_router(world_modeling_router)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Report service liveness."""
    return {"status": "ok"}


@app.get("/runtime")
def runtime_summary() -> dict[str, str]:
    """Expose lightweight runtime metadata."""
    return get_runtime_summary()
