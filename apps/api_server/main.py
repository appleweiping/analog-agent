"""FastAPI entrypoint for the analog-agent control plane."""

from __future__ import annotations

from fastapi import FastAPI

from apps.api_server.deps import get_runtime_summary
from apps.api_server.routes.interaction import router as interaction_router

app = FastAPI(title="analog-agent", version="0.1.0")
app.include_router(interaction_router)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Report service liveness."""
    return {"status": "ok"}


@app.get("/runtime")
def runtime_summary() -> dict[str, str]:
    """Expose lightweight runtime metadata."""
    return get_runtime_summary()
