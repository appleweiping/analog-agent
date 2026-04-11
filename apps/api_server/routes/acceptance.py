"""System-level E2E acceptance routes."""

from __future__ import annotations

from fastapi import APIRouter

from apps.orchestrator.job_runner import run_full_system_acceptance
from libs.schema.system_binding import AcceptanceTaskConfig, SystemAcceptanceResult

router = APIRouter(prefix="/acceptance", tags=["acceptance"])


@router.post("/run-full-system", response_model=SystemAcceptanceResult)
def run_acceptance(request: AcceptanceTaskConfig) -> SystemAcceptanceResult:
    """Run the formal Day-8 end-to-end acceptance chain."""

    return run_full_system_acceptance(request)
