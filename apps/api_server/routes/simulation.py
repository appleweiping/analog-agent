"""Simulation-layer API routes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from libs.schema.design_task import DesignTask
from libs.schema.planning import PlanningBundle, SearchState
from libs.schema.simulation import SimulationCompileResponse, SimulationExecutionResponse
from libs.simulation.compiler import compile_simulation_bundle
from libs.simulation.service import SimulationService

router = APIRouter(prefix="/simulation", tags=["simulation"])


class SimulationCompileRequest(BaseModel):
    """Request payload for fifth-layer bundle compilation."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    planning_bundle: PlanningBundle
    search_state: SearchState
    candidate_id: str
    fidelity_level: str = "quick_truth"
    backend_preference: str = "ngspice"
    escalation_reason: str = "planner_requested_truth_verification"
    paper_mode: bool = False
    model_binding_overrides: dict[str, float | int | str | bool] = Field(default_factory=dict)


@router.post("/compile", response_model=SimulationCompileResponse)
def compile_bundle(request: SimulationCompileRequest) -> SimulationCompileResponse:
    """Compile a formal SimulationBundle."""

    return compile_simulation_bundle(
        request.design_task,
        request.planning_bundle,
        request.search_state,
        request.candidate_id,
        fidelity_level=request.fidelity_level,
        backend_preference=request.backend_preference,
        escalation_reason=request.escalation_reason,
        model_binding_overrides=request.model_binding_overrides,
        paper_mode=request.paper_mode,
    )


@router.post("/verify", response_model=SimulationExecutionResponse)
def verify(request: SimulationCompileRequest) -> SimulationExecutionResponse:
    """Compile and execute fifth-layer truth verification."""

    service = SimulationService(request.design_task, request.planning_bundle, request.search_state)
    return service.verify_candidate(
        request.candidate_id,
        fidelity_level=request.fidelity_level,
        backend_preference=request.backend_preference,
        escalation_reason=request.escalation_reason,
        model_binding_overrides=request.model_binding_overrides,
        paper_mode=request.paper_mode,
    )
