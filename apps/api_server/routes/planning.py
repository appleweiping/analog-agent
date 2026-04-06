"""Planning-layer API routes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_task import DesignTask
from libs.schema.planning import (
    ActionPlanResponse,
    CandidateBatchResponse,
    PlanningBestResult,
    PlanningBundle,
    PlanningCompileResponse,
    SearchInitializationResponse,
    SearchState,
    SimulationFeedbackResponse,
    SimulationSelectionResponse,
    TerminationDecision,
)
from libs.schema.world_model import TruthCalibrationRecord, WorldModelBundle

router = APIRouter(prefix="/planning", tags=["planning"])


class PlanningContextRequest(BaseModel):
    """Base request payload containing task, planning bundle, and world-model bundle."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    planning_bundle: WorldModelBundle | None = None


class InitializePlanningRequest(BaseModel):
    """Request payload for search initialization."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    world_model_bundle: WorldModelBundle


class SearchStateRequest(BaseModel):
    """Request payload containing search context."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    world_model_bundle: WorldModelBundle
    planning_bundle: PlanningBundle
    search_state: SearchState


class SimulationFeedbackRequest(BaseModel):
    """Request payload for simulation feedback ingestion."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    world_model_bundle: WorldModelBundle
    planning_bundle: PlanningBundle
    search_state: SearchState
    candidate_id: str
    truth_record: TruthCalibrationRecord


def _service(design_task: DesignTask, world_model_bundle: WorldModelBundle, planning_bundle_obj: PlanningBundle | None = None) -> tuple[PlanningService, PlanningBundle]:
    compiled = compile_planning_bundle(design_task, world_model_bundle)
    if compiled.planning_bundle is None:
        raise ValueError("planning bundle failed to compile")
    bundle = planning_bundle_obj or compiled.planning_bundle
    return PlanningService(bundle, design_task, world_model_bundle), bundle


@router.post("/compile", response_model=PlanningCompileResponse)
def compile_bundle(request: InitializePlanningRequest) -> PlanningCompileResponse:
    """Compile a formal PlanningBundle."""

    return compile_planning_bundle(request.design_task, request.world_model_bundle)


@router.post("/initialize", response_model=SearchInitializationResponse)
def initialize(request: InitializePlanningRequest) -> SearchInitializationResponse:
    """Initialize planner search state."""

    compiled = compile_planning_bundle(request.design_task, request.world_model_bundle)
    if compiled.planning_bundle is None:
        raise ValueError("planning bundle failed to compile")
    return PlanningService(compiled.planning_bundle, request.design_task, request.world_model_bundle).initialize_search()


@router.post("/propose", response_model=CandidateBatchResponse)
def propose(request: SearchStateRequest) -> CandidateBatchResponse:
    """Propose new candidates."""

    service, _ = _service(request.design_task, request.world_model_bundle, request.planning_bundle)
    return service.propose_candidates(request.search_state)


@router.post("/evaluate", response_model=CandidateBatchResponse)
def evaluate(request: SearchStateRequest) -> CandidateBatchResponse:
    """Evaluate frontier candidates."""

    service, _ = _service(request.design_task, request.world_model_bundle, request.planning_bundle)
    return service.evaluate_candidates(request.search_state)


@router.post("/plan-next-actions", response_model=ActionPlanResponse)
def plan_next_actions(request: SearchStateRequest) -> ActionPlanResponse:
    """Plan a short action chain using world-model rollout."""

    service, _ = _service(request.design_task, request.world_model_bundle, request.planning_bundle)
    return service.plan_next_actions(request.search_state)


@router.post("/select-for-simulation", response_model=SimulationSelectionResponse)
def select_for_simulation(request: SearchStateRequest) -> SimulationSelectionResponse:
    """Select candidates for real simulation."""

    service, _ = _service(request.design_task, request.world_model_bundle, request.planning_bundle)
    return service.select_for_simulation(request.search_state)


@router.post("/ingest-feedback", response_model=SimulationFeedbackResponse)
def ingest_feedback(request: SimulationFeedbackRequest) -> SimulationFeedbackResponse:
    """Ingest simulator feedback into the planning loop."""

    service, _ = _service(request.design_task, request.world_model_bundle, request.planning_bundle)
    return service.ingest_simulation_feedback(request.search_state, request.candidate_id, request.truth_record)


@router.post("/advance-phase", response_model=CandidateBatchResponse)
def advance_phase(request: SearchStateRequest) -> CandidateBatchResponse:
    """Advance the planning phase."""

    service, _ = _service(request.design_task, request.world_model_bundle, request.planning_bundle)
    return service.advance_phase(request.search_state)


@router.post("/should-terminate", response_model=TerminationDecision)
def should_terminate(request: SearchStateRequest) -> TerminationDecision:
    """Check the planner termination policy."""

    service, _ = _service(request.design_task, request.world_model_bundle, request.planning_bundle)
    return service.should_terminate(request.search_state)


@router.post("/best-result", response_model=PlanningBestResult)
def best_result(request: SearchStateRequest) -> PlanningBestResult:
    """Return the best known planning result."""

    service, _ = _service(request.design_task, request.world_model_bundle, request.planning_bundle)
    return service.get_best_result(request.search_state)
