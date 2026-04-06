"""World-model API routes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from libs.schema.design_task import DesignTask
from libs.schema.world_model import (
    CalibrationUpdateResponse,
    DesignAction,
    FeasibilityPrediction,
    MetricsPrediction,
    TransitionPrediction,
    TruthCalibrationRecord,
    WorldModelCompileResponse,
    WorldState,
)
from libs.world_model.compiler import compile_world_model_bundle
from libs.world_model.service import WorldModelService
from libs.world_model.state_builder import build_world_state

router = APIRouter(prefix="/world-model", tags=["world-model"])


class BuildStateRequest(BaseModel):
    """Request payload for building a WorldState."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    parameter_values: dict[str, float | int | str | bool] = Field(default_factory=dict)
    corner: str | None = None
    temperature_c: float | None = None
    analysis_fidelity: str = "quick_screening"
    analysis_intent: str | None = None
    output_load_ohm: float | None = None


class PredictMetricsRequest(BaseModel):
    """Request payload for metric prediction."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    world_state: WorldState


class PredictTransitionRequest(BaseModel):
    """Request payload for transition prediction."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    world_state: WorldState
    action: DesignAction


class CalibrateRequest(BaseModel):
    """Request payload for calibration."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    world_state: WorldState
    truth_record: TruthCalibrationRecord


@router.post("/compile", response_model=WorldModelCompileResponse)
def compile_bundle(design_task: DesignTask) -> WorldModelCompileResponse:
    """Compile a DesignTask into a formal WorldModelBundle."""

    return compile_world_model_bundle(design_task)


@router.post("/build-state", response_model=WorldState)
def build_state(request: BuildStateRequest) -> WorldState:
    """Build a formal WorldState from a DesignTask."""

    return build_world_state(
        request.design_task,
        parameter_values=request.parameter_values,
        corner=request.corner,
        temperature_c=request.temperature_c,
        analysis_fidelity=request.analysis_fidelity,
        analysis_intent=request.analysis_intent,
        output_load_ohm=request.output_load_ohm,
    )


@router.post("/predict-metrics", response_model=MetricsPrediction)
def predict_metrics(request: PredictMetricsRequest) -> MetricsPrediction:
    """Predict task-aligned metrics for a WorldState."""

    compiled = compile_world_model_bundle(request.design_task)
    if compiled.world_model_bundle is None:
        raise ValueError("world model bundle failed to compile")
    return WorldModelService(compiled.world_model_bundle, request.design_task).predict_metrics(request.world_state)


@router.post("/predict-feasibility", response_model=FeasibilityPrediction)
def predict_feasibility(request: PredictMetricsRequest) -> FeasibilityPrediction:
    """Predict feasibility and risk for a WorldState."""

    compiled = compile_world_model_bundle(request.design_task)
    if compiled.world_model_bundle is None:
        raise ValueError("world model bundle failed to compile")
    return WorldModelService(compiled.world_model_bundle, request.design_task).predict_feasibility(request.world_state)


@router.post("/predict-transition", response_model=TransitionPrediction)
def predict_transition(request: PredictTransitionRequest) -> TransitionPrediction:
    """Predict the next state under a design action."""

    compiled = compile_world_model_bundle(request.design_task)
    if compiled.world_model_bundle is None:
        raise ValueError("world model bundle failed to compile")
    return WorldModelService(compiled.world_model_bundle, request.design_task).predict_transition(request.world_state, request.action)


@router.post("/calibrate", response_model=CalibrationUpdateResponse)
def calibrate(request: CalibrateRequest) -> CalibrationUpdateResponse:
    """Update bundle calibration state with real simulation truth."""

    compiled = compile_world_model_bundle(request.design_task)
    if compiled.world_model_bundle is None:
        raise ValueError("world model bundle failed to compile")
    return WorldModelService(compiled.world_model_bundle, request.design_task).calibrate_with_truth(request.world_state, request.truth_record)
