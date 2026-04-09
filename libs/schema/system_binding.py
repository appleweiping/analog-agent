"""Cross-layer system-binding schemas for real verification closure."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from libs.schema.design_task import DesignTask
from libs.schema.simulation import SimulationExecutionResponse
from libs.schema.world_model import (
    CalibrationUpdateResponse,
    DesignAction,
    FeasibilityPrediction,
    MetricsPrediction,
    SimulationValueEstimate,
    TransitionRecord,
    WorldModelBundle,
    WorldState,
)


class WorldModelTruthBindingRequest(BaseModel):
    """Formal request for the Day-2 prediction->truth->calibration cycle."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    candidate_id: str | None = None
    fidelity_level: str = "focused_validation"
    backend_preference: str = "ngspice"
    escalation_reason: str = "world_model_truth_binding"


class WorldModelTruthBindingResponse(BaseModel):
    """Formal response for the Day-2 world-model/system binding cycle."""

    model_config = ConfigDict(extra="forbid")

    world_model_bundle: WorldModelBundle
    world_state: WorldState
    prediction_action: DesignAction
    metrics_prediction: MetricsPrediction
    feasibility_prediction: FeasibilityPrediction
    simulation_value_estimate: SimulationValueEstimate
    transition_record: TransitionRecord
    simulation_execution: SimulationExecutionResponse
    calibration_update: CalibrationUpdateResponse
