"""Cross-layer system-binding schemas for real verification closure."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from libs.schema.design_task import DesignTask
from libs.schema.planning import CandidateRecord, PlanningBestResult, PlanningBundle, SearchState
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
    fidelity_level: str = "focused_truth"
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


class PlanningTruthLoopRequest(BaseModel):
    """Formal request for the Day-3 selective-simulation loop."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    max_steps: int = 3
    fidelity_level: str = "quick_truth"
    backend_preference: str = "ngspice"
    escalation_reason: str = "planning_truth_loop"


class PlanningStepSummary(BaseModel):
    """Structured per-step summary for the selective planner loop."""

    model_config = ConfigDict(extra="forbid")

    step_index: int
    candidate_pool_size: int
    selected_for_simulation: list[str] = Field(default_factory=list)
    simulated_candidate_ids: list[str] = Field(default_factory=list)
    requested_fidelity: dict[str, str] = Field(default_factory=dict)
    best_candidate_id: str | None = None
    best_priority_score: float = 0.0
    simulation_calls_used: int = 0


class BaselineComparisonSummary(BaseModel):
    """Comparison summary between selective planning and fuller simulation coverage."""

    model_config = ConfigDict(extra="forbid")

    selective_simulation_calls: int
    baseline_full_simulation_calls: int
    simulations_saved: int
    selective_best_candidate_id: str | None = None
    baseline_best_candidate_id: str | None = None
    selective_best_truth_metric: float = 0.0
    baseline_best_truth_metric: float = 0.0
    selective_quality_ratio: float = 0.0


class PlanningTruthLoopResponse(BaseModel):
    """Formal Day-3 response for planner-driven truth closure."""

    model_config = ConfigDict(extra="forbid")

    planning_bundle: PlanningBundle
    world_model_bundle: WorldModelBundle
    final_search_state: SearchState
    best_result: PlanningBestResult
    best_candidate: CandidateRecord | None = None
    simulation_executions: list[SimulationExecutionResponse] = Field(default_factory=list)
    step_summaries: list[PlanningStepSummary] = Field(default_factory=list)
    comparison_summary: BaselineComparisonSummary
