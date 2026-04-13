"""Cross-layer system-binding schemas for real verification closure."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from libs.schema.design_task import DesignTask
from libs.schema.experiment import ExperimentSuiteResult, MethodComparisonResult
from libs.schema.memory import EpisodeMemoryRecord, MemoryBundle
from libs.schema.planning import CandidateRecord, PlanningBestResult, PlanningBundle, SearchState
from libs.schema.simulation import SimulationExecutionResponse
from libs.schema.stats import StatsAggregationResult, VerificationStatsRecord
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


class AcceptanceTaskConfig(BaseModel):
    """Formal request payload for the Day-8 end-to-end acceptance runner."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    max_steps: int = 3
    default_fidelity: str = "quick_truth"
    backend_preference: str = "ngspice"
    escalation_reason: str = "full_system_acceptance"


class ArtifactTrace(BaseModel):
    """Structured artifact lineage record for one E2E acceptance run."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_type: str
    artifact_path: str
    candidate_id: str
    simulation_id: str
    verification_result_id: str
    model_reference: str
    truth_level: str
    validation_status: str


class CrossLayerTrace(BaseModel):
    """Structured lineage trace across L3-L4-L5-L6 for one candidate execution."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    search_id: str
    planning_id: str
    world_model_id: str
    candidate_id: str
    parent_candidate_id: str | None = None
    world_state_id: str
    simulation_request_id: str
    simulation_id: str
    verification_result_id: str
    calibration_id: str
    episode_memory_id: str | None = None
    requested_fidelity: str
    executed_fidelity: str
    truth_level: str
    validation_status: str
    planner_lifecycle_update: str
    memory_recorded: bool = False


class StepTrace(BaseModel):
    """Structured per-step trace for the Day-8 E2E runner."""

    model_config = ConfigDict(extra="forbid")

    step_index: int
    phase_before: str
    phase_after: str
    candidate_pool_size_before: int
    candidate_pool_size_after: int
    selected_candidate_ids: list[str] = Field(default_factory=list)
    requested_fidelity: dict[str, str] = Field(default_factory=dict)
    verification_result_ids: list[str] = Field(default_factory=list)
    measurement_statuses: dict[str, list[str]] = Field(default_factory=dict)
    validation_states: dict[str, str] = Field(default_factory=dict)
    planner_updates: dict[str, str] = Field(default_factory=dict)
    calibration_actions: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class AcceptanceSummary(BaseModel):
    """Structured system-level acceptance summary."""

    model_config = ConfigDict(extra="forbid")

    schema_completeness_ok: bool
    backend_execution_validity_ok: bool
    measurement_correctness_ok: bool
    fidelity_correctness_ok: bool
    validation_correctness_ok: bool
    feedback_consistency_ok: bool
    system_closed_loop_established: bool
    simulation_execution_count: int
    step_count: int
    memory_episode_count: int
    notes: list[str] = Field(default_factory=list)


class SystemAcceptanceResult(BaseModel):
    """Top-level Day-8 E2E acceptance result."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    world_model_id: str
    planning_id: str
    memory_id: str
    search_id: str
    best_candidate_id: str | None = None
    best_feasible_found: bool = False
    final_verification_result_id: str | None = None
    episode_memory_id: str | None = None
    memory_bundle: MemoryBundle
    episode_record: EpisodeMemoryRecord | None = None
    step_traces: list[StepTrace] = Field(default_factory=list)
    cross_layer_traces: list[CrossLayerTrace] = Field(default_factory=list)
    artifact_traces: list[ArtifactTrace] = Field(default_factory=list)
    verification_stats: list[VerificationStatsRecord] = Field(default_factory=list)
    stats_summary: StatsAggregationResult | None = None
    acceptance_summary: AcceptanceSummary


class FinalSystemCheckSummary(BaseModel):
    """Structured Day-12 submission-ready closure check."""

    model_config = ConfigDict(extra="forbid")

    l5_real_backend_primary: bool
    l5_quick_truth_established: bool
    l5_focused_truth_established: bool
    l5_measurement_contract_stable: bool
    l3_consumes_real_calibration_feedback: bool
    l3_world_model_is_calibratable: bool
    l4_updates_search_from_verification: bool
    l4_budget_and_fidelity_aware: bool
    l6_persists_real_episode_memory: bool
    l6_distinguishes_truth_levels: bool
    l2_to_l6_closed_loop: bool
    acceptance_suite_available: bool
    stats_foundation_available: bool
    ota_v1_acceptance_ok: bool
    ota_v1_experiment_ok: bool
    stats_export_ok: bool
    method_comparison_ok: bool
    current_truth_level: str
    real_pdk_connected: bool
    multi_task_supported: bool
    submission_ready: bool
    closure_statement: str
    notes: list[str] = Field(default_factory=list)


class SystemClosureResult(BaseModel):
    """Top-level Day-12 submission-ready freeze result."""

    model_config = ConfigDict(extra="forbid")

    acceptance_result: SystemAcceptanceResult
    baseline_suite: ExperimentSuiteResult
    methodology_suite: ExperimentSuiteResult
    method_conclusions: MethodComparisonResult
    final_check_summary: FinalSystemCheckSummary
