"""Schemas for the planning and optimization layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from libs.schema.design_spec import CIRCUIT_FAMILIES
from libs.schema.design_task import TASK_TYPES
from libs.schema.world_model import (
    DesignAction,
    FeasibilityPrediction,
    MetricsPrediction,
    RolloutResponse,
    SimulationValueEstimate,
    TrustAssessment,
    TruthCalibrationRecord,
    WorldModelBundle,
    WorldState,
)

PLANNING_SCHEMA_VERSION = "planning-schema-v1"
PLANNING_PHASES = (
    "feasibility_bootstrapping",
    "performance_refinement",
    "robustness_verification",
    "calibration_recovery",
    "terminated",
)
CANDIDATE_LIFECYCLE_STATUSES = (
    "proposed",
    "frontier",
    "screened_out",
    "queued_for_rollout",
    "queued_for_simulation",
    "verified",
    "rejected",
    "archived",
    "best_feasible",
    "best_infeasible",
)
DOMINANCE_STATUSES = ("unknown", "dominant", "nondominated", "dominated", "boundary_feasible", "boundary_infeasible")
PROPOSAL_SOURCES = ("initial_seed", "planner_mutation", "world_model_rollout", "restart_policy", "simulation_feedback", "topology_policy")
SEARCH_PROVENANCE_TYPES = ("initialized", "candidate_proposal", "candidate_evaluation", "simulation_feedback", "phase_transition", "restarted")
SELECTION_DECISIONS = ("keep", "defer", "simulate", "prioritize", "drop")
TERMINATION_REASONS = (
    "verified_solution_ready",
    "budget_exhausted",
    "stagnation_limit",
    "phase_goal_reached",
    "world_model_safety_block",
    "manual_stop",
)
TRACE_OUTCOMES = (
    "candidate_proposed",
    "candidate_evaluated",
    "action_planned",
    "simulation_selected",
    "simulation_feedback_ingested",
    "phase_advanced",
    "terminated",
)
VALIDATION_ERROR_CODES = (
    "schema_failure",
    "world_model_binding_failure",
    "unsupported_family",
    "unsupported_task_type",
    "search_state_failure",
    "candidate_lifecycle_failure",
    "budget_state_failure",
    "phase_state_failure",
    "trace_integrity_failure",
    "serving_contract_failure",
)
ACCEPTANCE_FAILURE_CODES = (
    "schema_failure",
    "decision_quality_failure",
    "budget_efficiency_failure",
    "feasibility_progression_failure",
    "world_model_safety_failure",
    "traceability_failure",
    "phase_transition_failure",
    "search_stagnation",
    "budget_misallocation",
    "trust_violation",
    "candidate_misranking",
    "regression_drift",
)


def _ordered_unique(values: list[str], order: tuple[str, ...] | None = None) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    if not order:
        return deduped
    positions = {value: index for index, value in enumerate(order)}
    return sorted(deduped, key=lambda item: (positions.get(item, len(positions)), item))


class WorldModelBinding(BaseModel):
    """Formal binding between the planner and the world-model layer."""

    model_config = ConfigDict(extra="forbid")

    world_model_id: str
    parent_task_id: str
    supported_circuit_families: list[str] = Field(default_factory=list)
    supported_task_types: list[str] = Field(default_factory=list)
    service_methods: list[str] = Field(default_factory=list)
    trust_policy_ref: str
    calibration_readiness: str

    @field_validator("supported_circuit_families")
    @classmethod
    def validate_families(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in CIRCUIT_FAMILIES]
        if invalid:
            raise ValueError(f"unsupported circuit families: {invalid}")
        return _ordered_unique(values, CIRCUIT_FAMILIES)

    @field_validator("supported_task_types")
    @classmethod
    def validate_task_types(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in TASK_TYPES]
        if invalid:
            raise ValueError(f"unsupported task types: {invalid}")
        return _ordered_unique(values, TASK_TYPES)

    @field_validator("service_methods")
    @classmethod
    def dedupe_methods(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SearchPolicy(BaseModel):
    """Formal strategy expression for the planning layer."""

    model_config = ConfigDict(extra="forbid")

    exploration_policy: str
    exploitation_policy: str
    feasibility_policy: str
    rollout_policy: str
    restart_policy: str
    local_refinement_policy: str
    topology_policy: str


class SearchStateSchema(BaseModel):
    """Schema contract for SearchState."""

    model_config = ConfigDict(extra="forbid")

    required_fields: list[str] = Field(default_factory=list)
    phase_values: list[str] = Field(default_factory=lambda: list(PLANNING_PHASES))
    provenance_values: list[str] = Field(default_factory=lambda: list(SEARCH_PROVENANCE_TYPES))

    @field_validator("required_fields", "phase_values", "provenance_values")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class CandidateSchema(BaseModel):
    """Schema contract for CandidateRecord."""

    model_config = ConfigDict(extra="forbid")

    required_fields: list[str] = Field(default_factory=list)
    lifecycle_values: list[str] = Field(default_factory=lambda: list(CANDIDATE_LIFECYCLE_STATUSES))
    dominance_values: list[str] = Field(default_factory=lambda: list(DOMINANCE_STATUSES))
    proposal_sources: list[str] = Field(default_factory=lambda: list(PROPOSAL_SOURCES))

    @field_validator("required_fields", "lifecycle_values", "dominance_values", "proposal_sources")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class TraceSchema(BaseModel):
    """Schema contract for OptimizationTrace."""

    model_config = ConfigDict(extra="forbid")

    required_fields: list[str] = Field(default_factory=list)
    outcome_tags: list[str] = Field(default_factory=lambda: list(TRACE_OUTCOMES))

    @field_validator("required_fields", "outcome_tags")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class RolloutConfig(BaseModel):
    """Formal rollout configuration for world-model imagination."""

    model_config = ConfigDict(extra="forbid")

    horizon: int
    beam_width: int
    max_branching_factor: int
    require_rollout_ready: bool = True
    allow_feasibility_rollout: bool = True


class SelectionPolicy(BaseModel):
    """Formal candidate-selection policy."""

    model_config = ConfigDict(extra="forbid")

    ranking_source: Literal["world_model_ranker"] = "world_model_ranker"
    prioritize_feasibility: bool = True
    prioritize_diversity: bool = False
    min_feasible_probability: float = 0.0
    uncertainty_penalty: float = 0.25
    simulation_value_weight: float = 0.25

    @field_validator("min_feasible_probability", "uncertainty_penalty", "simulation_value_weight")
    @classmethod
    def validate_weights(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("selection-policy scores must be within [0, 1]")
        return round(float(value), 4)


class EscalationPolicy(BaseModel):
    """Formal simulator-escalation policy."""

    model_config = ConfigDict(extra="forbid")

    min_simulation_value: float
    allow_must_escalate_override: bool = True
    max_batch_size: int
    allowed_service_tiers: list[str] = Field(default_factory=list)

    @field_validator("min_simulation_value")
    @classmethod
    def validate_simulation_value(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("min_simulation_value must be within [0, 1]")
        return round(float(value), 4)

    @field_validator("allowed_service_tiers")
    @classmethod
    def dedupe_tiers(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class BudgetController(BaseModel):
    """Formal budget-control policy."""

    model_config = ConfigDict(extra="forbid")

    max_proxy_evaluations: int
    max_rollouts: int
    max_real_simulations: int
    max_calibration_updates: int
    batch_size: int
    per_phase_proxy_caps: dict[str, int] = Field(default_factory=dict)

    @field_validator("per_phase_proxy_caps")
    @classmethod
    def validate_phase_caps(cls, values: dict[str, int]) -> dict[str, int]:
        invalid = [key for key in values if key not in PLANNING_PHASES]
        if invalid:
            raise ValueError(f"unsupported phase caps: {invalid}")
        return dict(sorted(values.items()))


class PhaseRule(BaseModel):
    """One formal phase-transition rule."""

    model_config = ConfigDict(extra="forbid")

    from_phase: str
    to_phase: str
    trigger: str

    @field_validator("from_phase", "to_phase")
    @classmethod
    def validate_phase(cls, value: str) -> str:
        if value not in PLANNING_PHASES:
            raise ValueError(f"unsupported phase: {value}")
        return value


class PhaseController(BaseModel):
    """Formal phase-control policy."""

    model_config = ConfigDict(extra="forbid")

    initial_phase: str
    allowed_transitions: list[PhaseRule] = Field(default_factory=list)

    @field_validator("initial_phase")
    @classmethod
    def validate_initial_phase(cls, value: str) -> str:
        if value not in PLANNING_PHASES:
            raise ValueError(f"unsupported initial phase: {value}")
        return value


class TerminationPolicy(BaseModel):
    """Formal termination policy."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int
    stagnation_patience: int
    stop_on_verified_feasible: bool = True
    require_verification_phase: bool = False
    min_feasible_count_before_terminate: int = 1


class ServiceMethodSpec(BaseModel):
    """Formal service-method signature."""

    model_config = ConfigDict(extra="forbid")

    input_schema: str
    output_schema: str
    deterministic: bool = True


class PlanningServingContract(BaseModel):
    """Formal serving contract for the planning layer."""

    model_config = ConfigDict(extra="forbid")

    initialize_search: ServiceMethodSpec
    propose_candidates: ServiceMethodSpec
    evaluate_candidates: ServiceMethodSpec
    plan_next_actions: ServiceMethodSpec
    select_for_simulation: ServiceMethodSpec
    ingest_simulation_feedback: ServiceMethodSpec
    advance_phase: ServiceMethodSpec
    should_terminate: ServiceMethodSpec
    get_best_result: ServiceMethodSpec


class PlanningMetadata(BaseModel):
    """Audit metadata for the fourth layer."""

    model_config = ConfigDict(extra="forbid")

    created_by_layer: Literal["planning_layer"] = "planning_layer"
    compile_timestamp: str
    source_task_signature: str
    source_world_model_signature: str
    implementation_version: str
    assumptions: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @field_validator("assumptions", "provenance")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class PlanningValidationIssue(BaseModel):
    """Validation issue for planning-layer compilation or execution."""

    model_config = ConfigDict(extra="forbid")

    code: str
    path: str
    message: str
    severity: Literal["error", "warning"]

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if value not in VALIDATION_ERROR_CODES:
            raise ValueError(f"unsupported validation code: {value}")
        return value


class PlanningValidationStatus(BaseModel):
    """Embedded validation state for planning objects."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    errors: list[PlanningValidationIssue] = Field(default_factory=list)
    warnings: list[PlanningValidationIssue] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)
    repair_history: list[str] = Field(default_factory=list)
    completeness_score: float

    @field_validator("unresolved_dependencies", "repair_history")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)

    @field_validator("completeness_score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("completeness_score must be within [0, 1]")
        return round(float(value), 4)


class CandidateEvaluationEvent(BaseModel):
    """One structured candidate evaluation event."""

    model_config = ConfigDict(extra="forbid")

    event_type: Literal["predicted", "ranked", "selected_for_simulation", "simulated", "calibrated"]
    timestamp: str
    notes: list[str] = Field(default_factory=list)

    @field_validator("notes")
    @classmethod
    def dedupe_notes(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class CandidateDecisionEvent(BaseModel):
    """One structured candidate decision event."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["keep", "defer", "simulate", "prioritize", "drop"]
    reason: str
    timestamp: str


class CandidateRecord(BaseModel):
    """Formal lifecycle record for one planner candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    task_id: str
    world_state_ref: str
    world_state_snapshot: WorldState
    parent_candidate_id: str | None = None
    generation_depth: int
    proposal_source: Literal["initial_seed", "planner_mutation", "world_model_rollout", "restart_policy", "simulation_feedback", "topology_policy"]
    proposal_action_chain: list[DesignAction] = Field(default_factory=list)
    predicted_metrics: MetricsPrediction | None = None
    predicted_feasibility: FeasibilityPrediction | None = None
    predicted_uncertainty: TrustAssessment | None = None
    simulation_value_estimate: SimulationValueEstimate | None = None
    priority_score: float = 0.0
    dominance_status: Literal["unknown", "dominant", "nondominated", "dominated", "boundary_feasible", "boundary_infeasible"] = "unknown"
    lifecycle_state: Literal[
        "proposed",
        "frontier",
        "screened_out",
        "queued_for_rollout",
        "queued_for_simulation",
        "verified",
        "rejected",
        "archived",
        "best_feasible",
        "best_infeasible",
    ] = "proposed"
    lifecycle_status: Literal[
        "proposed",
        "frontier",
        "screened_out",
        "queued_for_rollout",
        "queued_for_simulation",
        "verified",
        "rejected",
        "archived",
        "best_feasible",
        "best_infeasible",
    ] = "proposed"
    evaluation_history: list[CandidateEvaluationEvent] = Field(default_factory=list)
    decision_history: list[CandidateDecisionEvent] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)

    @field_validator("artifact_refs")
    @classmethod
    def dedupe_artifacts(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)

    @model_validator(mode="before")
    @classmethod
    def sync_lifecycle_fields(cls, values):
        if not isinstance(values, dict):
            return values
        lifecycle = values.get("lifecycle_state", values.get("lifecycle_status", "proposed"))
        values["lifecycle_state"] = lifecycle
        values["lifecycle_status"] = lifecycle
        return values


class CandidatePoolState(BaseModel):
    """Formal pool-state over all active and archived candidates."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateRecord] = Field(default_factory=list)
    active_candidate_ids: list[str] = Field(default_factory=list)
    archived_candidate_ids: list[str] = Field(default_factory=list)
    discarded_candidate_ids: list[str] = Field(default_factory=list)

    @field_validator("active_candidate_ids", "archived_candidate_ids", "discarded_candidate_ids")
    @classmethod
    def dedupe_ids(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class FrontierState(BaseModel):
    """Formal frontier-state for expansion planning."""

    model_config = ConfigDict(extra="forbid")

    frontier_candidate_ids: list[str] = Field(default_factory=list)
    expansion_round: int = 0
    max_frontier_size: int = 0

    @field_validator("frontier_candidate_ids")
    @classmethod
    def dedupe_ids(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class BudgetState(BaseModel):
    """Formal runtime budget state."""

    model_config = ConfigDict(extra="forbid")

    proxy_evaluation_budget: int
    proxy_evaluations_used: int
    rollout_budget: int
    rollouts_used: int
    simulation_budget: int
    simulations_used: int
    calibration_budget: int
    calibrations_used: int
    budget_pressure: float = 0.0

    @field_validator("budget_pressure")
    @classmethod
    def validate_pressure(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("budget_pressure must be within [0, 1]")
        return round(float(value), 4)


class PhaseState(BaseModel):
    """Formal runtime phase state."""

    model_config = ConfigDict(extra="forbid")

    current_phase: Literal[
        "feasibility_bootstrapping",
        "performance_refinement",
        "robustness_verification",
        "calibration_recovery",
        "terminated",
    ]
    phase_iteration: int = 0
    successful_phase_transitions: int = 0
    stagnation_counter: int = 0
    last_phase_change_step: int = 0


class CandidateSummary(BaseModel):
    """Compact candidate summary embedded in SearchState."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    world_state_ref: str
    feasible_probability: float
    priority_score: float
    lifecycle_status: str

    @field_validator("feasible_probability")
    @classmethod
    def validate_probability(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("feasible_probability must be within [0, 1]")
        return round(float(value), 4)


class StrategyContext(BaseModel):
    """Structured runtime strategy context."""

    model_config = ConfigDict(extra="forbid")

    active_policy: str
    exploration_enabled: bool
    rollout_enabled: bool
    last_selected_candidate_id: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("notes")
    @classmethod
    def dedupe_notes(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class RiskContext(BaseModel):
    """Structured runtime risk context."""

    model_config = ConfigDict(extra="forbid")

    average_uncertainty: float
    max_ood_score: float
    trust_alert_count: int
    calibration_required: bool = False
    active_failure_modes: list[str] = Field(default_factory=list)

    @field_validator("average_uncertainty", "max_ood_score")
    @classmethod
    def validate_scores(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("risk-context scores must be within [0, 1]")
        return round(float(value), 4)

    @field_validator("active_failure_modes")
    @classmethod
    def dedupe_failures(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SearchProvenance(BaseModel):
    """Structured provenance for SearchState."""

    model_config = ConfigDict(extra="forbid")

    source: Literal[
        "initialized",
        "candidate_proposal",
        "candidate_evaluation",
        "simulation_feedback",
        "phase_transition",
        "restarted",
    ]
    created_at: str
    artifact_refs: list[str] = Field(default_factory=list)

    @field_validator("artifact_refs")
    @classmethod
    def dedupe_artifacts(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class WorldModelQueryRecord(BaseModel):
    """Structured world-model query used in optimization traces."""

    model_config = ConfigDict(extra="forbid")

    method: str
    state_id: str
    action_id: str | None = None
    timestamp: str


class SimulationDecision(BaseModel):
    """Structured simulation decision embedded in traces and selections."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["keep", "defer", "simulate", "prioritize", "drop"]
    candidate_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @field_validator("candidate_ids", "reasons")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class OptimizationTrace(BaseModel):
    """Formal optimization trace emitted at each search step."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    task_id: str
    episode_id: str
    step_index: int
    search_state_snapshot: str
    selected_candidate_id: str | None = None
    executed_action_chain: list[DesignAction] = Field(default_factory=list)
    world_model_queries: list[WorldModelQueryRecord] = Field(default_factory=list)
    simulation_decision: SimulationDecision
    simulation_result_ref: str | None = None
    reward_or_progress_signal: float = 0.0
    decision_rationale: list[str] = Field(default_factory=list)
    trust_snapshot: TrustAssessment
    budget_snapshot: BudgetState
    outcome_tag: Literal[
        "candidate_proposed",
        "candidate_evaluated",
        "action_planned",
        "simulation_selected",
        "simulation_feedback_ingested",
        "phase_advanced",
        "terminated",
    ]

    @field_validator("decision_rationale")
    @classmethod
    def dedupe_rationale(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SearchState(BaseModel):
    """Formal execution state for the planning layer."""

    model_config = ConfigDict(extra="forbid")

    search_id: str
    task_id: str
    episode_id: str
    current_world_state: WorldState
    candidate_pool_state: CandidatePoolState
    frontier_state: FrontierState
    evaluated_state_refs: list[str] = Field(default_factory=list)
    pending_simulation_refs: list[str] = Field(default_factory=list)
    budget_state: BudgetState
    phase_state: PhaseState
    best_known_feasible: CandidateSummary | None = None
    best_known_infeasible: CandidateSummary | None = None
    strategy_context: StrategyContext
    risk_context: RiskContext
    provenance: SearchProvenance
    trace_log: list[OptimizationTrace] = Field(default_factory=list)

    @field_validator("evaluated_state_refs", "pending_simulation_refs")
    @classmethod
    def dedupe_refs(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class TerminationDecision(BaseModel):
    """Structured search-termination decision."""

    model_config = ConfigDict(extra="forbid")

    should_terminate: bool
    reason: Literal[
        "verified_solution_ready",
        "budget_exhausted",
        "stagnation_limit",
        "phase_goal_reached",
        "world_model_safety_block",
        "manual_stop",
    ] | None = None
    recommended_action: Literal["continue", "terminate", "recalibrate", "simulate_more"] = "continue"


class PlanningBestResult(BaseModel):
    """Structured best-result summary returned to orchestrators."""

    model_config = ConfigDict(extra="forbid")

    candidate: CandidateRecord | None = None
    phase: str
    summary: dict[str, float | int | str | bool] = Field(default_factory=dict)
    termination_decision: TerminationDecision


class PlanningBundle(BaseModel):
    """Top-level formal bundle for planning and optimization."""

    model_config = ConfigDict(extra="forbid")

    planning_id: str
    schema_version: str = PLANNING_SCHEMA_VERSION
    parent_task_id: str
    world_model_binding: WorldModelBinding
    search_policy: SearchPolicy
    search_state_schema: SearchStateSchema
    candidate_schema: CandidateSchema
    rollout_config: RolloutConfig
    selection_policy: SelectionPolicy
    escalation_policy: EscalationPolicy
    budget_controller: BudgetController
    phase_controller: PhaseController
    termination_policy: TerminationPolicy
    trace_schema: TraceSchema
    serving_contract: PlanningServingContract
    metadata: PlanningMetadata
    validation_status: PlanningValidationStatus


class PlanningCompilationReport(BaseModel):
    """Structured compile report for the planning layer."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["compiled", "compiled_with_warnings", "invalid"]
    derived_fields: list[str] = Field(default_factory=list)
    validation_errors: list[PlanningValidationIssue] = Field(default_factory=list)
    validation_warnings: list[PlanningValidationIssue] = Field(default_factory=list)
    acceptance_summary: dict[str, float | int | str] = Field(default_factory=dict)

    @field_validator("derived_fields")
    @classmethod
    def dedupe_fields(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class PlanningCompileResponse(BaseModel):
    """Top-level compile response for PlanningBundle."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["compiled", "compiled_with_warnings", "invalid"]
    planning_bundle: PlanningBundle | None = None
    report: PlanningCompilationReport


class SearchInitializationResponse(BaseModel):
    """Search initialization response."""

    model_config = ConfigDict(extra="forbid")

    planning_bundle: PlanningBundle
    search_state: SearchState


class CandidateBatchResponse(BaseModel):
    """Response for proposal and evaluation stages."""

    model_config = ConfigDict(extra="forbid")

    search_state: SearchState
    candidates: list[CandidateRecord] = Field(default_factory=list)
    traces: list[OptimizationTrace] = Field(default_factory=list)


class ActionPlanResponse(BaseModel):
    """Response for lookahead planning."""

    model_config = ConfigDict(extra="forbid")

    search_state: SearchState
    anchor_candidate_id: str | None = None
    action_chain: list[DesignAction] = Field(default_factory=list)
    rollout_response: RolloutResponse | None = None
    traces: list[OptimizationTrace] = Field(default_factory=list)


class SimulationSelectionResponse(BaseModel):
    """Response for simulator-escalation selection."""

    model_config = ConfigDict(extra="forbid")

    search_state: SearchState
    selected_candidates: list[CandidateRecord] = Field(default_factory=list)
    traces: list[OptimizationTrace] = Field(default_factory=list)


class SimulationFeedbackResponse(BaseModel):
    """Response for ingesting simulator feedback."""

    model_config = ConfigDict(extra="forbid")

    search_state: SearchState
    updated_world_model_bundle: WorldModelBundle
    traces: list[OptimizationTrace] = Field(default_factory=list)


class PlanningAcceptanceFailureRecord(BaseModel):
    """Structured failure taxonomy entry for planning acceptance."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if value not in ACCEPTANCE_FAILURE_CODES:
            raise ValueError(f"unsupported acceptance failure code: {value}")
        return value


class PlanningAcceptanceSummary(BaseModel):
    """Aggregated acceptance metrics for the planning layer."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int
    schema_validity_rate: float
    decision_quality_rate: float
    budget_efficiency_rate: float
    feasibility_progression_rate: float
    world_model_safety_rate: float
    traceability_rate: float
    average_proxy_evaluations: float
    average_simulations: float
    feasible_hit_rate: float
    failures: list[PlanningAcceptanceFailureRecord] = Field(default_factory=list)

    @field_validator(
        "schema_validity_rate",
        "decision_quality_rate",
        "budget_efficiency_rate",
        "feasibility_progression_rate",
        "world_model_safety_rate",
        "traceability_rate",
        "feasible_hit_rate",
    )
    @classmethod
    def validate_rates(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("acceptance rates must be within [0, 1]")
        return round(float(value), 4)
