"""Schemas for the world model layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from libs.schema.design_spec import CIRCUIT_FAMILIES
from libs.schema.design_task import TASK_TYPES

WORLD_MODEL_SCHEMA_VERSION = "world-model-schema-v1"
ANALYSIS_FIDELITIES = ("quick_screening", "partial_simulation", "full_ground_truth")
STATE_PROVENANCE_TYPES = ("offline_dataset", "real_simulation", "trajectory_replay", "model_rollout", "hybrid_calibration")
SOURCE_STAGES = ("initial", "predicted", "simulated")
FIELD_PROVENANCE_TYPES = ("truth", "prediction", "missing", "calibrated")
ACTION_FAMILIES = (
    "parameter_update",
    "coupled_update",
    "bias_reallocation",
    "compensation_adjustment",
    "topology_switch",
    "template_slot_mutation",
    "fidelity_upgrade",
    "candidate_prune",
)
ACTION_OPERATORS = ("set", "shift", "scale", "swap", "freeze", "unfreeze", "promote_fidelity", "drop_candidate")
ACTION_TARGET_KINDS = ("variable", "variable_group", "topology_slot", "evaluation_strategy", "candidate")
EXPECTED_SCOPES = ("operating_point", "ac_stability", "power", "noise", "area", "evaluation_cost", "feasibility")
ACTION_SOURCES = ("planner", "critic", "expert_rule", "warm_start_policy", "manual")
TRUST_LEVELS = ("high", "medium", "low", "blocked")
SERVICE_TIERS = ("screening_only", "ranking_ready", "rollout_ready", "must_escalate", "hard_block")
MODEL_HEAD_NAMES = (
    "metric_predictor",
    "feasibility_predictor",
    "transition_predictor",
    "uncertainty_estimator",
    "trust_estimator",
    "simulation_value_estimator",
)
CALIBRATION_READINESS = ("experimental", "screening", "rollout_ready")
UPDATE_POLICIES = ("offline_retrain", "periodic_incremental", "online_finetune")
LABEL_QUALITIES = ("high_fidelity", "partial_fidelity", "pseudo_label")
VALIDATION_ERROR_CODES = (
    "schema_failure",
    "unsupported_family",
    "unsupported_task_type",
    "state_domain_error",
    "state_context_error",
    "action_validity_error",
    "serving_contract_error",
    "ood_risk_error",
    "calibration_state_error",
    "training_state_error",
)
ACCEPTANCE_FAILURE_CODES = (
    "schema_failure",
    "service_contract_failure",
    "metric_error_excessive",
    "transition_direction_failure",
    "constraint_margin_failure",
    "uncertainty_miscalibration",
    "ood_trust_failure",
    "calibration_update_failure",
    "planning_utility_failure",
    "regression_drift",
)
WORLD_MODEL_METRICS = (
    "dc_gain_db",
    "gbw_hz",
    "phase_margin_deg",
    "slew_rate_v_per_us",
    "power_w",
    "area_um2",
    "noise_nv_per_sqrt_hz",
    "input_referred_noise_nv_per_sqrt_hz",
    "output_swing_v",
    "input_common_mode_v",
    "psrr_db",
    "line_regulation_mv_per_v",
    "load_regulation_mv_per_ma",
    "offset_mv",
    "delay_ns",
    "temperature_coefficient_ppm_per_c",
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


class EncodingSpec(BaseModel):
    """Structured encoding specification for one schema channel."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    fields: list[str] = Field(default_factory=list)
    version: str = "1.0"

    @field_validator("fields")
    @classmethod
    def dedupe_fields(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class StateSchema(BaseModel):
    """Formal state-schema contract."""

    model_config = ConfigDict(extra="forbid")

    topology_encoding: EncodingSpec
    parameter_encoding: EncodingSpec
    environment_encoding: EncodingSpec
    op_encoding: EncodingSpec
    history_encoding: EncodingSpec
    uncertainty_encoding: EncodingSpec


class ActionSchema(BaseModel):
    """Formal action-schema contract."""

    model_config = ConfigDict(extra="forbid")

    target_selection: EncodingSpec
    operator_encoding: EncodingSpec
    magnitude_encoding: EncodingSpec
    scope_encoding: EncodingSpec
    validity_encoding: EncodingSpec


class MetricEstimate(BaseModel):
    """Structured metric estimate used in observations and predictions."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    value: float
    lower_bound: float
    upper_bound: float
    uncertainty: float
    trust_level: Literal["high", "medium", "low", "blocked"]
    source: Literal["truth", "prediction", "missing", "calibrated"]

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, value: str) -> str:
        if value not in WORLD_MODEL_METRICS:
            raise ValueError(f"unsupported metric: {value}")
        return value

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if value not in FIELD_PROVENANCE_TYPES:
            raise ValueError(f"unsupported field source: {value}")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> "MetricEstimate":
        if self.lower_bound > self.upper_bound:
            raise ValueError("metric lower_bound must be <= upper_bound")
        return self


class ConstraintObservation(BaseModel):
    """Structured feasibility observation for one constraint group or rule."""

    model_config = ConfigDict(extra="forbid")

    constraint_name: str
    constraint_group: str
    satisfied_probability: float
    margin: float
    violation_severity: float
    source: Literal["truth", "prediction", "missing", "calibrated"]

    @field_validator("satisfied_probability")
    @classmethod
    def validate_probability(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("satisfied_probability must be within [0, 1]")
        return round(float(value), 4)


class GraphStatistics(BaseModel):
    """Compact structural summary for world-state consumption."""

    model_config = ConfigDict(extra="forbid")

    node_count: int
    edge_count: int
    symmetry_group_count: int
    fanout_estimate: float


class TopologyContext(BaseModel):
    """Formal topology anchor inherited from DesignTask."""

    model_config = ConfigDict(extra="forbid")

    topology_mode: Literal["fixed", "template_family", "search_space"]
    template_id: str | None = None
    template_version: str | None = None
    port_names: list[str] = Field(default_factory=list)
    instance_names: list[str] = Field(default_factory=list)
    instance_roles: list[str] = Field(default_factory=list)
    topology_constraints: list[str] = Field(default_factory=list)
    graph_statistics: GraphStatistics

    @field_validator("port_names", "instance_names", "instance_roles", "topology_constraints")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ParameterValue(BaseModel):
    """One design-space variable carried in a world state."""

    model_config = ConfigDict(extra="forbid")

    variable_name: str
    value: float | int | str | bool
    normalized_value: float
    is_frozen: bool
    coupling_group: str | None = None
    is_active: bool = True


class EnvironmentState(BaseModel):
    """Formal environment and evaluation conditions."""

    model_config = ConfigDict(extra="forbid")

    corner: str
    temperature_c: float
    supply_voltage_v: float | None = None
    load_cap_f: float | None = None
    output_load_ohm: float | None = None
    bias_mode: str = "nominal"
    analysis_assumptions: list[str] = Field(default_factory=list)

    @field_validator("analysis_assumptions")
    @classmethod
    def dedupe_assumptions(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class EvaluationContext(BaseModel):
    """Formal evaluation intent for one world-state query."""

    model_config = ConfigDict(extra="forbid")

    analysis_intent: str
    analysis_fidelity: Literal["quick_screening", "partial_simulation", "full_ground_truth"]
    target_metrics: list[str] = Field(default_factory=list)
    constraint_groups: list[str] = Field(default_factory=list)
    objective_mode: str

    @field_validator("target_metrics")
    @classmethod
    def validate_metrics(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in WORLD_MODEL_METRICS]
        if invalid:
            raise ValueError(f"unsupported target metrics: {invalid}")
        return _ordered_unique(values, WORLD_MODEL_METRICS)

    @field_validator("constraint_groups")
    @classmethod
    def dedupe_constraint_groups(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class OperatingPointState(BaseModel):
    """Structured operating-point proxy state."""

    model_config = ConfigDict(extra="forbid")

    node_voltage_summary: dict[str, float] = Field(default_factory=dict)
    device_region_map: dict[str, Literal["cutoff", "subthreshold", "linear", "saturation", "unknown"]] = Field(default_factory=dict)
    gm: dict[str, float] = Field(default_factory=dict)
    gds: dict[str, float] = Field(default_factory=dict)
    ro: dict[str, float] = Field(default_factory=dict)
    drain_current_a: dict[str, float] = Field(default_factory=dict)
    overdrive_v: dict[str, float] = Field(default_factory=dict)
    gm_over_id: dict[str, float] = Field(default_factory=dict)
    branch_balance: dict[str, float] = Field(default_factory=dict)
    pole_zero_summary: dict[str, float] = Field(default_factory=dict)
    stability_proxy: dict[str, float] = Field(default_factory=dict)


class StructuralFeatures(BaseModel):
    """Structured graph and role features for a world state."""

    model_config = ConfigDict(extra="forbid")

    module_roles: list[str] = Field(default_factory=list)
    symmetry_groups: list[list[str]] = Field(default_factory=list)
    key_subcircuits: list[str] = Field(default_factory=list)
    graph_statistics: GraphStatistics

    @field_validator("module_roles", "key_subcircuits")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class HistoryEntry(BaseModel):
    """One recent transition element in the state history."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    metric_deltas: dict[str, float] = Field(default_factory=dict)
    failure_modes: list[str] = Field(default_factory=list)
    timestamp: str

    @field_validator("metric_deltas")
    @classmethod
    def validate_delta_metrics(cls, values: dict[str, float]) -> dict[str, float]:
        invalid = [metric for metric in values if metric not in WORLD_MODEL_METRICS]
        if invalid:
            raise ValueError(f"unsupported metric deltas: {invalid}")
        return dict(sorted(values.items()))

    @field_validator("failure_modes")
    @classmethod
    def dedupe_failures(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class HistoryContext(BaseModel):
    """Short trajectory summary for world-model queries."""

    model_config = ConfigDict(extra="forbid")

    recent_actions: list[HistoryEntry] = Field(default_factory=list)
    trajectory_depth: int = 0
    last_outcome: str | None = None


class UncertaintyFieldState(BaseModel):
    """Per-field provenance and confidence record."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    source: Literal["truth", "prediction", "missing", "calibrated"]
    confidence: float

    @field_validator("source")
    @classmethod
    def validate_field_source(cls, value: str) -> str:
        if value not in FIELD_PROVENANCE_TYPES:
            raise ValueError(f"unsupported field source: {value}")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be within [0, 1]")
        return round(float(value), 4)


class UncertaintyContext(BaseModel):
    """Structured uncertainty context accompanying each state."""

    model_config = ConfigDict(extra="forbid")

    field_states: list[UncertaintyFieldState] = Field(default_factory=list)
    epistemic_score: float
    aleatoric_score: float
    ood_score: float

    @field_validator("epistemic_score", "aleatoric_score", "ood_score")
    @classmethod
    def validate_scores(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("uncertainty scores must be within [0, 1]")
        return round(float(value), 4)


class ProvenanceRecord(BaseModel):
    """Structured provenance for one world state."""

    model_config = ConfigDict(extra="forbid")

    state_origin: Literal["offline_dataset", "real_simulation", "trajectory_replay", "model_rollout", "hybrid_calibration"]
    source_stage: Literal["initial", "predicted", "simulated"] = "predicted"
    analysis_fidelity: Literal["quick_screening", "partial_simulation", "full_ground_truth"]
    artifact_refs: list[str] = Field(default_factory=list)
    created_at: str

    @field_validator("artifact_refs")
    @classmethod
    def dedupe_artifacts(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class WorldState(BaseModel):
    """Formal world-state representation."""

    model_config = ConfigDict(extra="forbid")

    state_id: str
    task_id: str
    topology_context: TopologyContext
    parameter_state: list[ParameterValue] = Field(default_factory=list)
    environment_state: EnvironmentState
    evaluation_context: EvaluationContext
    operating_point_state: OperatingPointState
    structural_features: StructuralFeatures
    performance_observation: list[MetricEstimate] = Field(default_factory=list)
    constraint_observation: list[ConstraintObservation] = Field(default_factory=list)
    history_context: HistoryContext
    uncertainty_context: UncertaintyContext
    provenance: ProvenanceRecord


class ActionTarget(BaseModel):
    """Structured action target."""

    model_config = ConfigDict(extra="forbid")

    target_kind: Literal["variable", "variable_group", "topology_slot", "evaluation_strategy", "candidate"]
    variable_names: list[str] = Field(default_factory=list)
    topology_slot: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("variable_names")
    @classmethod
    def dedupe_targets(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ActionValidityGuard(BaseModel):
    """Formal action-validity constraints."""

    model_config = ConfigDict(extra="forbid")

    requires_domain_membership: bool = True
    requires_coupling_integrity: bool = True
    blocked_when_frozen: bool = True
    allowed_topology_modes: list[str] = Field(default_factory=lambda: ["fixed", "template_family", "search_space"])
    allowed_task_types: list[str] = Field(default_factory=lambda: list(TASK_TYPES))

    @field_validator("allowed_topology_modes")
    @classmethod
    def validate_topology_modes(cls, values: list[str]) -> list[str]:
        allowed = {"fixed", "template_family", "search_space"}
        invalid = [value for value in values if value not in allowed]
        if invalid:
            raise ValueError(f"unsupported topology modes: {invalid}")
        return _ordered_unique(values)

    @field_validator("allowed_task_types")
    @classmethod
    def validate_task_types(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in TASK_TYPES]
        if invalid:
            raise ValueError(f"unsupported task types: {invalid}")
        return _ordered_unique(values, TASK_TYPES)


class DesignAction(BaseModel):
    """Formal design-action representation."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    task_id: str
    action_family: Literal[
        "parameter_update",
        "coupled_update",
        "bias_reallocation",
        "compensation_adjustment",
        "topology_switch",
        "template_slot_mutation",
        "fidelity_upgrade",
        "candidate_prune",
    ]
    action_target: ActionTarget
    action_operator: Literal["set", "shift", "scale", "swap", "freeze", "unfreeze", "promote_fidelity", "drop_candidate"]
    action_payload: dict[str, float | int | str | bool] = Field(default_factory=dict)
    expected_scope: list[Literal["operating_point", "ac_stability", "power", "noise", "area", "evaluation_cost", "feasibility"]] = Field(default_factory=list)
    validity_guard: ActionValidityGuard
    source: Literal["planner", "critic", "expert_rule", "warm_start_policy", "manual"]
    timestamp: str

    @field_validator("expected_scope")
    @classmethod
    def validate_scope(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in EXPECTED_SCOPES]
        if invalid:
            raise ValueError(f"unsupported expected scopes: {invalid}")
        return _ordered_unique(values, EXPECTED_SCOPES)


class DeltaFeatures(BaseModel):
    """Structured transition deltas."""

    model_config = ConfigDict(extra="forbid")

    metric_deltas: dict[str, float] = Field(default_factory=dict)
    margin_deltas: dict[str, float] = Field(default_factory=dict)
    operating_point_deltas: dict[str, float] = Field(default_factory=dict)

    @field_validator("metric_deltas")
    @classmethod
    def validate_metric_deltas(cls, values: dict[str, float]) -> dict[str, float]:
        invalid = [metric for metric in values if metric not in WORLD_MODEL_METRICS]
        if invalid:
            raise ValueError(f"unsupported transition metric deltas: {invalid}")
        return dict(sorted(values.items()))


class TrustAssessment(BaseModel):
    """Structured trust and escalation assessment."""

    model_config = ConfigDict(extra="forbid")

    trust_level: Literal["high", "medium", "low", "blocked"]
    service_tier: Literal["screening_only", "ranking_ready", "rollout_ready", "must_escalate", "hard_block"]
    confidence: float
    uncertainty_score: float
    ood_score: float
    must_escalate: bool = False
    hard_block: bool = False
    reasons: list[str] = Field(default_factory=list)

    @field_validator("confidence", "uncertainty_score", "ood_score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("trust scores must be within [0, 1]")
        return round(float(value), 4)

    @field_validator("reasons")
    @classmethod
    def dedupe_reasons(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class TransitionRecord(BaseModel):
    """Structured transition record used for training and rollout archiving."""

    model_config = ConfigDict(extra="forbid")

    transition_id: str
    task_id: str
    state_t: WorldState
    action_t: DesignAction
    state_t_plus_1: WorldState
    delta_features: DeltaFeatures
    predicted_metrics: list[MetricEstimate] = Field(default_factory=list)
    ground_truth_metrics: list[MetricEstimate] = Field(default_factory=list)
    predicted_constraints: list[ConstraintObservation] = Field(default_factory=list)
    ground_truth_constraints: list[ConstraintObservation] = Field(default_factory=list)
    analysis_fidelity: Literal["quick_screening", "partial_simulation", "full_ground_truth"]
    trust_snapshot: TrustAssessment
    raw_artifact_refs: list[str] = Field(default_factory=list)
    label_quality: Literal["high_fidelity", "partial_fidelity", "pseudo_label"]

    @field_validator("raw_artifact_refs")
    @classmethod
    def dedupe_artifacts(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class HeadDefinition(BaseModel):
    """Structured head definition inside the bundle."""

    model_config = ConfigDict(extra="forbid")

    head_name: Literal[
        "metric_predictor",
        "feasibility_predictor",
        "transition_predictor",
        "uncertainty_estimator",
        "trust_estimator",
        "simulation_value_estimator",
    ]
    enabled: bool = True
    output_schema: str
    supported_metrics: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("supported_metrics")
    @classmethod
    def validate_supported_metrics(cls, values: list[str]) -> list[str]:
        invalid = [metric for metric in values if metric not in WORLD_MODEL_METRICS]
        if invalid:
            raise ValueError(f"unsupported supported_metrics: {invalid}")
        return _ordered_unique(values, WORLD_MODEL_METRICS)

    @field_validator("notes")
    @classmethod
    def dedupe_notes(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class PredictionHeads(BaseModel):
    """Bundle-level prediction head registry."""

    model_config = ConfigDict(extra="forbid")

    metric_prediction_head: HeadDefinition
    feasibility_head: HeadDefinition
    transition_head: HeadDefinition
    uncertainty_head: HeadDefinition
    trust_head: HeadDefinition
    simulation_value_head: HeadDefinition


class MetricErrorSummary(BaseModel):
    """Per-metric calibration summary."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    mae: float
    relative_error: float
    signed_bias: float = 0.0
    rank_correlation: float
    boundary_error: float
    sample_count: int

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, value: str) -> str:
        if value not in WORLD_MODEL_METRICS:
            raise ValueError(f"unsupported calibration metric: {value}")
        return value


class ConstraintReliabilitySummary(BaseModel):
    """Per-constraint-group calibration summary."""

    model_config = ConfigDict(extra="forbid")

    constraint_group: str
    brier_score: float
    expected_calibration_error: float
    confidence: float
    sample_count: int


class OODStatistics(BaseModel):
    """Bundle-level OOD statistics."""

    model_config = ConfigDict(extra="forbid")

    query_count: int
    high_risk_count: int
    high_risk_rate: float

    @field_validator("high_risk_rate")
    @classmethod
    def validate_rate(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("high_risk_rate must be within [0, 1]")
        return round(float(value), 4)


class LocalPatchRecord(BaseModel):
    """Local calibration patch record."""

    model_config = ConfigDict(extra="forbid")

    patch_id: str
    reason: str
    affected_metrics: list[str] = Field(default_factory=list)
    applied_at: str

    @field_validator("affected_metrics")
    @classmethod
    def validate_affected_metrics(cls, values: list[str]) -> list[str]:
        invalid = [metric for metric in values if metric not in WORLD_MODEL_METRICS]
        if invalid:
            raise ValueError(f"unsupported affected metrics: {invalid}")
        return _ordered_unique(values, WORLD_MODEL_METRICS)


class UsableRegion(BaseModel):
    """Calibration-informed usable region."""

    model_config = ConfigDict(extra="forbid")

    circuit_family: str
    template_id: str | None = None
    parameter_ranges: dict[str, list[float]] = Field(default_factory=dict)
    evaluation_intents: list[str] = Field(default_factory=list)
    readiness: Literal["experimental", "screening", "rollout_ready"]

    @field_validator("circuit_family")
    @classmethod
    def validate_family(cls, value: str) -> str:
        if value not in CIRCUIT_FAMILIES:
            raise ValueError(f"unsupported circuit family: {value}")
        return value

    @field_validator("evaluation_intents")
    @classmethod
    def dedupe_intents(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class CalibrationState(BaseModel):
    """Structured bundle calibration state."""

    model_config = ConfigDict(extra="forbid")

    calibration_version: str
    last_calibrated_at: str | None = None
    reference_simulator_signature: str
    per_metric_error_summary: list[MetricErrorSummary] = Field(default_factory=list)
    constraint_reliability_summary: list[ConstraintReliabilitySummary] = Field(default_factory=list)
    ood_statistics: OODStatistics
    local_patch_history: list[LocalPatchRecord] = Field(default_factory=list)
    usable_regions: list[UsableRegion] = Field(default_factory=list)


class SampleCounts(BaseModel):
    """Training sample statistics."""

    model_config = ConfigDict(extra="forbid")

    static_samples: int
    transition_samples: int
    failure_samples: int
    multi_fidelity_samples: int


class LossSummary(BaseModel):
    """Aggregate training-loss summary."""

    model_config = ConfigDict(extra="forbid")

    metric_loss: float
    feasibility_loss: float
    transition_loss: float
    uncertainty_loss: float
    total_loss: float


class ReplayStatistics(BaseModel):
    """Replay-buffer statistics for online refinement."""

    model_config = ConfigDict(extra="forbid")

    replay_sample_count: int
    trajectory_sample_count: int
    failure_fraction: float

    @field_validator("failure_fraction")
    @classmethod
    def validate_failure_fraction(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("failure_fraction must be within [0, 1]")
        return round(float(value), 4)


class TrainingState(BaseModel):
    """Structured training governance state."""

    model_config = ConfigDict(extra="forbid")

    dataset_signature: str
    sample_counts: SampleCounts
    family_coverage: list[str] = Field(default_factory=list)
    fidelity_mix: dict[str, float] = Field(default_factory=dict)
    objective_mix: list[str] = Field(default_factory=list)
    constraint_mix: list[str] = Field(default_factory=list)
    loss_summary: LossSummary
    best_checkpoint_ref: str
    update_policy: Literal["offline_retrain", "periodic_incremental", "online_finetune"]
    replay_statistics: ReplayStatistics

    @field_validator("family_coverage")
    @classmethod
    def validate_family_coverage(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in CIRCUIT_FAMILIES]
        if invalid:
            raise ValueError(f"unsupported family coverage values: {invalid}")
        return _ordered_unique(values, CIRCUIT_FAMILIES)

    @field_validator("fidelity_mix")
    @classmethod
    def validate_fidelity_mix(cls, values: dict[str, float]) -> dict[str, float]:
        invalid = [key for key in values if key not in ANALYSIS_FIDELITIES]
        if invalid:
            raise ValueError(f"unsupported fidelity_mix keys: {invalid}")
        normalized = {}
        for key, value in values.items():
            if value < 0.0 or value > 1.0:
                raise ValueError("fidelity_mix values must be within [0, 1]")
            normalized[key] = round(float(value), 4)
        return dict(sorted(normalized.items()))

    @field_validator("objective_mix", "constraint_mix")
    @classmethod
    def dedupe_string_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ServiceMethodSpec(BaseModel):
    """Formal service-method signature."""

    model_config = ConfigDict(extra="forbid")

    input_schema: str
    output_schema: str
    deterministic: bool = True


class ServingContract(BaseModel):
    """Structured serving contract for downstream layers."""

    model_config = ConfigDict(extra="forbid")

    predict_metrics: ServiceMethodSpec
    predict_feasibility: ServiceMethodSpec
    predict_transition: ServiceMethodSpec
    rollout: ServiceMethodSpec
    rank_candidates: ServiceMethodSpec
    estimate_simulation_value: ServiceMethodSpec
    calibrate_with_truth: ServiceMethodSpec
    validate_state: ServiceMethodSpec


class ThresholdSpec(BaseModel):
    """Threshold block used in trust policies."""

    model_config = ConfigDict(extra="forbid")

    max_ood_score: float
    max_uncertainty_score: float
    min_confidence: float

    @field_validator("max_ood_score", "max_uncertainty_score", "min_confidence")
    @classmethod
    def validate_thresholds(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("threshold values must be within [0, 1]")
        return round(float(value), 4)


class TrustPolicy(BaseModel):
    """Structured trust and escalation policy."""

    model_config = ConfigDict(extra="forbid")

    screening_threshold: ThresholdSpec
    ranking_threshold: ThresholdSpec
    rollout_threshold: ThresholdSpec
    must_escalate_conditions: list[str] = Field(default_factory=list)
    hard_block_conditions: list[str] = Field(default_factory=list)

    @field_validator("must_escalate_conditions", "hard_block_conditions")
    @classmethod
    def dedupe_conditions(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class WorldModelMetadata(BaseModel):
    """Audit metadata for third-layer bundle compilation."""

    model_config = ConfigDict(extra="forbid")

    created_by_layer: Literal["world_model_layer"] = "world_model_layer"
    compile_timestamp: str
    source_task_signature: str
    implementation_version: str
    assumptions: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @field_validator("assumptions", "provenance")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class WorldModelValidationIssue(BaseModel):
    """Structured validation issue for world-model compilation and serving."""

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


class WorldModelValidationStatus(BaseModel):
    """Embedded validation state for world-model objects."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    errors: list[WorldModelValidationIssue] = Field(default_factory=list)
    warnings: list[WorldModelValidationIssue] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)
    repair_history: list[str] = Field(default_factory=list)
    completeness_score: float

    @field_validator("unresolved_dependencies", "repair_history")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)

    @field_validator("completeness_score")
    @classmethod
    def validate_completeness(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("completeness_score must be within [0, 1]")
        return round(float(value), 4)


class WorldModelBundle(BaseModel):
    """Top-level world-model bundle object."""

    model_config = ConfigDict(extra="forbid")

    world_model_id: str
    schema_version: str = WORLD_MODEL_SCHEMA_VERSION
    parent_task_id: str
    supported_circuit_families: list[str] = Field(default_factory=list)
    supported_task_types: list[str] = Field(default_factory=list)
    state_schema: StateSchema
    action_schema: ActionSchema
    prediction_heads: PredictionHeads
    calibration_state: CalibrationState
    training_state: TrainingState
    serving_contract: ServingContract
    trust_policy: TrustPolicy
    metadata: WorldModelMetadata
    validation_status: WorldModelValidationStatus

    @field_validator("supported_circuit_families")
    @classmethod
    def validate_supported_families(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in CIRCUIT_FAMILIES]
        if invalid:
            raise ValueError(f"unsupported circuit families: {invalid}")
        return _ordered_unique(values, CIRCUIT_FAMILIES)

    @field_validator("supported_task_types")
    @classmethod
    def validate_supported_task_types(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in TASK_TYPES]
        if invalid:
            raise ValueError(f"unsupported task types: {invalid}")
        return _ordered_unique(values, TASK_TYPES)


class WorldModelSample(BaseModel):
    """Structured training or replay sample for world-model supervision."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    task_context: str
    state_t: WorldState
    action_t: DesignAction | None = None
    state_t_plus_1: WorldState | None = None
    analysis_fidelity: Literal["quick_screening", "partial_simulation", "full_ground_truth"]
    ground_truth_metrics: list[MetricEstimate] = Field(default_factory=list)
    ground_truth_constraints: list[ConstraintObservation] = Field(default_factory=list)
    raw_simulation_refs: list[str] = Field(default_factory=list)
    label_quality: Literal["high_fidelity", "partial_fidelity", "pseudo_label"]
    distribution_tag: str
    split_tag: Literal["train", "validation", "test", "replay"]

    @field_validator("raw_simulation_refs")
    @classmethod
    def dedupe_refs(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class MetricsPrediction(BaseModel):
    """Structured metrics prediction output."""

    model_config = ConfigDict(extra="forbid")

    state_id: str
    task_id: str
    metrics: list[MetricEstimate] = Field(default_factory=list)
    auxiliary_features: dict[str, float] = Field(default_factory=dict)
    trust_assessment: TrustAssessment


class FeasibilityPrediction(BaseModel):
    """Structured feasibility prediction output."""

    model_config = ConfigDict(extra="forbid")

    state_id: str
    task_id: str
    overall_feasibility: float
    per_group_constraints: list[ConstraintObservation] = Field(default_factory=list)
    most_likely_failure_reasons: list[str] = Field(default_factory=list)
    confidence: float
    trust_assessment: TrustAssessment

    @field_validator("overall_feasibility", "confidence")
    @classmethod
    def validate_probability(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("probabilities must be within [0, 1]")
        return round(float(value), 4)

    @field_validator("most_likely_failure_reasons")
    @classmethod
    def dedupe_reasons(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class TransitionPrediction(BaseModel):
    """Structured transition prediction output."""

    model_config = ConfigDict(extra="forbid")

    transition_id: str
    task_id: str
    next_state: WorldState
    delta_features: DeltaFeatures
    predicted_metrics: list[MetricEstimate] = Field(default_factory=list)
    predicted_constraints: list[ConstraintObservation] = Field(default_factory=list)
    analysis_fidelity: Literal["quick_screening", "partial_simulation", "full_ground_truth"]
    trust_assessment: TrustAssessment


class RankedCandidate(BaseModel):
    """One ranked candidate returned to downstream planning."""

    model_config = ConfigDict(extra="forbid")

    state_id: str
    score: float
    feasible_probability: float
    service_tier: Literal["screening_only", "ranking_ready", "rollout_ready", "must_escalate", "hard_block"]
    recommended_action: str

    @field_validator("feasible_probability")
    @classmethod
    def validate_feasible_probability(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("feasible_probability must be within [0, 1]")
        return round(float(value), 4)


class CandidateRanking(BaseModel):
    """Ranking response for batch candidate screening."""

    model_config = ConfigDict(extra="forbid")

    ranked_candidates: list[RankedCandidate] = Field(default_factory=list)
    recommended_threshold: float
    trust_assessment: TrustAssessment


class SimulationValueEstimate(BaseModel):
    """Estimated value of escalating a candidate to real simulation."""

    model_config = ConfigDict(extra="forbid")

    state_id: str
    estimated_value: float
    decision: Literal["defer", "simulate", "prioritize"]
    reasons: list[str] = Field(default_factory=list)
    trust_assessment: TrustAssessment

    @field_validator("reasons")
    @classmethod
    def dedupe_reasons(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class RolloutStep(BaseModel):
    """One step in an imagined trajectory."""

    model_config = ConfigDict(extra="forbid")

    step_index: int
    action: DesignAction
    transition: TransitionPrediction
    simulation_value: SimulationValueEstimate


class RolloutResponse(BaseModel):
    """Multi-step imagined rollout response."""

    model_config = ConfigDict(extra="forbid")

    initial_state_id: str
    horizon: int
    steps: list[RolloutStep] = Field(default_factory=list)
    terminal_state: WorldState
    trust_assessment: TrustAssessment


class TruthMetric(BaseModel):
    """Structured simulator-ground-truth metric for calibration."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    value: float

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, value: str) -> str:
        if value not in WORLD_MODEL_METRICS:
            raise ValueError(f"unsupported truth metric: {value}")
        return value


class TruthConstraint(BaseModel):
    """Structured simulator-ground-truth constraint result."""

    model_config = ConfigDict(extra="forbid")

    constraint_name: str
    constraint_group: str
    is_satisfied: bool
    margin: float


class TruthCalibrationRecord(BaseModel):
    """Ground-truth calibration input from the real simulator layer."""

    model_config = ConfigDict(extra="forbid")

    simulator_signature: str
    analysis_fidelity: Literal["quick_screening", "partial_simulation", "full_ground_truth"]
    truth_level: Literal["demonstrator_truth", "configured_truth"]
    validation_status: Literal["strong", "weak", "invalid"]
    metrics: list[TruthMetric] = Field(default_factory=list)
    constraints: list[TruthConstraint] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    provenance_tags: list[str] = Field(default_factory=list)
    timestamp: str

    @field_validator("artifact_refs", "provenance_tags")
    @classmethod
    def dedupe_artifacts(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class CalibrationUpdateResponse(BaseModel):
    """Structured calibration update output."""

    model_config = ConfigDict(extra="forbid")

    updated_bundle: WorldModelBundle
    updated_metrics: list[MetricErrorSummary] = Field(default_factory=list)
    updated_usable_regions: list[UsableRegion] = Field(default_factory=list)
    trust_assessment: TrustAssessment


class WorldModelCompilationReport(BaseModel):
    """Structured compile report for the world model layer."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["compiled", "compiled_with_warnings", "invalid"]
    derived_fields: list[str] = Field(default_factory=list)
    supported_metrics: list[str] = Field(default_factory=list)
    validation_errors: list[WorldModelValidationIssue] = Field(default_factory=list)
    validation_warnings: list[WorldModelValidationIssue] = Field(default_factory=list)
    acceptance_summary: dict[str, float | int | str] = Field(default_factory=dict)

    @field_validator("derived_fields")
    @classmethod
    def dedupe_fields(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)

    @field_validator("supported_metrics")
    @classmethod
    def validate_supported_metrics(cls, values: list[str]) -> list[str]:
        invalid = [metric for metric in values if metric not in WORLD_MODEL_METRICS]
        if invalid:
            raise ValueError(f"unsupported supported metrics: {invalid}")
        return _ordered_unique(values, WORLD_MODEL_METRICS)


class WorldModelCompileResponse(BaseModel):
    """Top-level world-model compile response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["compiled", "compiled_with_warnings", "invalid"]
    world_model_bundle: WorldModelBundle | None = None
    report: WorldModelCompilationReport


class AcceptanceFailureRecord(BaseModel):
    """Structured failure taxonomy entry for third-layer acceptance."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if value not in ACCEPTANCE_FAILURE_CODES:
            raise ValueError(f"unsupported acceptance failure code: {value}")
        return value
