"""Schemas for the memory and reflection layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from libs.schema.design_spec import CIRCUIT_FAMILIES
from libs.schema.design_task import TASK_TYPES

MEMORY_SCHEMA_VERSION = "memory-schema-v1"
MEMORY_MODES = (
    "episodic_memory",
    "cross_episode_consolidation",
    "cross_task_transfer",
    "governed_reflection",
)
PATTERN_TYPES = (
    "failure_pattern",
    "success_pattern",
    "calibration_pattern",
    "search_pattern",
    "budget_pattern",
    "robustness_pattern",
)
FRESHNESS_STATES = ("fresh", "stable", "stale", "expired")
GOVERNANCE_STATES = ("active", "candidate", "conflicted", "deprecated", "forgotten")
ADVICE_TYPES = (
    "initialization_hint",
    "search_adjustment",
    "trust_adjustment",
    "validation_focus",
    "calibration_priority",
    "budget_adjustment",
)
TARGET_LAYERS = ("layer2", "layer3", "layer4", "layer5")
VALIDATION_ERROR_CODES = (
    "schema_failure",
    "field_type_failure",
    "enum_failure",
    "reference_consistency_failure",
    "evidence_traceability_failure",
    "cross_object_consistency_failure",
    "pattern_evidence_threshold_failure",
    "reflection_policy_mapping_failure",
    "governance_failure",
)
ACCEPTANCE_FAILURE_CODES = (
    "schema_failure",
    "knowledge_invalidity",
    "memory_misretrieval",
    "pattern_overgeneralization",
    "evidence_insufficiency",
    "feedback_misleading",
    "negative_transfer",
    "governance_failure",
    "regression_drift",
)


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


class TaskSignature(BaseModel):
    """Deterministic task signature used for retrieval and transfer."""

    model_config = ConfigDict(extra="forbid")

    circuit_family: str
    task_type: str
    constraint_vector: list[str] = Field(default_factory=list)
    environment_profile: list[str] = Field(default_factory=list)
    evaluation_profile: list[str] = Field(default_factory=list)
    design_space_shape: list[str] = Field(default_factory=list)
    difficulty_profile_hash: str

    @field_validator("circuit_family")
    @classmethod
    def validate_family(cls, value: str) -> str:
        if value not in CIRCUIT_FAMILIES:
            raise ValueError(f"unsupported circuit family: {value}")
        return value

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, value: str) -> str:
        if value not in TASK_TYPES:
            raise ValueError(f"unsupported task type: {value}")
        return value

    @field_validator("constraint_vector", "environment_profile", "evaluation_profile", "design_space_shape")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class EvidenceReference(BaseModel):
    """Formal evidence binding back to upstream layer artifacts."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_layer: Literal["layer2", "layer3", "layer4", "layer5"]
    source_object_type: str
    source_object_id: str
    evidence_kind: str
    artifact_refs: list[str] = Field(default_factory=list)

    @field_validator("artifact_refs")
    @classmethod
    def dedupe_refs(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class InitialConditionsSnapshot(BaseModel):
    """Structured initial conditions for one episode."""

    model_config = ConfigDict(extra="forbid")

    init_strategy: str
    seed_count: int
    randomization_strategy: str
    warm_start_source: str | None = None


class ConstraintProfile(BaseModel):
    """Structured constraint snapshot for task-conditioned memory."""

    model_config = ConfigDict(extra="forbid")

    hard_constraint_names: list[str] = Field(default_factory=list)
    tightness: str
    feasibility_rules: list[str] = Field(default_factory=list)
    operating_region_rules: list[str] = Field(default_factory=list)

    @field_validator("hard_constraint_names", "feasibility_rules", "operating_region_rules")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SearchSummary(BaseModel):
    """Compact structured summary of one search episode."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    search_id: str
    total_candidates: int
    verified_candidates: int
    rejected_candidates: int
    trace_length: int
    final_phase: str
    termination_reason: str | None = None
    simulation_budget_used: int
    proxy_budget_used: int
    rollout_budget_used: int
    calibration_budget_used: int


class CandidateOutcomeSummary(BaseModel):
    """Compact candidate outcome summary stored inside memory."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    lifecycle_status: str
    feasible_probability: float | None = None
    priority_score: float | None = None
    world_state_ref: str | None = None
    truth_feasibility_status: str | None = None

    @field_validator("feasible_probability")
    @classmethod
    def validate_optional_probability(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if value < 0.0 or value > 1.0:
            raise ValueError("feasible_probability must be within [0, 1]")
        return round(float(value), 4)


class ActionSequenceRecord(BaseModel):
    """Structured action-sequence effectiveness record."""

    model_config = ConfigDict(extra="forbid")

    sequence_id: str
    action_ids: list[str] = Field(default_factory=list)
    action_families: list[str] = Field(default_factory=list)
    effect_type: Literal["effective", "ineffective"]
    observed_effects: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("action_ids", "action_families", "observed_effects", "evidence_refs")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class WorldModelBehaviorSummary(BaseModel):
    """Structured summary of world-model reliability behavior."""

    model_config = ConfigDict(extra="forbid")

    trust_alert_count: int
    must_escalate_count: int
    disagreement_flags: list[str] = Field(default_factory=list)
    calibration_priority: str | None = None
    observed_bias_metrics: list[str] = Field(default_factory=list)

    @field_validator("disagreement_flags", "observed_bias_metrics")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SimulationBudgetProfile(BaseModel):
    """Structured budget usage snapshot for one episode."""

    model_config = ConfigDict(extra="forbid")

    proxy_evaluations_used: int
    rollouts_used: int
    simulations_used: int
    calibrations_used: int
    budget_pressure: float

    @field_validator("budget_pressure")
    @classmethod
    def validate_pressure(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("budget pressure must be within [0, 1]")
        return round(float(value), 4)


class PhaseTransitionRecord(BaseModel):
    """Structured phase transition summary."""

    model_config = ConfigDict(extra="forbid")

    from_phase: str
    to_phase: str
    trigger: str
    step_index: int
    supporting_trace_refs: list[str] = Field(default_factory=list)

    @field_validator("supporting_trace_refs")
    @classmethod
    def dedupe_refs(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class TurningPointRecord(BaseModel):
    """Structured turning-point record extracted from one episode."""

    model_config = ConfigDict(extra="forbid")

    turning_point_id: str
    turning_point_type: Literal["feasibility_recovery", "trust_violation", "phase_shift", "verification_success", "verification_failure"]
    candidate_id: str | None = None
    description_tags: list[str] = Field(default_factory=list)
    supporting_evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("description_tags", "supporting_evidence_refs")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class FinalOutcomeSummary(BaseModel):
    """Structured final episode outcome."""

    model_config = ConfigDict(extra="forbid")

    outcome_status: Literal["verified_success", "partial_success", "failure", "budget_exhausted", "trust_blocked"]
    best_candidate_id: str | None = None
    best_feasibility_status: str | None = None
    robustness_status: str | None = None
    failure_classes: list[str] = Field(default_factory=list)

    @field_validator("failure_classes")
    @classmethod
    def dedupe_failures(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class EpisodeMemoryRecord(BaseModel):
    """Formal episodic memory record."""

    model_config = ConfigDict(extra="forbid")

    episode_memory_id: str
    task_id: str
    task_signature: TaskSignature
    circuit_family: str
    task_type: str
    initial_conditions: InitialConditionsSnapshot
    constraint_profile: ConstraintProfile
    search_summary: SearchSummary
    best_feasible_result: CandidateOutcomeSummary | None = None
    best_infeasible_result: CandidateOutcomeSummary | None = None
    dominant_failure_modes: list[str] = Field(default_factory=list)
    effective_action_sequences: list[ActionSequenceRecord] = Field(default_factory=list)
    ineffective_action_sequences: list[ActionSequenceRecord] = Field(default_factory=list)
    world_model_behavior_summary: WorldModelBehaviorSummary
    simulation_budget_profile: SimulationBudgetProfile
    phase_transition_trace: list[PhaseTransitionRecord] = Field(default_factory=list)
    turning_points: list[TurningPointRecord] = Field(default_factory=list)
    final_outcome: FinalOutcomeSummary
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    confidence_score: float
    timestamp: str

    @field_validator("circuit_family")
    @classmethod
    def validate_family(cls, value: str) -> str:
        if value not in CIRCUIT_FAMILIES:
            raise ValueError(f"unsupported circuit family: {value}")
        return value

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, value: str) -> str:
        if value not in TASK_TYPES:
            raise ValueError(f"unsupported task type: {value}")
        return value

    @field_validator("dominant_failure_modes")
    @classmethod
    def dedupe_failures(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence_score must be within [0, 1]")
        return round(float(value), 4)


class ApplicabilityScope(BaseModel):
    """Structured applicability scope for patterns and advice."""

    model_config = ConfigDict(extra="forbid")

    circuit_families: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    topology_modes: list[str] = Field(default_factory=list)
    difficulty_bands: list[str] = Field(default_factory=list)
    environment_tags: list[str] = Field(default_factory=list)

    @field_validator("circuit_families")
    @classmethod
    def validate_families(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in CIRCUIT_FAMILIES]
        if invalid:
            raise ValueError(f"unsupported circuit families: {invalid}")
        return _ordered_unique(values)

    @field_validator("task_types")
    @classmethod
    def validate_task_types(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in TASK_TYPES]
        if invalid:
            raise ValueError(f"unsupported task types: {invalid}")
        return _ordered_unique(values)

    @field_validator("topology_modes", "difficulty_bands", "environment_tags")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class InterventionRecord(BaseModel):
    """Structured recommended intervention or anti-pattern."""

    model_config = ConfigDict(extra="forbid")

    label: str
    action: str
    payload: dict[str, str | float | int | bool] = Field(default_factory=dict)


class PatternMemoryRecord(BaseModel):
    """Formal cross-episode pattern record."""

    model_config = ConfigDict(extra="forbid")

    pattern_id: str
    pattern_type: str
    applicability_scope: ApplicabilityScope
    trigger_signature: list[str] = Field(default_factory=list)
    context_constraints: list[str] = Field(default_factory=list)
    observed_contexts: list[str] = Field(default_factory=list)
    recommended_interventions: list[InterventionRecord] = Field(default_factory=list)
    anti_patterns: list[InterventionRecord] = Field(default_factory=list)
    expected_effects: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    supporting_evidence_count: int
    supporting_episode_refs: list[str] = Field(default_factory=list)
    confidence_level: float
    freshness_state: str
    activation_policy: list[str] = Field(default_factory=list)
    governance_state: str

    @field_validator("pattern_type")
    @classmethod
    def validate_pattern_type(cls, value: str) -> str:
        if value not in PATTERN_TYPES:
            raise ValueError(f"unsupported pattern type: {value}")
        return value

    @field_validator("freshness_state")
    @classmethod
    def validate_freshness(cls, value: str) -> str:
        if value not in FRESHNESS_STATES:
            raise ValueError(f"unsupported freshness state: {value}")
        return value

    @field_validator("governance_state")
    @classmethod
    def validate_governance_state(cls, value: str) -> str:
        if value not in GOVERNANCE_STATES:
            raise ValueError(f"unsupported governance state: {value}")
        return value

    @field_validator("trigger_signature", "context_constraints", "observed_contexts", "expected_effects", "risk_notes", "supporting_episode_refs", "activation_policy")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence_level must be within [0, 1]")
        return round(float(value), 4)


class DiagnosisSummary(BaseModel):
    """Structured diagnosis section inside a reflection report."""

    model_config = ConfigDict(extra="forbid")

    key_findings: list[str] = Field(default_factory=list)
    dominant_patterns: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("key_findings", "dominant_patterns", "evidence_refs")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class CounterfactualHypothesis(BaseModel):
    """Evidence-backed counterfactual hypothesis."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    premise: str
    expected_effect: str
    confidence: float
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be within [0, 1]")
        return round(float(value), 4)

    @field_validator("evidence_refs")
    @classmethod
    def dedupe_refs(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class PolicyUpdateRecommendation(BaseModel):
    """Structured recommendation emitted by reflection."""

    model_config = ConfigDict(extra="forbid")

    target_layer: str
    update_type: str
    payload: dict[str, str | float | int | bool] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("target_layer")
    @classmethod
    def validate_target_layer(cls, value: str) -> str:
        if value not in TARGET_LAYERS:
            raise ValueError(f"unsupported target layer: {value}")
        return value

    @field_validator("evidence_refs")
    @classmethod
    def dedupe_refs(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ConfidenceAssessment(BaseModel):
    """Structured confidence assessment for one reflection."""

    model_config = ConfigDict(extra="forbid")

    evidence_count: int
    confidence_level: float
    uncertainty_notes: list[str] = Field(default_factory=list)

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence_level must be within [0, 1]")
        return round(float(value), 4)

    @field_validator("uncertainty_notes")
    @classmethod
    def dedupe_notes(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ReflectionReport(BaseModel):
    """Formal reflection report."""

    model_config = ConfigDict(extra="forbid")

    reflection_id: str
    task_id: str
    episode_scope: list[str] = Field(default_factory=list)
    reflection_scope: Literal["episode", "cross_episode", "cross_task"]
    failure_synthesis: DiagnosisSummary
    success_synthesis: DiagnosisSummary
    search_diagnosis: DiagnosisSummary
    world_model_diagnosis: DiagnosisSummary
    simulation_diagnosis: DiagnosisSummary
    counterfactual_hypotheses: list[CounterfactualHypothesis] = Field(default_factory=list)
    recommended_policy_updates: list[PolicyUpdateRecommendation] = Field(default_factory=list)
    confidence_assessment: ConfidenceAssessment
    evidence_refs: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @field_validator("episode_scope", "evidence_refs", "provenance")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class FeedbackAdvice(BaseModel):
    """Formal advisory feedback emitted by memory."""

    model_config = ConfigDict(extra="forbid")

    advice_id: str
    target_layer: str
    target_scope: str
    advice_type: str
    advice_payload: dict[str, str | float | int | bool] = Field(default_factory=dict)
    applicability_scope: ApplicabilityScope
    confidence_level: float
    priority_level: Literal["low", "medium", "high"] = "medium"
    expiry_condition: str
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("target_layer")
    @classmethod
    def validate_target_layer(cls, value: str) -> str:
        if value not in TARGET_LAYERS:
            raise ValueError(f"unsupported target layer: {value}")
        return value

    @field_validator("advice_type")
    @classmethod
    def validate_advice_type(cls, value: str) -> str:
        if value not in ADVICE_TYPES:
            raise ValueError(f"unsupported advice type: {value}")
        return value

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence_level must be within [0, 1]")
        return round(float(value), 4)

    @field_validator("evidence_refs")
    @classmethod
    def dedupe_refs(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class RetrievalHit(BaseModel):
    """One retrieval hit from the memory system."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["episode", "pattern", "reflection"]
    source_id: str
    score: float
    evidence_count: int
    rationale_tags: list[str] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("retrieval score must be within [0, 1]")
        return round(float(value), 4)

    @field_validator("rationale_tags")
    @classmethod
    def dedupe_tags(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class RetrievalResult(BaseModel):
    """Structured retrieval result."""

    model_config = ConfigDict(extra="forbid")

    task_signature: TaskSignature
    episode_hits: list[RetrievalHit] = Field(default_factory=list)
    pattern_hits: list[RetrievalHit] = Field(default_factory=list)
    reflection_hits: list[RetrievalHit] = Field(default_factory=list)
    feedback_advice: list[FeedbackAdvice] = Field(default_factory=list)
    retrieval_precision_proxy: float
    negative_transfer_risk: float

    @field_validator("retrieval_precision_proxy", "negative_transfer_risk")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("retrieval scores must be within [0, 1]")
        return round(float(value), 4)


class MemoryScopeDefinition(BaseModel):
    """Formal scope definition of the memory system."""

    model_config = ConfigDict(extra="forbid")

    supported_modes: list[str] = Field(default_factory=lambda: list(MEMORY_MODES))
    supported_target_layers: list[str] = Field(default_factory=lambda: list(TARGET_LAYERS))
    supported_pattern_types: list[str] = Field(default_factory=lambda: list(PATTERN_TYPES))

    @field_validator("supported_modes")
    @classmethod
    def validate_modes(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in MEMORY_MODES]
        if invalid:
            raise ValueError(f"unsupported memory modes: {invalid}")
        return _ordered_unique(values)

    @field_validator("supported_target_layers")
    @classmethod
    def validate_layers(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in TARGET_LAYERS]
        if invalid:
            raise ValueError(f"unsupported target layers: {invalid}")
        return _ordered_unique(values)

    @field_validator("supported_pattern_types")
    @classmethod
    def validate_patterns(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in PATTERN_TYPES]
        if invalid:
            raise ValueError(f"unsupported pattern types: {invalid}")
        return _ordered_unique(values)


class StoreSchema(BaseModel):
    """Schema contract for one store."""

    model_config = ConfigDict(extra="forbid")

    required_fields: list[str] = Field(default_factory=list)
    key_field: str

    @field_validator("required_fields")
    @classmethod
    def dedupe_fields(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class RetrievalPolicySpec(BaseModel):
    """Formal retrieval policy."""

    model_config = ConfigDict(extra="forbid")

    task_signature_weight: float = 0.5
    circuit_family_weight: float = 0.15
    task_type_weight: float = 0.1
    evidence_weight: float = 0.15
    recency_weight: float = 0.1
    top_k: int = 5
    minimum_score: float = 0.25

    @field_validator("task_signature_weight", "circuit_family_weight", "task_type_weight", "evidence_weight", "recency_weight", "minimum_score")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("retrieval policy values must be within [0, 1]")
        return round(float(value), 4)


class ConsolidationPolicy(BaseModel):
    """Formal consolidation policy."""

    model_config = ConfigDict(extra="forbid")

    minimum_pattern_support: int = 2
    require_verification_evidence: bool = True
    require_feedback_mapping: bool = True
    max_episode_action_sequences: int = 4


class QualityPolicy(BaseModel):
    """Quality-governance policy."""

    model_config = ConfigDict(extra="forbid")

    minimum_pattern_confidence: float = 0.55
    minimum_reflection_confidence: float = 0.5
    block_low_evidence_global_advice: bool = True
    conflict_resolution_mode: Literal["confidence_first", "recency_first", "deprecate_both"] = "confidence_first"

    @field_validator("minimum_pattern_confidence", "minimum_reflection_confidence")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("quality thresholds must be within [0, 1]")
        return round(float(value), 4)


class ForgettingPolicy(BaseModel):
    """Formal forgetting and decay policy."""

    model_config = ConfigDict(extra="forbid")

    max_pattern_records: int = 128
    max_episode_records: int = 256
    stale_after_episode_gap: int = 25
    expire_after_episode_gap: int = 80
    confidence_decay: float = 0.05

    @field_validator("confidence_decay")
    @classmethod
    def validate_decay(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence_decay must be within [0, 1]")
        return round(float(value), 4)


class FeedbackContract(BaseModel):
    """Formal feedback contract for downstream layers."""

    model_config = ConfigDict(extra="forbid")

    supported_advice_types: list[str] = Field(default_factory=lambda: list(ADVICE_TYPES))
    target_layers: list[str] = Field(default_factory=lambda: list(TARGET_LAYERS))

    @field_validator("supported_advice_types")
    @classmethod
    def validate_advice_types(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in ADVICE_TYPES]
        if invalid:
            raise ValueError(f"unsupported advice types: {invalid}")
        return _ordered_unique(values)

    @field_validator("target_layers")
    @classmethod
    def validate_layers(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in TARGET_LAYERS]
        if invalid:
            raise ValueError(f"unsupported target layers: {invalid}")
        return _ordered_unique(values)


class IndexingState(BaseModel):
    """Structured indexing state."""

    model_config = ConfigDict(extra="forbid")

    indexed_task_signatures: list[str] = Field(default_factory=list)
    episode_count: int = 0
    pattern_count: int = 0
    reflection_count: int = 0
    last_consolidated_episode_id: str | None = None

    @field_validator("indexed_task_signatures")
    @classmethod
    def dedupe_signatures(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class MemoryMetadata(BaseModel):
    """Audit metadata for the memory layer."""

    model_config = ConfigDict(extra="forbid")

    created_by_layer: Literal["memory_reflection_layer"] = "memory_reflection_layer"
    implementation_version: str
    assumptions: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @field_validator("assumptions", "provenance")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class MemoryValidationIssue(BaseModel):
    """Structured validation issue for sixth-layer objects."""

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


class MemoryValidationStatus(BaseModel):
    """Embedded validation status for MemoryBundle."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    errors: list[MemoryValidationIssue] = Field(default_factory=list)
    warnings: list[MemoryValidationIssue] = Field(default_factory=list)
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


class MemoryBundle(BaseModel):
    """Top-level long-term knowledge bundle for the sixth layer."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    schema_version: str = MEMORY_SCHEMA_VERSION
    scope_definition: MemoryScopeDefinition
    episode_store_schema: StoreSchema
    pattern_store_schema: StoreSchema
    reflection_store_schema: StoreSchema
    retrieval_policy: RetrievalPolicySpec
    consolidation_policy: ConsolidationPolicy
    quality_policy: QualityPolicy
    forgetting_policy: ForgettingPolicy
    feedback_contract: FeedbackContract
    episode_records: list[EpisodeMemoryRecord] = Field(default_factory=list)
    pattern_records: list[PatternMemoryRecord] = Field(default_factory=list)
    reflection_records: list[ReflectionReport] = Field(default_factory=list)
    indexing_state: IndexingState
    metadata: MemoryMetadata
    validation_status: MemoryValidationStatus


class MemoryCompilationReport(BaseModel):
    """Structured compile report for the sixth layer."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["compiled", "compiled_with_warnings", "invalid"]
    derived_fields: list[str] = Field(default_factory=list)
    validation_errors: list[MemoryValidationIssue] = Field(default_factory=list)
    validation_warnings: list[MemoryValidationIssue] = Field(default_factory=list)
    acceptance_summary: dict[str, float | int | str] = Field(default_factory=dict)

    @field_validator("derived_fields")
    @classmethod
    def dedupe_fields(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class MemoryCompileResponse(BaseModel):
    """Top-level compile response for MemoryBundle."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["compiled", "compiled_with_warnings", "invalid"]
    memory_bundle: MemoryBundle | None = None
    report: MemoryCompilationReport


class IngestionResponse(BaseModel):
    """Response emitted after trajectory consolidation."""

    model_config = ConfigDict(extra="forbid")

    memory_bundle: MemoryBundle
    episode_record: EpisodeMemoryRecord
    new_patterns: list[PatternMemoryRecord] = Field(default_factory=list)
    reflection_report: ReflectionReport | None = None
    emitted_feedback: list[FeedbackAdvice] = Field(default_factory=list)


class MemoryAcceptanceFailureRecord(BaseModel):
    """Structured failure taxonomy entry for sixth-layer acceptance."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if value not in ACCEPTANCE_FAILURE_CODES:
            raise ValueError(f"unsupported acceptance failure code: {value}")
        return value
