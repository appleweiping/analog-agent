"""Schemas for the task formalization layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from libs.schema.design_spec import CIRCUIT_FAMILIES

TASK_TYPES = ("sizing", "topology_sizing", "calibration")
TOPOLOGY_MODES = ("fixed", "template_family", "search_space")
ANALYSIS_TYPES = ("op", "ac", "tran", "noise", "pvt_sweep", "monte_carlo")
OBJECTIVE_DIRECTIONS = ("maximize", "minimize")
OBJECTIVE_MODES = ("feasibility", "single", "multi_objective", "constrained_single")
SCALARIZATION_MODES = ("none", "weighted_sum", "tchebycheff", "lexicographic", "pareto_only")
VARIABLE_KINDS = ("continuous", "integer", "categorical", "binary")
VARIABLE_DTYPES = ("float", "int", "string", "bool")
VARIABLE_SCALES = ("linear", "log")
FIELD_SOURCE_VALUES = ("template_default", "process_rule", "expert_rule", "user_override", "system_inferred")
CONSTRAINT_RELATIONS = (">=", "<=", "==", "in_range")
EVALUATION_STAGES = ("op", "ac", "tran", "noise", "pvt", "mc")
COST_CLASSES = ("cheap", "moderate", "expensive")
FIDELITY_POLICIES = ("single_fidelity", "staged_fidelity")
SOLVER_FAMILIES = ("bayesopt", "cmaes", "rl", "model_based_mpc", "hybrid")
SEARCH_STAGES = ("coarse_exploration", "direct_local_refinement", "feasibility_first")
PARALLELISM_HINTS = ("single", "batch", "multi_stage")
TIGHTNESS_LEVELS = ("low", "medium", "high")
FEASIBILITY_LEVELS = ("low", "medium", "high", "unknown")
VALIDATION_ERROR_CODES = (
    "schema_failure",
    "missing_problem_core",
    "constraint_direction_error",
    "evaluation_coverage_error",
    "topology_mode_inconsistency",
    "unresolved_dependency_error",
    "semantic_consistency_error",
    "solver_readiness_error",
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


class GraphNode(BaseModel):
    """Node in a task-graph representation."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    operation: str
    consumes: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)

    @field_validator("consumes", "produces")
    @classmethod
    def dedupe_fields(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class GraphEdge(BaseModel):
    """Directed dependency edge in a task graph."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    condition: str | None = None


class TopologyPort(BaseModel):
    """External topology port definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    direction: Literal["input", "output", "bias", "supply", "ground"]


class InstanceSlot(BaseModel):
    """Template slot for a circuit instance."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    device_type: str
    tunable_parameters: list[str] = Field(default_factory=list)

    @field_validator("tunable_parameters")
    @classmethod
    def dedupe_parameters(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ConnectivityRule(BaseModel):
    """Structured topology connectivity rule."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    relation: str


class TopologyConstraint(BaseModel):
    """Structured topology constraint."""

    model_config = ConfigDict(extra="forbid")

    name: str
    mandatory: bool = True
    description: str


class GraphRepresentation(BaseModel):
    """Optional graph-based topology representation."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class TopologySpec(BaseModel):
    """Formal topology anchor for a DesignTask."""

    model_config = ConfigDict(extra="forbid")

    topology_mode: Literal["fixed", "template_family", "search_space"]
    template_id: str | None = None
    template_version: str | None = None
    ports: list[TopologyPort] = Field(default_factory=list)
    instances_schema: list[InstanceSlot] = Field(default_factory=list)
    connectivity_schema: list[ConnectivityRule] = Field(default_factory=list)
    topology_constraints: list[TopologyConstraint] = Field(default_factory=list)
    optional_graph_repr: GraphRepresentation | None = None


class VariableDomain(BaseModel):
    """Explicit domain definition for one design variable."""

    model_config = ConfigDict(extra="forbid")

    lower: float | int | None = None
    upper: float | int | None = None
    choices: list[str | float | int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_domain(self) -> "VariableDomain":
        if self.choices:
            return self
        if self.lower is None or self.upper is None:
            raise ValueError("continuous and integer domains must define lower and upper bounds")
        if self.lower > self.upper:
            raise ValueError("domain lower bound must be <= upper bound")
        return self


class DesignVariable(BaseModel):
    """Formal variable definition in the optimization search space."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    kind: Literal["continuous", "integer", "categorical", "binary"]
    dtype: Literal["float", "int", "string", "bool"]
    domain: VariableDomain
    scale: Literal["linear", "log"]
    units: str
    default: float | int | str | bool | None = None
    source: Literal["template_default", "process_rule", "expert_rule", "user_override", "system_inferred"]
    is_required: bool = True
    coupling_group: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "DesignVariable":
        if self.kind in {"categorical", "binary"} and not self.domain.choices:
            raise ValueError("categorical and binary variables must provide explicit choices")
        if self.kind == "binary" and sorted(self.domain.choices) != [0, 1]:
            raise ValueError("binary variables must use [0, 1] as choices")
        return self


class VariableRelationConstraint(BaseModel):
    """Structured constraint inside the design-space definition."""

    model_config = ConfigDict(extra="forbid")

    left: str
    relation: Literal[">=", "<=", "=="]
    right: str | float | int
    reason: str


class ConditionalVariable(BaseModel):
    """Variable activated only under explicit conditions."""

    model_config = ConfigDict(extra="forbid")

    name: str
    active_when: str


class NormalizationPolicy(BaseModel):
    """Normalization strategy for downstream planners and models."""

    model_config = ConfigDict(extra="forbid")

    continuous_strategy: Literal["linear", "log", "mixed"]
    categorical_strategy: Literal["one_hot", "index"]
    clip_to_domain: bool = True


class DesignSpace(BaseModel):
    """Formal design-space description."""

    model_config = ConfigDict(extra="forbid")

    variables: list[DesignVariable] = Field(default_factory=list)
    global_constraints: list[VariableRelationConstraint] = Field(default_factory=list)
    derived_variables: list[str] = Field(default_factory=list)
    frozen_variables: list[str] = Field(default_factory=list)
    conditional_variables: list[ConditionalVariable] = Field(default_factory=list)
    normalization_policy: NormalizationPolicy

    @field_validator("derived_variables", "frozen_variables")
    @classmethod
    def dedupe_names(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ObjectiveTerm(BaseModel):
    """One formal optimization term."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    direction: Literal["maximize", "minimize"]
    weight: float = 1.0
    transform: Literal["identity", "log", "penalty"] = "identity"
    normalization: Literal["none", "zscore", "minmax"] = "none"
    source_constraint_relation: str | None = None


class ObjectiveSpec(BaseModel):
    """Formal objective definition."""

    model_config = ConfigDict(extra="forbid")

    objective_mode: Literal["feasibility", "single", "multi_objective", "constrained_single"]
    terms: list[ObjectiveTerm] = Field(default_factory=list)
    scalarization: Literal["none", "weighted_sum", "tchebycheff", "lexicographic", "pareto_only"] = "none"
    reference_point: dict[str, float] = Field(default_factory=dict)
    priority_policy: Literal["balanced", "feasibility_first", "lexicographic"] = "balanced"
    reporting_metrics: list[str] = Field(default_factory=list)

    @field_validator("reporting_metrics")
    @classmethod
    def dedupe_reporting_metrics(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ConstraintSpec(BaseModel):
    """Formal solver-facing constraint definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    metric: str
    relation: Literal[">=", "<=", "==", "in_range"]
    threshold: float | None = None
    lower_threshold: float | None = None
    upper_threshold: float | None = None
    tolerance: float = 0.0
    evaluation_stage: Literal["op", "ac", "tran", "noise", "pvt", "mc"]
    penalty_policy: Literal["hard_fail", "hinge", "quadratic"] = "hard_fail"
    source: Literal["user", "template_rule", "physics_rule", "system_inferred"]
    criticality: Literal["core", "important", "auxiliary"] = "core"

    @model_validator(mode="after")
    def validate_thresholds(self) -> "ConstraintSpec":
        if self.relation == "in_range":
            if self.lower_threshold is None or self.upper_threshold is None:
                raise ValueError("range constraints require lower_threshold and upper_threshold")
        elif self.threshold is None:
            raise ValueError("non-range constraints require threshold")
        return self


class ConstraintGroup(BaseModel):
    """Structured grouping of related constraints."""

    model_config = ConfigDict(extra="forbid")

    name: str
    members: list[str] = Field(default_factory=list)

    @field_validator("members")
    @classmethod
    def dedupe_members(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ConstraintSet(BaseModel):
    """Formal constraint system for the optimization problem."""

    model_config = ConfigDict(extra="forbid")

    hard_constraints: list[ConstraintSpec] = Field(default_factory=list)
    soft_constraints: list[ConstraintSpec] = Field(default_factory=list)
    feasibility_rules: list[str] = Field(default_factory=list)
    operating_region_rules: list[str] = Field(default_factory=list)
    constraint_groups: list[ConstraintGroup] = Field(default_factory=list)

    @field_validator("feasibility_rules", "operating_region_rules")
    @classmethod
    def dedupe_rules(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class AnalysisConfig(BaseModel):
    """Structured analysis configuration."""

    model_config = ConfigDict(extra="forbid")

    parameters: dict[str, float | int | str | bool] = Field(default_factory=dict)


class AnalysisSpec(BaseModel):
    """One structured analysis step."""

    model_config = ConfigDict(extra="forbid")

    analysis_type: Literal["op", "ac", "tran", "noise", "pvt_sweep", "monte_carlo"]
    order: int
    config: AnalysisConfig = Field(default_factory=AnalysisConfig)
    required_metrics: list[str] = Field(default_factory=list)
    estimated_cost: Literal["cheap", "moderate", "expensive"]

    @field_validator("required_metrics")
    @classmethod
    def dedupe_metrics(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class MetricExtractor(BaseModel):
    """Structured metric extraction rule."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    from_analysis: Literal["op", "ac", "tran", "noise", "pvt_sweep", "monte_carlo"]
    method: str


class ConditionPolicy(BaseModel):
    """Structured policy for one environmental dimension."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["inherit", "fixed", "sweep"]
    values: list[float | str] = Field(default_factory=list)


class StopCondition(BaseModel):
    """Structured early-stop condition for evaluation."""

    model_config = ConfigDict(extra="forbid")

    trigger: str
    action: Literal["stop", "continue", "fallback"]


class EvaluationPlan(BaseModel):
    """Evaluation plan connecting the optimization problem to simulator stages."""

    model_config = ConfigDict(extra="forbid")

    analyses: list[AnalysisSpec] = Field(default_factory=list)
    metric_extractors: list[MetricExtractor] = Field(default_factory=list)
    corners_policy: ConditionPolicy
    temperature_policy: ConditionPolicy
    load_policy: ConditionPolicy
    simulation_budget_class: Literal["cheap", "moderate", "expensive"]
    fidelity_policy: Literal["single_fidelity", "staged_fidelity"]
    stop_conditions: list[StopCondition] = Field(default_factory=list)


class CandidateSeed(BaseModel):
    """Structured initial candidate seed."""

    model_config = ConfigDict(extra="forbid")

    seed_id: str
    values: dict[str, float | int | str | bool] = Field(default_factory=dict)
    source: str


class RandomizationPolicy(BaseModel):
    """Initialization randomization policy."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    strategy: Literal["none", "lhs", "sobol", "gaussian_jitter", "hybrid"]
    amplitude: float = 0.0


class ReproducibilitySpec(BaseModel):
    """Reproducibility metadata for initialization."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    tag: str


class InitialState(BaseModel):
    """Initialization plan for the DesignTask."""

    model_config = ConfigDict(extra="forbid")

    init_strategy: Literal["template_default", "lhs", "sobol", "expert_seed", "replay_memory", "hybrid"]
    seed_candidates: list[CandidateSeed] = Field(default_factory=list)
    template_defaults: dict[str, float | int | str | bool] = Field(default_factory=dict)
    warm_start_source: str | None = None
    randomization_policy: RandomizationPolicy
    reproducibility: ReproducibilitySpec


class CriterionSpec(BaseModel):
    """Structured success or failure criterion."""

    model_config = ConfigDict(extra="forbid")

    name: str
    metric: str | None = None
    relation: Literal[">=", "<=", "==", "event"]
    threshold: float | str | None = None


class FailureRoute(BaseModel):
    """Structured failure-handling route."""

    model_config = ConfigDict(extra="forbid")

    trigger: str
    route_to: str
    reason: str


class TaskGraph(BaseModel):
    """Static task-execution graph."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    entrypoints: list[str] = Field(default_factory=list)
    success_criteria: list[CriterionSpec] = Field(default_factory=list)
    failure_routes: list[FailureRoute] = Field(default_factory=list)

    @field_validator("entrypoints")
    @classmethod
    def dedupe_entrypoints(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class DifficultyProfile(BaseModel):
    """Structured task-difficulty estimate."""

    model_config = ConfigDict(extra="forbid")

    variable_dimension: int
    discrete_degree: float
    constraint_tightness: Literal["low", "medium", "high"]
    evaluation_cost: Literal["cheap", "moderate", "expensive"]
    expected_feasibility: Literal["low", "medium", "high", "unknown"]
    sensitivity_hint: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)

    @field_validator("sensitivity_hint", "risk_flags")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SolverHint(BaseModel):
    """Task-level hints for downstream optimization layers."""

    model_config = ConfigDict(extra="forbid")

    recommended_solver_family: Literal["bayesopt", "cmaes", "rl", "model_based_mpc", "hybrid"]
    recommended_search_stage: Literal["coarse_exploration", "direct_local_refinement", "feasibility_first"]
    surrogate_friendly: bool
    needs_feasibility_first: bool
    parallelism_hint: Literal["single", "batch", "multi_stage"]
    budget_hint: Literal["low", "medium", "high"]


class TaskMetadata(BaseModel):
    """Audit and provenance metadata for DesignTask compilation."""

    model_config = ConfigDict(extra="forbid")

    created_by_layer: Literal["task_formalization_layer"] = "task_formalization_layer"
    compile_timestamp: str
    schema_version: str
    source_spec_signature: str
    assumptions: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @field_validator("assumptions", "provenance")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class TaskValidationIssue(BaseModel):
    """Structured validation issue for DesignTask compilation."""

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


class ValidationStatus(BaseModel):
    """Embedded task validation state."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    errors: list[TaskValidationIssue] = Field(default_factory=list)
    warnings: list[TaskValidationIssue] = Field(default_factory=list)
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
        return round(value, 4)


class DesignTask(BaseModel):
    """Formal optimization-problem representation compiled from DesignSpec."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    parent_spec_id: str
    task_type: Literal["sizing", "topology_sizing", "calibration"]
    circuit_family: Literal[
        "two_stage_ota",
        "folded_cascode_ota",
        "telescopic_ota",
        "comparator",
        "ldo",
        "bandgap",
        "unknown",
    ]
    topology: TopologySpec
    design_space: DesignSpace
    objective: ObjectiveSpec
    constraints: ConstraintSet
    evaluation_plan: EvaluationPlan
    initial_state: InitialState
    task_graph: TaskGraph
    difficulty_profile: DifficultyProfile
    solver_hint: SolverHint
    metadata: TaskMetadata
    validation_status: ValidationStatus


class TaskCompilationReport(BaseModel):
    """Structured compile report for the task formalization layer."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["compiled", "compiled_with_warnings", "invalid"]
    field_sources: dict[str, str] = Field(default_factory=dict)
    unresolved_dependencies: list[str] = Field(default_factory=list)
    derived_fields: list[str] = Field(default_factory=list)
    validation_errors: list[TaskValidationIssue] = Field(default_factory=list)
    validation_warnings: list[TaskValidationIssue] = Field(default_factory=list)
    acceptance_summary: dict[str, float | int | str] = Field(default_factory=dict)

    @field_validator("field_sources")
    @classmethod
    def validate_sources(cls, values: dict[str, str]) -> dict[str, str]:
        invalid = {key: value for key, value in values.items() if value not in FIELD_SOURCE_VALUES}
        if invalid:
            raise ValueError(f"unsupported field source values: {invalid}")
        return dict(sorted(values.items()))

    @field_validator("unresolved_dependencies", "derived_fields")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class TaskCompileResponse(BaseModel):
    """Top-level compile response for DesignTask compilation."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["compiled", "compiled_with_warnings", "invalid"]
    design_task: DesignTask | None = None
    report: TaskCompilationReport
