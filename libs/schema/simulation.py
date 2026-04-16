"""Schemas for the ground-truth simulation and verification layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from libs.schema.design_spec import CIRCUIT_FAMILIES
from libs.schema.planning import CandidateRecord, PlanningBundle, SearchState
from libs.schema.stats import VerificationStatsRecord
from libs.schema.world_model import TruthCalibrationRecord

SIMULATION_SCHEMA_VERSION = "simulation-schema-v1"
SIMULATION_BACKENDS = ("ngspice", "xyce", "spectre_compat")
ANALYSIS_TYPES = (
    "op",
    "ac",
    "tran",
    "noise",
    "pvt_sweep",
    "load_sweep",
    "temperature_sweep",
    "monte_carlo",
)
FIDELITY_LEVELS = (
    "quick_truth",
    "focused_truth",
    "focused_validation",
    "full_robustness_certification",
    "targeted_failure_analysis",
)
PRIORITY_CLASSES = ("low", "normal", "high", "critical")
RESOURCE_TIERS = ("low", "medium", "high")
EXECUTION_POLICIES = ("serial", "phase_gated", "parallel_allowed")
FEASIBILITY_STATUSES = (
    "feasible_nominal",
    "feasible_certified",
    "near_feasible",
    "constraint_fail",
    "unstable",
    "robustness_fail",
    "simulation_invalid",
    "measurement_invalid",
)
FAILURE_CLASSES = (
    "none",
    "operating_region_failure",
    "stability_failure",
    "drive_bandwidth_failure",
    "noise_power_area_tradeoff_failure",
    "robustness_failure",
    "netlist_failure",
    "simulator_failure",
    "measurement_failure",
)
COMPLETION_STATUSES = ("success", "partial_success", "netlist_failure", "simulator_failure", "measurement_failure", "timeout")
VALIDATION_ERROR_CODES = (
    "schema_failure",
    "candidate_binding_failure",
    "backend_binding_failure",
    "netlist_render_failure",
    "analysis_plan_failure",
    "measurement_contract_failure",
    "verification_policy_failure",
    "artifact_registry_failure",
    "feedback_contract_failure",
    "execution_failure",
)
BACKEND_ERROR_TYPES = (
    "none",
    "invocation_error",
    "timeout",
    "netlist_error",
    "simulation_error",
    "measurement_error",
    "artifact_error",
)
MEASUREMENT_STATUSES = (
    "measured",
    "missing",
    "indeterminate",
    "analysis_failed",
    "extraction_failed",
)
MEASUREMENT_FAILURE_CODES = (
    "none",
    "analysis_failure",
    "measurement_failure",
    "curve_exists_but_no_crossing",
    "invalid_phase_readout",
    "insufficient_curve_quality",
    "power_unavailable",
    "power_supply_missing",
    "current_direction_ambiguous",
    "no_metric_source",
    "multiple_crossings",
    "partial_analysis_failure",
)
MODEL_TYPES = ("builtin", "external")
MODEL_SOURCE_TYPES = ("path", "registry", "inline")
MODEL_VALIDITY_TIERS = ("demonstrator_truth", "configured_truth")
PHYSICAL_VALIDITY_STATES = ("strong", "weak", "invalid")
ACCEPTANCE_FAILURE_CODES = (
    "schema_failure",
    "execution_failure",
    "measurement_failure",
    "constraint_failure",
    "diagnosis_failure",
    "robustness_failure",
    "feedback_failure",
    "backend_inconsistency",
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


class BackendBinding(BaseModel):
    """Formal simulator backend binding."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["ngspice", "xyce", "spectre_compat"]
    backend_version: str
    capability_map: list[str] = Field(default_factory=list)
    invocation_mode: Literal["native", "compat", "mock_truth"] = "mock_truth"
    support_multi_analysis: bool = True

    @field_validator("capability_map")
    @classmethod
    def dedupe_capabilities(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class IntegrityCheckResult(BaseModel):
    """Structured integrity check for realized netlists."""

    model_config = ConfigDict(extra="forbid")

    check_name: str
    passed: bool
    detail: str


class TemplateBinding(BaseModel):
    """Topology/template binding for one netlist instance."""

    model_config = ConfigDict(extra="forbid")

    template_id: str | None = None
    template_version: str | None = None
    topology_mode: Literal["fixed", "template_family", "search_space"]
    circuit_family: str

    @field_validator("circuit_family")
    @classmethod
    def validate_family(cls, value: str) -> str:
        if value not in CIRCUIT_FAMILIES:
            raise ValueError(f"unsupported circuit family: {value}")
        return value


class ParameterBinding(BaseModel):
    """Concrete parameter binding rendered into a netlist."""

    model_config = ConfigDict(extra="forbid")

    variable_name: str
    netlist_target: str
    value_si: float | int | str | bool
    units: str
    source: str


class ModelSource(BaseModel):
    """Structured model-source descriptor for physical verification."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["path", "registry", "inline"]
    locator: str
    registry_key: str | None = None
    inline_signature: str | None = None


class ModelValidityLevel(BaseModel):
    """Structured physical-validity tier for one model binding."""

    model_config = ConfigDict(extra="forbid")

    truth_level: Literal["demonstrator_truth", "configured_truth"]
    detail: str
    industrial_confidence: float

    @field_validator("industrial_confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("industrial_confidence must be within [0, 1]")
        return round(float(value), 4)


class ModelBinding(BaseModel):
    """Physical model/environment binding for one simulation."""

    model_config = ConfigDict(extra="forbid")

    model_type: Literal["builtin", "external"]
    model_source: ModelSource
    process_node: str
    corner: str
    temperature_c: float
    supply_voltage_v: float | None = None
    backend_model_ref: str
    binding_confidence: float
    validity_level: ModelValidityLevel

    @field_validator("binding_confidence")
    @classmethod
    def validate_binding_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("binding_confidence must be within [0, 1]")
        return round(float(value), 4)


class StimulusBinding(BaseModel):
    """Structured stimulus definition."""

    model_config = ConfigDict(extra="forbid")

    source_name: str
    stimulus_type: Literal["bias", "ac_input", "pulse", "load", "supply"]
    parameters: dict[str, float | int | str | bool] = Field(default_factory=dict)


class AnalysisStatement(BaseModel):
    """Structured analysis statement for one backend run."""

    model_config = ConfigDict(extra="forbid")

    analysis_type: Literal["op", "ac", "tran", "noise", "pvt_sweep", "load_sweep", "temperature_sweep", "monte_carlo"]
    order: int
    parameters: dict[str, float | int | str | bool] = Field(default_factory=dict)
    required_metrics: list[str] = Field(default_factory=list)

    @field_validator("required_metrics")
    @classmethod
    def dedupe_metrics(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SavePolicy(BaseModel):
    """Structured save policy for raw artifacts."""

    model_config = ConfigDict(extra="forbid")

    save_node_voltages: list[str] = Field(default_factory=list)
    save_branch_currents: list[str] = Field(default_factory=list)
    save_waveforms: bool = True
    save_operating_point: bool = True

    @field_validator("save_node_voltages", "save_branch_currents")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class MeasurementHook(BaseModel):
    """Structured measurement hook bound to one analysis."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    from_analysis: Literal["op", "ac", "tran", "noise", "pvt_sweep", "load_sweep", "temperature_sweep", "monte_carlo"]
    method: str


class NetlistInstance(BaseModel):
    """Formal realized netlist instance."""

    model_config = ConfigDict(extra="forbid")

    netlist_id: str
    template_binding: TemplateBinding
    parameter_binding: list[ParameterBinding] = Field(default_factory=list)
    model_binding: ModelBinding
    stimulus_binding: list[StimulusBinding] = Field(default_factory=list)
    analysis_statements: list[AnalysisStatement] = Field(default_factory=list)
    save_policy: SavePolicy
    measurement_hooks: list[MeasurementHook] = Field(default_factory=list)
    integrity_checks: list[IntegrityCheckResult] = Field(default_factory=list)
    render_status: Literal["ready", "invalid"] = "ready"
    rendered_netlist: str


class AnalysisPlan(BaseModel):
    """Execution-ready analysis plan."""

    model_config = ConfigDict(extra="forbid")

    ordered_analyses: list[AnalysisStatement] = Field(default_factory=list)
    analysis_dependencies: dict[str, list[str]] = Field(default_factory=dict)
    fidelity_level: Literal["quick_truth", "focused_truth", "focused_validation", "full_robustness_certification", "targeted_failure_analysis"]
    execution_policy: Literal["serial", "phase_gated", "parallel_allowed"]
    early_termination_rules: list[str] = Field(default_factory=list)

    @field_validator("early_termination_rules")
    @classmethod
    def dedupe_rules(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class MeasurementDefinition(BaseModel):
    """Structured definition for one measurable simulator-facing metric."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    units: str
    required_analysis_types: list[str] = Field(default_factory=list)
    semantic_role: Literal["performance", "stability", "power", "diagnostic"] = "performance"
    expected_range: list[float] = Field(default_factory=list)

    @field_validator("required_analysis_types")
    @classmethod
    def validate_analysis_types(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in ANALYSIS_TYPES]
        if invalid:
            raise ValueError(f"unsupported required_analysis_types: {invalid}")
        return _ordered_unique(values, ANALYSIS_TYPES)


class ExtractionMethod(BaseModel):
    """Structured extraction method."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    method: str
    from_analysis: Literal["op", "ac", "tran", "noise", "pvt_sweep", "load_sweep", "temperature_sweep", "monte_carlo"]
    preferred_source_field: str | None = None
    failure_conditions: list[str] = Field(default_factory=list)

    @field_validator("failure_conditions")
    @classmethod
    def dedupe_failure_conditions(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class PostprocessingRule(BaseModel):
    """Structured postprocessing rule for extracted measurements."""

    model_config = ConfigDict(extra="forbid")

    rule_name: str
    applies_to_metrics: list[str] = Field(default_factory=list)
    parameters: dict[str, float | int | str | bool] = Field(default_factory=dict)

    @field_validator("applies_to_metrics")
    @classmethod
    def dedupe_metrics(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ValidationCheck(BaseModel):
    """Structured validation check applied to one extracted metric."""

    model_config = ConfigDict(extra="forbid")

    check_name: str
    applies_to_metrics: list[str] = Field(default_factory=list)
    failure_severity: Literal["low", "medium", "high", "critical"] = "medium"
    parameters: dict[str, float | int | str | bool] = Field(default_factory=dict)

    @field_validator("applies_to_metrics")
    @classmethod
    def dedupe_metrics(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class FallbackStrategy(BaseModel):
    """Structured fallback strategy when a metric cannot be extracted cleanly."""

    model_config = ConfigDict(extra="forbid")

    strategy_name: str
    applies_to_metrics: list[str] = Field(default_factory=list)
    trigger_condition: str
    action: str

    @field_validator("applies_to_metrics")
    @classmethod
    def dedupe_metrics(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class MeasurementContract(BaseModel):
    """Formal measurement extraction contract."""

    model_config = ConfigDict(extra="forbid")

    measurement_definitions: list[MeasurementDefinition] = Field(default_factory=list)
    extraction_methods: list[ExtractionMethod] = Field(default_factory=list)
    postprocessing_rules: list[PostprocessingRule] = Field(default_factory=list)
    fallback_strategies: list[FallbackStrategy] = Field(default_factory=list)
    validation_checks: list[ValidationCheck] = Field(default_factory=list)

    @property
    def metric_definitions(self) -> list[MeasurementDefinition]:
        """Backward-compatible access for older call sites."""

        return self.measurement_definitions

    @field_validator("measurement_definitions")
    @classmethod
    def dedupe_definitions(cls, values: list[MeasurementDefinition]) -> list[MeasurementDefinition]:
        seen: set[str] = set()
        ordered: list[MeasurementDefinition] = []
        for value in values:
            if value.metric in seen:
                continue
            seen.add(value.metric)
            ordered.append(value)
        return ordered


class SeverityRule(BaseModel):
    """Structured severity rule for verification failures."""

    model_config = ConfigDict(extra="forbid")

    condition: str
    severity: Literal["low", "medium", "high", "critical"]


class VerificationPolicy(BaseModel):
    """Formal constraint-verification policy."""

    model_config = ConfigDict(extra="forbid")

    constraint_evaluation_rules: list[str] = Field(default_factory=list)
    feasibility_aggregation: Literal["all_hard_constraints", "criticality_weighted", "phase_aware"]
    severity_levels: list[SeverityRule] = Field(default_factory=list)
    pass_criteria: list[str] = Field(default_factory=list)

    @field_validator("constraint_evaluation_rules", "pass_criteria")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class MonteCarloSettings(BaseModel):
    """Structured Monte Carlo policy."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sample_count: int = 0
    sigma_level: float = 3.0


class RobustnessPolicy(BaseModel):
    """Formal robustness-verification policy."""

    model_config = ConfigDict(extra="forbid")

    required_corners: list[str] = Field(default_factory=list)
    temperature_range_c: list[float] = Field(default_factory=list)
    load_conditions: list[float] = Field(default_factory=list)
    monte_carlo_settings: MonteCarloSettings = Field(default_factory=MonteCarloSettings)
    escalation_policy: list[str] = Field(default_factory=list)

    @field_validator("required_corners", "escalation_policy")
    @classmethod
    def dedupe_strings(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ArtifactRecord(BaseModel):
    """Structured artifact entry."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_type: Literal["netlist", "stdout", "stderr", "raw_waveform", "measurement_table", "verification_report"]
    path: str
    simulation_provenance: SimulationProvenance | None = None
    validation_status: ValidationStatus | None = None
    execution_context: ArtifactExecutionContext | None = None


class ValidationStatus(BaseModel):
    """Physical-validity status for one simulation execution/result."""

    model_config = ConfigDict(extra="forbid")

    truth_level: Literal["demonstrator_truth", "configured_truth"]
    validity_state: Literal["strong", "weak", "invalid"]
    model_binding_present: bool
    summary: str
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings")
    @classmethod
    def dedupe_warnings(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ArtifactExecutionContext(BaseModel):
    """Replay-oriented execution context attached to persisted artifacts."""

    model_config = ConfigDict(extra="forbid")

    paper_mode: bool = False
    paper_safe: bool = False
    replayable: bool = False
    resolved_simulator_binary: str | None = None
    backend_error_type: str | None = None
    execution_runtime_sec: float | None = None
    replay_hint: str | None = None


class PaperTruthPolicy(BaseModel):
    """Formal policy describing which truth path is acceptable for paper-facing runs."""

    model_config = ConfigDict(extra="forbid")

    policy_name: str
    paper_mode: bool = False
    native_truth_required: bool = True
    configured_truth_preferred: bool = True
    allow_demonstrator_truth: bool = True
    forbid_mock_truth: bool = True
    summary: str


class SimulationProvenance(BaseModel):
    """Structured provenance for fifth-layer truth results."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["ngspice", "xyce", "spectre_compat"]
    backend_version: str
    invocation_mode: Literal["native", "compat", "mock_truth"]
    fidelity_level: Literal["quick_truth", "focused_truth", "full_robustness_certification", "targeted_failure_analysis"]
    truth_level: Literal["demonstrator_truth", "configured_truth"]
    model_binding: ModelBinding
    resolved_simulator_binary: str | None = None
    paper_mode: bool = False
    paper_safe: bool = False
    artifact_lineage: list[str] = Field(default_factory=list)
    provenance_tags: list[str] = Field(default_factory=list)

    @field_validator("artifact_lineage", "provenance_tags")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ArtifactRegistry(BaseModel):
    """Registry of persisted simulation artifacts."""

    model_config = ConfigDict(extra="forbid")

    run_directory: str
    records: list[ArtifactRecord] = Field(default_factory=list)


class BackendRunRequest(BaseModel):
    """Structured backend execution request."""

    model_config = ConfigDict(extra="forbid")

    simulator_binary_path: str
    netlist_path: str
    log_path: str
    timeout_sec: int
    working_directory: str | None = None
    environment_overrides: dict[str, str] = Field(default_factory=dict)
    fidelity_tag: str


class BackendRunResult(BaseModel):
    """Structured backend execution response."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    returncode: int | None = None
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    log_exists: bool = False
    log_path: str
    runtime_sec: float | None = None
    error_type: str = "none"
    raw_completion_status: str

    @field_validator("error_type")
    @classmethod
    def validate_error_type(cls, value: str) -> str:
        if value not in BACKEND_ERROR_TYPES:
            raise ValueError(f"unsupported backend error type: {value}")
        return value


class FidelityLevel(BaseModel):
    """Formal fidelity-level object used by the fifth layer."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["quick_truth", "focused_truth", "focused_validation", "full_robustness_certification", "targeted_failure_analysis"]
    canonical_name: Literal["quick_truth", "focused_truth", "full_robustness_certification", "targeted_failure_analysis"]
    purpose: str


class FidelitySelectionReason(BaseModel):
    """Structured reason for fidelity selection or escalation."""

    model_config = ConfigDict(extra="forbid")

    reason_code: str
    detail: str | None = None


class ExecutionPolicy(BaseModel):
    """Formal execution policy derived from fidelity selection."""

    model_config = ConfigDict(extra="forbid")

    policy_name: str
    ordered_analysis_scope: list[str] = Field(default_factory=list)
    measurement_targets: list[str] = Field(default_factory=list)
    early_termination_rules: list[str] = Field(default_factory=list)

    @field_validator("ordered_analysis_scope")
    @classmethod
    def validate_scope(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in ANALYSIS_TYPES]
        if invalid:
            raise ValueError(f"unsupported ordered_analysis_scope: {invalid}")
        return _ordered_unique(values, ANALYSIS_TYPES)

    @field_validator("measurement_targets", "early_termination_rules")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class FidelityPolicy(BaseModel):
    """Formal policy describing supported fidelity levels and escalation behavior."""

    model_config = ConfigDict(extra="forbid")

    default_fidelity: Literal["quick_truth", "focused_truth"]
    supported_fidelities: list[FidelityLevel] = Field(default_factory=list)
    escalation_rules: list[str] = Field(default_factory=list)

    @field_validator("escalation_rules")
    @classmethod
    def dedupe_rules(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class VerificationExecutionProfile(BaseModel):
    """Resolved execution profile attached to a compiled verification bundle/result."""

    model_config = ConfigDict(extra="forbid")

    requested_fidelity: str
    resolved_fidelity: Literal["quick_truth", "focused_truth", "full_robustness_certification", "targeted_failure_analysis"]
    execution_policy: ExecutionPolicy
    selection_reason: FidelitySelectionReason
    provenance: list[str] = Field(default_factory=list)

    @field_validator("provenance")
    @classmethod
    def dedupe_provenance(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class EscalationAdvice(BaseModel):
    """Structured advice emitted back to the planner about fidelity escalation."""

    model_config = ConfigDict(extra="forbid")

    advice: Literal["retain_current_fidelity", "upgrade_to_focused_truth", "repeat_focused_truth", "defer_escalation"]
    recommended_fidelity: Literal["quick_truth", "focused_truth", "full_robustness_certification", "targeted_failure_analysis"]
    escalation_reason: str
    confidence: float = 0.5

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be within [0, 1]")
        return round(float(value), 4)


class ServiceMethodSpec(BaseModel):
    """Formal service-method signature for the fifth layer."""

    model_config = ConfigDict(extra="forbid")

    input_schema: str
    output_schema: str
    deterministic: bool = True


class SimulationServingContract(BaseModel):
    """Formal serving contract for the fifth layer."""

    model_config = ConfigDict(extra="forbid")

    realize_netlist: ServiceMethodSpec
    run_simulation: ServiceMethodSpec
    extract_measurements: ServiceMethodSpec
    verify_constraints: ServiceMethodSpec
    certify_robustness: ServiceMethodSpec
    emit_calibration_feedback: ServiceMethodSpec
    archive_artifacts: ServiceMethodSpec
    validate_backend: ServiceMethodSpec


class FeedbackContract(BaseModel):
    """Formal feedback-emission contract."""

    model_config = ConfigDict(extra="forbid")

    emit_calibration_feedback: bool = True
    emit_planner_feedback: bool = True
    emit_failure_diagnostics: bool = True


class SimulationMetadata(BaseModel):
    """Audit metadata for fifth-layer bundles."""

    model_config = ConfigDict(extra="forbid")

    created_by_layer: Literal["simulation_layer"] = "simulation_layer"
    compile_timestamp: str
    source_task_signature: str
    source_candidate_signature: str
    implementation_version: str
    paper_truth_policy: PaperTruthPolicy | None = None
    assumptions: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @field_validator("assumptions", "provenance")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SimulationValidationIssue(BaseModel):
    """Validation issue for simulation-layer objects."""

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


class SimulationValidationStatus(BaseModel):
    """Embedded validation state for simulation objects."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    errors: list[SimulationValidationIssue] = Field(default_factory=list)
    warnings: list[SimulationValidationIssue] = Field(default_factory=list)
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


class ResourceBudget(BaseModel):
    """Structured resource budget for one simulation request."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int
    max_parallel_analyses: int = 1
    resource_tier: Literal["low", "medium", "high"] = "medium"


class SimulationRequest(BaseModel):
    """Formal request for ground-truth verification."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    task_id: str
    candidate_id: str
    world_state_ref: str
    planner_context_ref: str
    analysis_scope: list[str] = Field(default_factory=list)
    fidelity_level: Literal["quick_truth", "focused_truth", "focused_validation", "full_robustness_certification", "targeted_failure_analysis"]
    backend_preference: Literal["ngspice", "xyce", "spectre_compat"]
    environment_overrides: dict[str, float | int | str | bool] = Field(default_factory=dict)
    measurement_targets: list[str] = Field(default_factory=list)
    escalation_reason: str
    priority_class: Literal["low", "normal", "high", "critical"] = "normal"
    resource_budget: ResourceBudget
    model_binding: ModelBinding | None = None
    provenance: list[str] = Field(default_factory=list)

    @field_validator("analysis_scope")
    @classmethod
    def validate_scope(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in ANALYSIS_TYPES]
        if invalid:
            raise ValueError(f"unsupported analysis scope values: {invalid}")
        return _ordered_unique(values, ANALYSIS_TYPES)

    @field_validator("measurement_targets", "provenance")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SimulationBundle(BaseModel):
    """Top-level formal verification bundle."""

    model_config = ConfigDict(extra="forbid")

    simulation_id: str
    schema_version: str = SIMULATION_SCHEMA_VERSION
    parent_task_id: str
    candidate_id: str
    planner_context_ref: str
    backend_binding: BackendBinding
    model_binding: ModelBinding
    simulation_provenance: SimulationProvenance
    netlist_instance: NetlistInstance
    analysis_plan: AnalysisPlan
    measurement_contract: MeasurementContract
    fidelity_policy: FidelityPolicy
    execution_profile: VerificationExecutionProfile
    verification_policy: VerificationPolicy
    robustness_policy: RobustnessPolicy
    artifact_registry: ArtifactRegistry
    feedback_contract: FeedbackContract
    serving_contract: SimulationServingContract
    metadata: SimulationMetadata
    validation_status: SimulationValidationStatus


class AnalysisExecutionRecord(BaseModel):
    """Structured execution record for one analysis."""

    model_config = ConfigDict(extra="forbid")

    analysis_type: Literal["op", "ac", "tran", "noise", "pvt_sweep", "load_sweep", "temperature_sweep", "monte_carlo"]
    success: bool
    runtime_ms: int
    backend_status: str
    artifact_refs: list[str] = Field(default_factory=list)

    @field_validator("artifact_refs")
    @classmethod
    def dedupe_refs(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class MeasurementStatus(BaseModel):
    """Structured status for one extracted measurement."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["measured", "missing", "indeterminate", "analysis_failed", "extraction_failed"]
    detail: str | None = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in MEASUREMENT_STATUSES:
            raise ValueError(f"unsupported measurement status: {value}")
        return value


class MeasurementFailureReason(BaseModel):
    """Structured reason describing why extraction failed or is unreliable."""

    model_config = ConfigDict(extra="forbid")

    code: Literal[
        "none",
        "analysis_failure",
        "measurement_failure",
        "curve_exists_but_no_crossing",
        "invalid_phase_readout",
        "insufficient_curve_quality",
        "power_unavailable",
        "power_supply_missing",
        "current_direction_ambiguous",
        "no_metric_source",
        "multiple_crossings",
        "partial_analysis_failure",
    ] = "none"
    detail: str | None = None

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if value not in MEASUREMENT_FAILURE_CODES:
            raise ValueError(f"unsupported measurement failure code: {value}")
        return value


class MeasuredMetric(BaseModel):
    """Structured measured metric."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    value: float
    units: str
    source_analysis: str
    extraction_confidence: float

    @field_validator("extraction_confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("extraction_confidence must be within [0, 1]")
        return round(float(value), 4)


class MeasurementResult(BaseModel):
    """Formal extraction result for one metric, including failure semantics."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    units: str
    source_analysis: str
    status: MeasurementStatus
    failure_reason: MeasurementFailureReason = Field(default_factory=MeasurementFailureReason)
    value: float | None = None
    raw_value: float | None = None
    postprocessed_value: float | None = None
    confidence: float = 0.0
    provenance: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be within [0, 1]")
        return round(float(value), 4)

    @field_validator("provenance")
    @classmethod
    def dedupe_provenance(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class MeasurementReport(BaseModel):
    """Structured measurement report."""

    model_config = ConfigDict(extra="forbid")

    report_id: str
    candidate_id: str
    executed_analyses: list[AnalysisExecutionRecord] = Field(default_factory=list)
    measurement_results: list[MeasurementResult] = Field(default_factory=list)
    measured_metrics: list[MeasuredMetric] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)
    validation_checks: list[IntegrityCheckResult] = Field(default_factory=list)

    @field_validator("extraction_notes")
    @classmethod
    def dedupe_notes(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ConstraintAssessmentRecord(BaseModel):
    """Structured constraint assessment against measured truth."""

    model_config = ConfigDict(extra="forbid")

    constraint_name: str
    constraint_group: str
    metric: str
    is_satisfied: bool
    assessment_basis: Literal["measurement_value", "measurement_unavailable"] = "measurement_value"
    measured_value: float | None = None
    margin: float
    severity: Literal["low", "medium", "high", "critical"]
    measurement_status: str = "measured"
    measurement_failure_reason: str = "none"


class FailureAttribution(BaseModel):
    """Structured failure attribution object."""

    model_config = ConfigDict(extra="forbid")

    primary_failure_class: Literal[
        "none",
        "design_failure",
        "simulation_invalid",
        "analysis_failure",
        "operating_region_failure",
        "stability_failure",
        "drive_bandwidth_failure",
        "noise_power_area_tradeoff_failure",
        "robustness_failure",
        "netlist_failure",
        "simulator_failure",
        "measurement_failure",
    ] = "none"
    contributing_factors: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    recommended_focus: list[str] = Field(default_factory=list)

    @field_validator("contributing_factors", "evidence", "recommended_focus")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class RobustnessCertificate(BaseModel):
    """Structured robustness summary/certificate."""

    model_config = ConfigDict(extra="forbid")

    certificate_id: str
    certification_status: Literal["not_applicable", "nominal_only", "partial_robust", "robust_certified", "robustness_failed"]
    evaluated_conditions: list[str] = Field(default_factory=list)
    pass_rate: float
    weakest_condition: str | None = None
    worst_case_margin: float | None = None
    summary: list[str] = Field(default_factory=list)

    @field_validator("pass_rate")
    @classmethod
    def validate_rate(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("pass_rate must be within [0, 1]")
        return round(float(value), 4)

    @field_validator("evaluated_conditions", "summary")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class CalibrationFeedback(BaseModel):
    """Structured calibration feedback emitted to the third layer."""

    model_config = ConfigDict(extra="forbid")

    calibration_id: str
    candidate_id: str
    world_model_ref: str
    truth_record: TruthCalibrationRecord
    residual_metrics: dict[str, float] = Field(default_factory=dict)
    constraint_disagreement: list[str] = Field(default_factory=list)
    trust_violation_flags: list[str] = Field(default_factory=list)
    usable_region_update: list[str] = Field(default_factory=list)
    retrain_priority: Literal["low", "medium", "high"] = "medium"
    provenance: list[str] = Field(default_factory=list)

    @field_validator("constraint_disagreement", "trust_violation_flags", "usable_region_update", "provenance")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class PlannerFeedback(BaseModel):
    """Structured feedback emitted to the planning layer."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    lifecycle_update: Literal["verified", "rejected", "needs_more_simulation", "boundary_candidate"]
    strategy_correction: list[str] = Field(default_factory=list)
    phase_hint: str | None = None
    trust_alerts: list[str] = Field(default_factory=list)
    feedback_basis: Literal["design_failure", "measurement_failure", "simulation_failure", "verification_success"] = "verification_success"
    escalation_advice: list[EscalationAdvice] = Field(default_factory=list)
    recommended_fidelity: Literal["quick_truth", "focused_truth", "full_robustness_certification", "targeted_failure_analysis"] | None = None
    escalation_reason: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)

    @field_validator("strategy_correction", "trust_alerts", "artifact_refs")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class VerificationResult(BaseModel):
    """Top-level physical verification result."""

    model_config = ConfigDict(extra="forbid")

    result_id: str
    candidate_id: str
    executed_fidelity: Literal["quick_truth", "focused_truth", "full_robustness_certification", "targeted_failure_analysis"]
    backend_signature: str
    simulation_provenance: SimulationProvenance
    validation_status: ValidationStatus
    execution_profile: VerificationExecutionProfile
    measurement_report: MeasurementReport
    constraint_assessment: list[ConstraintAssessmentRecord] = Field(default_factory=list)
    feasibility_status: Literal[
        "feasible_nominal",
        "feasible_certified",
        "near_feasible",
        "constraint_fail",
        "unstable",
        "robustness_fail",
        "simulation_invalid",
        "measurement_invalid",
    ]
    failure_attribution: FailureAttribution
    robustness_summary: RobustnessCertificate
    diagnostic_features: dict[str, float | int | str | bool] = Field(default_factory=dict)
    artifact_refs: list[str] = Field(default_factory=list)
    calibration_payload: CalibrationFeedback
    planner_feedback: PlannerFeedback
    completion_status: Literal["success", "partial_success", "netlist_failure", "simulator_failure", "measurement_failure", "timeout"]

    @field_validator("artifact_refs")
    @classmethod
    def dedupe_artifacts(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SimulationCompilationReport(BaseModel):
    """Structured compile report for the fifth layer."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["compiled", "compiled_with_warnings", "invalid"]
    derived_fields: list[str] = Field(default_factory=list)
    validation_errors: list[SimulationValidationIssue] = Field(default_factory=list)
    validation_warnings: list[SimulationValidationIssue] = Field(default_factory=list)
    acceptance_summary: dict[str, float | int | str] = Field(default_factory=dict)

    @field_validator("derived_fields")
    @classmethod
    def dedupe_fields(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SimulationCompileResponse(BaseModel):
    """Top-level compile response for SimulationBundle."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["compiled", "compiled_with_warnings", "invalid"]
    simulation_bundle: SimulationBundle | None = None
    report: SimulationCompilationReport


class BackendValidationReport(BaseModel):
    """Validation report for one backend binding."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["ngspice", "xyce", "spectre_compat"]
    is_available: bool
    invocation_mode: Literal["native", "compat", "mock_truth"]
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings")
    @classmethod
    def dedupe_warnings(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SimulationExecutionResponse(BaseModel):
    """Top-level execution response for the fifth layer."""

    model_config = ConfigDict(extra="forbid")

    simulation_bundle: SimulationBundle
    simulation_request: SimulationRequest
    backend_report: BackendValidationReport
    verification_result: VerificationResult
    verification_stats: VerificationStatsRecord


class SimulationAcceptanceFailureRecord(BaseModel):
    """Structured fifth-layer acceptance failure taxonomy."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if value not in ACCEPTANCE_FAILURE_CODES:
            raise ValueError(f"unsupported acceptance failure code: {value}")
        return value
