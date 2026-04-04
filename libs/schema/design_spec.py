"""Schemas for the interaction layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CIRCUIT_FAMILIES = (
    "two_stage_ota",
    "folded_cascode_ota",
    "telescopic_ota",
    "comparator",
    "ldo",
    "bandgap",
    "unknown",
)

OBJECTIVE_METRICS = (
    "dc_gain_db",
    "gbw_hz",
    "phase_margin_deg",
    "slew_rate_v_per_us",
    "power_w",
    "area_um2",
    "noise_nv_per_sqrt_hz",
)

CONSTRAINT_METRICS = (
    "dc_gain_db",
    "gbw_hz",
    "phase_margin_deg",
    "power_w",
    "slew_rate_v_per_us",
    "input_referred_noise_nv_per_sqrt_hz",
    "output_swing_v",
    "input_common_mode_v",
)

CORNER_ORDER = ("tt", "ss", "ff", "sf", "fs")
TESTBENCH_ORDER = ("op", "ac", "tran", "noise")
FIELD_SOURCE_VALUES = ("user_provided", "system_inferred", "defaulted")

DESIGN_VARIABLES_BY_FAMILY: dict[str, list[str]] = {
    "two_stage_ota": ["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
    "folded_cascode_ota": ["w_in", "l_in", "w_cas", "l_cas", "ibias", "cc"],
    "telescopic_ota": ["w_in", "l_in", "w_cas", "l_cas", "ibias", "vcm"],
    "comparator": ["w_in", "l_in", "w_latch", "l_latch", "ibias"],
    "ldo": ["w_pass", "l_pass", "w_err", "l_err", "ibias", "c_comp"],
    "bandgap": ["area_ratio", "r1", "r2", "w_core", "l_core", "ibias"],
    "unknown": [],
}

MISSING_INFO_ORDER = (
    "circuit_family",
    "process_node",
    "load_cap_f",
    "input_common_mode_v",
    "output_swing_v",
    "temperature_range",
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
    index = {value: position for position, value in enumerate(order)}
    return sorted(deduped, key=lambda item: (index.get(item, len(index)), item))


class MetricRange(BaseModel):
    """Normalized hard-constraint range for a metric."""

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None
    target: float | None = None
    priority: Literal["hard"] = "hard"

    @model_validator(mode="after")
    def validate_range(self) -> "MetricRange":
        if self.min is None and self.max is None and self.target is None:
            raise ValueError("at least one of min, max, or target must be provided")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min must be less than or equal to max")
        return self


class Objectives(BaseModel):
    """Optimization objectives derived from natural-language intents."""

    model_config = ConfigDict(extra="forbid")

    maximize: list[str] = Field(default_factory=list)
    minimize: list[str] = Field(default_factory=list)

    @field_validator("maximize", "minimize")
    @classmethod
    def validate_metrics(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in OBJECTIVE_METRICS]
        if invalid:
            raise ValueError(f"unsupported objective metrics: {invalid}")
        return _ordered_unique(values, OBJECTIVE_METRICS)

    @model_validator(mode="after")
    def validate_overlap(self) -> "Objectives":
        overlap = set(self.maximize) & set(self.minimize)
        if overlap:
            raise ValueError(f"metrics cannot be both maximized and minimized: {sorted(overlap)}")
        return self


class Environment(BaseModel):
    """Normalized environment and load conditions."""

    model_config = ConfigDict(extra="forbid")

    temperature_c: list[float] = Field(default_factory=list)
    corners: list[str] = Field(default_factory=list)
    load_cap_f: float | None = None
    output_load_ohm: float | None = None
    supply_voltage_v: float | None = None

    @field_validator("temperature_c")
    @classmethod
    def canonicalize_temperatures(cls, values: list[float]) -> list[float]:
        return sorted({float(value) for value in values})

    @field_validator("corners")
    @classmethod
    def canonicalize_corners(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in CORNER_ORDER]
        if invalid:
            raise ValueError(f"unsupported process corners: {invalid}")
        return _ordered_unique(values, CORNER_ORDER)


class DesignSpec(BaseModel):
    """Single-source-of-truth interaction-layer output."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    circuit_family: Literal[
        "two_stage_ota",
        "folded_cascode_ota",
        "telescopic_ota",
        "comparator",
        "ldo",
        "bandgap",
        "unknown",
    ]
    process_node: str | None = None
    supply_voltage_v: float | None = None
    objectives: Objectives = Field(default_factory=Objectives)
    hard_constraints: dict[str, MetricRange] = Field(default_factory=dict)
    environment: Environment = Field(default_factory=Environment)
    testbench_plan: list[str] = Field(default_factory=lambda: ["op"])
    design_variables: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    compile_confidence: float = 0.0

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task_id must not be empty")
        return value

    @field_validator("process_node")
    @classmethod
    def normalize_process_node(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace(" ", "").lower()
        if not normalized.endswith("nm") or not normalized[:-2].isdigit():
            raise ValueError("process_node must use the canonical format like 65nm")
        return normalized

    @field_validator("hard_constraints")
    @classmethod
    def validate_constraint_keys(cls, values: dict[str, MetricRange]) -> dict[str, MetricRange]:
        invalid = [key for key in values if key not in CONSTRAINT_METRICS]
        if invalid:
            raise ValueError(f"unsupported hard constraints: {invalid}")
        return {key: values[key] for key in sorted(values, key=CONSTRAINT_METRICS.index)}

    @field_validator("testbench_plan")
    @classmethod
    def validate_testbench_plan(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in TESTBENCH_ORDER]
        if invalid:
            raise ValueError(f"unsupported testbench analyses: {invalid}")
        return _ordered_unique(values, TESTBENCH_ORDER)

    @field_validator("design_variables")
    @classmethod
    def canonicalize_design_variables(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)

    @field_validator("missing_information")
    @classmethod
    def canonicalize_missing_information(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values, MISSING_INFO_ORDER)

    @field_validator("notes")
    @classmethod
    def canonicalize_notes(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)

    @field_validator("compile_confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("compile_confidence must be between 0 and 1")
        return round(float(value), 4)


class ValidationIssue(BaseModel):
    """Structured validation issue."""

    model_config = ConfigDict(extra="forbid")

    code: str
    path: str
    message: str
    severity: Literal["error", "warning"] = "error"


class ValidationReport(BaseModel):
    """Deterministic validation result for a DesignSpec."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    error_types: list[str] = Field(default_factory=list)


class ClarificationRequest(BaseModel):
    """Structured clarification request for interactive mode."""

    model_config = ConfigDict(extra="forbid")

    missing_information: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)


class CompilationReport(BaseModel):
    """Structured compiler report accompanying each compile run."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["compiled", "clarification_required", "invalid"]
    mode: Literal["strict", "interactive"]
    field_sources: dict[str, str] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    suggested_clarifications: list[str] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    parser_notes: list[str] = Field(default_factory=list)
    repair_attempts: int = 0

    @field_validator("field_sources")
    @classmethod
    def validate_field_sources(cls, values: dict[str, str]) -> dict[str, str]:
        invalid = {key: value for key, value in values.items() if value not in FIELD_SOURCE_VALUES}
        if invalid:
            raise ValueError(f"unsupported field sources: {invalid}")
        return dict(sorted(values.items()))

    @field_validator("missing_fields")
    @classmethod
    def validate_missing_fields(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values, MISSING_INFO_ORDER)

    @field_validator("ambiguities", "suggested_clarifications", "parser_notes")
    @classmethod
    def dedupe_strings(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class CompileResponse(BaseModel):
    """Top-level compile response used by APIs and tests."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["strict", "interactive"]
    status: Literal["compiled", "clarification_required", "invalid"]
    design_spec: DesignSpec | None = None
    report: CompilationReport
    clarification_request: ClarificationRequest | None = None
