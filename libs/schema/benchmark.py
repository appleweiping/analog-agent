"""Formal schema for multi-task benchmark design."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from libs.schema.design_spec import CIRCUIT_FAMILIES

EXECUTION_READINESS = ("frozen_runnable", "spec_ready", "planned")
BENCHMARK_ROLES = ("paper_primary", "paper_secondary", "generalization_probe", "future_extension")
TASK_CATEGORIES = ("amplifier", "regulator", "reference")
PHYSICAL_VALIDITY_LEVELS = ("demonstrator_truth", "configured_truth")


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


class BenchmarkFreezePolicy(BaseModel):
    """Version freeze and evolution rule for one benchmark definition."""

    model_config = ConfigDict(extra="forbid")

    frozen_version: str
    paper_track: bool = True
    change_rule: str


class BenchmarkTaskSpec(BaseModel):
    """Family-specific task-facing benchmark specification."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    process_node: str
    supply_voltage_v: float
    objectives: dict[str, list[str]] = Field(default_factory=dict)
    hard_constraints: dict[str, dict[str, float]] = Field(default_factory=dict)
    environment: dict[str, object] = Field(default_factory=dict)
    testbench_plan: list[str] = Field(default_factory=list)
    design_variables: list[str] = Field(default_factory=list)

    @field_validator("design_variables", "testbench_plan")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class BenchmarkMetricContract(BaseModel):
    """Formal measurement and reporting contract for one benchmark."""

    model_config = ConfigDict(extra="forbid")

    primary_metrics: list[str] = Field(default_factory=list)
    auxiliary_metrics: list[str] = Field(default_factory=list)
    reporting_metrics: list[str] = Field(default_factory=list)

    @field_validator("primary_metrics", "auxiliary_metrics", "reporting_metrics")
    @classmethod
    def dedupe_metrics(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class BenchmarkExecutionDefaults(BaseModel):
    """Default execution semantics for one benchmark path."""

    model_config = ConfigDict(extra="forbid")

    backend_preference: str
    default_fidelity: str
    promoted_fidelity: str
    truth_level: Literal["demonstrator_truth", "configured_truth"]
    model_type: Literal["builtin", "external"]


class BenchmarkTaskDefinition(BaseModel):
    """Formal design-only or runnable benchmark task definition."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str
    version: str
    family: str
    category: Literal["amplifier", "regulator", "reference"]
    benchmark_role: Literal["paper_primary", "paper_secondary", "generalization_probe", "future_extension"]
    execution_readiness: Literal["frozen_runnable", "spec_ready", "planned"]
    vertical_slice_bound: bool = False
    physical_validity_target: Literal["demonstrator_truth", "configured_truth"]
    freeze_policy: BenchmarkFreezePolicy
    task: BenchmarkTaskSpec
    measurement_contract: BenchmarkMetricContract
    execution_defaults: BenchmarkExecutionDefaults
    intended_templates: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @field_validator("family")
    @classmethod
    def validate_family(cls, value: str) -> str:
        if value not in CIRCUIT_FAMILIES:
            raise ValueError(f"unsupported circuit family: {value}")
        return value

    @field_validator("intended_templates", mode="before")
    @classmethod
    def sort_template_map(cls, value):
        return dict(sorted((value or {}).items()))

    @field_validator("notes")
    @classmethod
    def dedupe_notes(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class BenchmarkSuiteDefinition(BaseModel):
    """Formal multi-task benchmark suite definition."""

    model_config = ConfigDict(extra="forbid")

    suite_id: str
    version: str
    primary_benchmark_id: str
    benchmark_ids: list[str] = Field(default_factory=list)
    paper_claim_scope: list[str] = Field(default_factory=list)
    reporting_axes: list[str] = Field(default_factory=list)
    supported_modes: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("benchmark_ids", "paper_claim_scope", "reporting_axes", "supported_modes", "notes")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)
