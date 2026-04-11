"""Formal statistics schemas for research reporting and benchmark export."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


class MetricGapRecord(BaseModel):
    """Structured prediction-versus-truth gap for one metric."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    predicted_value: float | None = None
    ground_truth_value: float | None = None
    absolute_error: float | None = None
    relative_error: float | None = None


class PredictionGapStatsSummary(BaseModel):
    """Structured summary of prediction-ground-truth gap statistics."""

    model_config = ConfigDict(extra="forbid")

    record_count: int
    metrics_covered: list[str] = Field(default_factory=list)
    average_absolute_error: dict[str, float] = Field(default_factory=dict)
    average_relative_error: dict[str, float] = Field(default_factory=dict)
    by_task_id: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_family: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_fidelity: dict[str, dict[str, float]] = Field(default_factory=dict)

    @field_validator("metrics_covered")
    @classmethod
    def dedupe_metrics(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class FidelityStatsSummary(BaseModel):
    """Structured fidelity-usage and escalation summary."""

    model_config = ConfigDict(extra="forbid")

    total_real_verifications: int
    quick_truth_calls: int
    focused_truth_calls: int
    escalation_count: int
    escalation_reason_distribution: dict[str, int] = Field(default_factory=dict)
    focused_truth_ratio: float
    fidelity_escalation_rate: float

    @field_validator("focused_truth_ratio", "fidelity_escalation_rate")
    @classmethod
    def validate_rate(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("fidelity rates must be within [0, 1]")
        return round(float(value), 6)


class FailureModeStatsSummary(BaseModel):
    """Structured failure-mode frequency summary."""

    model_config = ConfigDict(extra="forbid")

    total_records: int
    frequency: dict[str, int] = Field(default_factory=dict)
    ratio: dict[str, float] = Field(default_factory=dict)
    by_fidelity: dict[str, dict[str, int]] = Field(default_factory=dict)
    by_truth_level: dict[str, dict[str, int]] = Field(default_factory=dict)
    by_family: dict[str, dict[str, int]] = Field(default_factory=dict)


class VerificationStatsRecord(BaseModel):
    """Structured statistics record for one real verification run."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    candidate_id: str
    task_id: str
    family: str
    truth_level: str
    fidelity_level: str
    analysis_types: list[str] = Field(default_factory=list)
    runtime_sec: float
    measured_metrics: dict[str, float] = Field(default_factory=dict)
    measurement_statuses: dict[str, str] = Field(default_factory=dict)
    feasibility_status: str
    dominant_failure_mode: str
    prediction_ground_truth_gap: list[MetricGapRecord] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    planner_escalation_reason: str | None = None
    validation_status: str

    @field_validator("analysis_types", "artifact_refs")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class ExperimentStatsRecord(BaseModel):
    """Structured statistics record for one experiment run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str
    mode: Literal["full_simulation_baseline", "no_world_model_baseline", "full_system"]
    simulation_call_count: int
    candidate_count: int
    best_feasible_found: bool
    convergence_step: int | None = None
    verification_records: list[VerificationStatsRecord] = Field(default_factory=list)
    prediction_gap_summary: PredictionGapStatsSummary
    fidelity_summary: FidelityStatsSummary
    failure_mode_summary: FailureModeStatsSummary


class StatsAggregationResult(BaseModel):
    """Structured aggregate summary for acceptance / experiment / benchmark scopes."""

    model_config = ConfigDict(extra="forbid")

    aggregation_scope: Literal["acceptance_run", "experiment_suite", "benchmark_suite"]
    source_ids: list[str] = Field(default_factory=list)
    total_real_simulation_calls: int
    measurement_success_rate: float
    feasible_verification_rate: float
    predicted_feasible_to_verified_feasible_conversion: float
    simulation_reduction_ratio: float | None = None
    average_prediction_gap: PredictionGapStatsSummary
    dominant_failure_mode_distribution: FailureModeStatsSummary
    fidelity_summary: FidelityStatsSummary
    failure_mode_summary: FailureModeStatsSummary
    prediction_gap_summary: PredictionGapStatsSummary

    @field_validator(
        "measurement_success_rate",
        "feasible_verification_rate",
        "predicted_feasible_to_verified_feasible_conversion",
    )
    @classmethod
    def validate_rate(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("aggregation rates must be within [0, 1]")
        return round(float(value), 6)

    @field_validator("simulation_reduction_ratio")
    @classmethod
    def validate_optional_rate(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if value < -1.0 or value > 1.0:
            raise ValueError("simulation_reduction_ratio must be within [-1, 1]")
        return round(float(value), 6)

    @field_validator("source_ids")
    @classmethod
    def dedupe_source_ids(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)
