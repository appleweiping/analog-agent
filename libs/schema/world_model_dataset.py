"""Formal dataset and training schemas for trainable world-model upgrades."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from libs.schema.design_spec import CIRCUIT_FAMILIES


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


class DatasetMetricValue(BaseModel):
    """One metric value carried in dataset records."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    value: float


class WorldModelDatasetRecord(BaseModel):
    """One trainable-surrogate record exported from verified experiments."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    dataset_split: Literal["train", "eval"]
    source_kind: Literal["experiment_verification"]
    source_run_id: str
    task_id: str
    family: str
    mode: str
    candidate_id: str
    fidelity_level: str
    truth_level: str
    validation_status: str
    feasibility_status: str
    dominant_failure_mode: str
    runtime_sec: float
    parameter_values: dict[str, float | int | str | bool] = Field(default_factory=dict)
    normalized_parameters: dict[str, float] = Field(default_factory=dict)
    environment: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    predicted_metrics: list[DatasetMetricValue] = Field(default_factory=list)
    measured_metrics: list[DatasetMetricValue] = Field(default_factory=list)
    prediction_gap: list[DatasetMetricValue] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)

    @field_validator("family")
    @classmethod
    def validate_family(cls, value: str) -> str:
        if value not in CIRCUIT_FAMILIES:
            raise ValueError(f"unsupported family: {value}")
        return value

    @field_validator("artifact_refs")
    @classmethod
    def dedupe_artifacts(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class FamilyDatasetSummary(BaseModel):
    """Aggregate dataset coverage for one circuit family."""

    model_config = ConfigDict(extra="forbid")

    family: str
    record_count: int
    train_count: int
    eval_count: int
    modes: list[str] = Field(default_factory=list)
    target_metrics: list[str] = Field(default_factory=list)

    @field_validator("family")
    @classmethod
    def validate_family(cls, value: str) -> str:
        if value not in CIRCUIT_FAMILIES:
            raise ValueError(f"unsupported family: {value}")
        return value

    @field_validator("modes", "target_metrics")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class WorldModelDatasetBundle(BaseModel):
    """Structured dataset bundle for trainable surrogate work."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    schema_version: str = "world-model-dataset-v1"
    created_at: str
    source_scope: Literal["experiment_suite", "benchmark_multitask"]
    sampling_policy: Literal["family_as_observed", "family_balanced_cap"]
    split_policy: str
    source_run_ids: list[str] = Field(default_factory=list)
    family_coverage: list[str] = Field(default_factory=list)
    feature_keys: list[str] = Field(default_factory=list)
    target_metrics: list[str] = Field(default_factory=list)
    family_summaries: list[FamilyDatasetSummary] = Field(default_factory=list)
    records: list[WorldModelDatasetRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @field_validator("source_run_ids", "family_coverage", "feature_keys", "target_metrics", "notes", "provenance")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)


class SurrogateTrainingConfig(BaseModel):
    """Structured config for the first trainable surrogate baseline."""

    model_config = ConfigDict(extra="forbid")

    name: str
    model_family: Literal["tabular_knn"]
    distance_metric: Literal["weighted_l1"]
    target_metrics: list[str] = Field(default_factory=list)
    k_neighbors: int = 3
    train_fraction: float = 0.8
    minimum_eval_records: int = 1
    family_balanced: bool = False
    uncertainty_mode: Literal["neighbor_spread"] = "neighbor_spread"

    @field_validator("target_metrics")
    @classmethod
    def dedupe_targets(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)

    @field_validator("train_fraction")
    @classmethod
    def validate_fraction(cls, value: float) -> float:
        if value <= 0.0 or value >= 1.0:
            raise ValueError("train_fraction must be within (0, 1)")
        return round(float(value), 6)


class SurrogateMetricSummary(BaseModel):
    """Evaluation summary for one predicted metric."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    mae: float
    relative_mae: float
    mean_target_value: float
    covered_eval_records: int


class SurrogateTrainingRun(BaseModel):
    """Structured training output for the first trainable surrogate baseline."""

    model_config = ConfigDict(extra="forbid")

    training_id: str
    dataset_id: str
    created_at: str
    config: SurrogateTrainingConfig
    training_record_count: int
    evaluation_record_count: int
    feature_keys: list[str] = Field(default_factory=list)
    target_metrics: list[str] = Field(default_factory=list)
    per_metric_summary: list[SurrogateMetricSummary] = Field(default_factory=list)
    overall_mae: float
    overall_relative_mae: float
    model_payload: dict[str, object] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @field_validator("feature_keys", "target_metrics", "notes")
    @classmethod
    def dedupe_lists(cls, values: list[str]) -> list[str]:
        return _ordered_unique(values)
