"""Formal schemas for repeated-episode memory evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from libs.schema.paper_evidence import FigureSpec, TableSpec

MEMORY_ABLATION_MODES = (
    "no_memory",
    "full_memory",
)
MEMORY_TRANSFER_MODES = (
    "no_memory",
    "governed_transfer",
    "forced_transfer",
)


class MemoryEpisodeStatsRecord(BaseModel):
    """Structured per-episode record for memory ablation."""

    model_config = ConfigDict(extra="forbid")

    episode_index: int
    mode: Literal["no_memory", "full_memory"]
    task_id: str
    family: str
    memory_episode_count_before: int = 0
    retrieved_episode_count: int = 0
    advice_count: int = 0
    retrieval_precision_proxy: float = 0.0
    negative_transfer_risk: float = 0.0
    warm_start_applied: bool = False
    warm_start_source: str | None = None
    best_candidate_id: str | None = None
    best_feasible_found: bool = False
    real_simulation_calls: int = 0
    step_to_first_feasible: int | None = None
    dominant_failure_modes: list[str] = Field(default_factory=list)
    repeated_failure_count: int = 0
    episode_memory_id: str | None = None

    @field_validator("retrieval_precision_proxy", "negative_transfer_risk")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("retrieval scores must be within [0, 1]")
        return round(float(value), 6)


class MemoryModeSummary(BaseModel):
    """Aggregated summary for one repeated-episode memory mode."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["no_memory", "full_memory"]
    episode_count: int
    feasible_hit_rate: float
    average_real_simulation_calls: float
    average_step_to_first_feasible: float
    average_repeated_failure_count: float
    warm_start_application_rate: float
    average_retrieval_precision: float
    average_negative_transfer_risk: float

    @field_validator(
        "feasible_hit_rate",
        "warm_start_application_rate",
        "average_retrieval_precision",
        "average_negative_transfer_risk",
    )
    @classmethod
    def validate_rate(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("memory summary rates must be within [0, 1]")
        return round(float(value), 6)


class MemoryAblationSummary(BaseModel):
    """Top-level conclusions for repeated-episode memory ablation."""

    model_config = ConfigDict(extra="forbid")

    memory_reduces_simulation_calls: bool
    memory_reduces_step_to_first_feasible: bool
    memory_reduces_repeated_failures: bool
    memory_uses_retrieval_in_practice: bool
    notes: list[str] = Field(default_factory=list)


class MemoryAblationSuiteResult(BaseModel):
    """Formal repeated-episode ablation result for one vertical slice."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    modes: list[Literal["no_memory", "full_memory"]] = Field(default_factory=list)
    episode_records: list[MemoryEpisodeStatsRecord] = Field(default_factory=list)
    mode_summaries: list[MemoryModeSummary] = Field(default_factory=list)
    summary: MemoryAblationSummary


class MemoryAblationEvidenceBundle(BaseModel):
    """Paper-facing bundle for repeated-episode memory evidence."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    modes: list[Literal["no_memory", "full_memory"]] = Field(default_factory=list)
    figures: list[FigureSpec] = Field(default_factory=list)
    tables: list[TableSpec] = Field(default_factory=list)
    summary: MemoryAblationSummary
    json_output_path: str


class MemoryTransferStatsRecord(BaseModel):
    """Structured per-episode record for cross-task memory transfer."""

    model_config = ConfigDict(extra="forbid")

    source_task_slug: str
    target_task_slug: str
    transfer_kind: Literal["same_family", "cross_family"]
    episode_index: int
    mode: Literal["no_memory", "governed_transfer", "forced_transfer"]
    source_episode_count: int = 0
    retrieved_episode_count: int = 0
    retrieval_precision_proxy: float = 0.0
    negative_transfer_risk: float = 0.0
    warm_start_applied: bool = False
    warm_start_source: str | None = None
    best_feasible_found: bool = False
    real_simulation_calls: int = 0
    step_to_first_feasible: int | None = None
    repeated_failure_count: int = 0
    harmful_transfer_applied: bool = False

    @field_validator("retrieval_precision_proxy", "negative_transfer_risk")
    @classmethod
    def validate_transfer_score(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("transfer scores must be within [0, 1]")
        return round(float(value), 6)


class MemoryTransferModeSummary(BaseModel):
    """Aggregated summary for one cross-task transfer mode."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["no_memory", "governed_transfer", "forced_transfer"]
    episode_count: int
    feasible_hit_rate: float
    average_real_simulation_calls: float
    average_step_to_first_feasible: float
    average_repeated_failure_count: float
    warm_start_application_rate: float
    average_retrieval_precision: float
    average_negative_transfer_risk: float
    harmful_transfer_rate: float

    @field_validator(
        "feasible_hit_rate",
        "warm_start_application_rate",
        "average_retrieval_precision",
        "average_negative_transfer_risk",
        "harmful_transfer_rate",
    )
    @classmethod
    def validate_transfer_rate(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("transfer summary rates must be within [0, 1]")
        return round(float(value), 6)


class MemoryTransferSummary(BaseModel):
    """Top-level conclusions for cross-task memory transfer."""

    model_config = ConfigDict(extra="forbid")

    governed_transfer_beneficial: bool
    governance_blocks_harmful_transfer: bool
    forced_transfer_exposes_negative_transfer: bool
    notes: list[str] = Field(default_factory=list)


class MemoryTransferSuiteResult(BaseModel):
    """Formal transfer result between a source and target vertical slice."""

    model_config = ConfigDict(extra="forbid")

    source_task_slug: str
    target_task_slug: str
    transfer_kind: Literal["same_family", "cross_family"]
    modes: list[Literal["no_memory", "governed_transfer", "forced_transfer"]] = Field(default_factory=list)
    transfer_records: list[MemoryTransferStatsRecord] = Field(default_factory=list)
    mode_summaries: list[MemoryTransferModeSummary] = Field(default_factory=list)
    summary: MemoryTransferSummary


class MemoryTransferEvidenceBundle(BaseModel):
    """Paper-facing bundle for cross-task memory transfer evidence."""

    model_config = ConfigDict(extra="forbid")

    source_task_slug: str
    target_task_slug: str
    transfer_kind: Literal["same_family", "cross_family"]
    modes: list[Literal["no_memory", "governed_transfer", "forced_transfer"]] = Field(default_factory=list)
    figures: list[FigureSpec] = Field(default_factory=list)
    tables: list[TableSpec] = Field(default_factory=list)
    summary: MemoryTransferSummary
    json_output_path: str
