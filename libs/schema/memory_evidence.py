"""Formal schemas for repeated-episode memory evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from libs.schema.paper_evidence import FigureSpec, TableSpec

MEMORY_ABLATION_MODES = (
    "no_memory",
    "episodic_retrieval_only",
    "episodic_plus_reflection",
    "full_memory",
)
MEMORY_TRANSFER_MODES = (
    "no_memory",
    "governed_transfer",
    "no_governance",
    "forced_transfer",
)


class MemoryEpisodeStatsRecord(BaseModel):
    """Structured per-episode record for memory ablation."""

    model_config = ConfigDict(extra="forbid")

    episode_index: int
    mode: Literal["no_memory", "episodic_retrieval_only", "episodic_plus_reflection", "full_memory"]
    task_id: str
    family: str
    memory_episode_count_before: int = 0
    retrieved_episode_count: int = 0
    advice_count: int = 0
    advice_consumed_count: int = 0
    governance_block_count: int = 0
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
    repeated_measurement_failure_count: int = 0
    dominant_failure_repeated: bool = False
    measurement_failure_repeated: bool = False
    same_failure_mode_consecutive_count: int = 0
    prediction_gap: float = 0.0
    calibration_patch_count: int = 0
    advice_aligned_selection: bool = False
    retrieval_led_to_success: bool = False
    best_metric_values: list["MemoryMetricSnapshot"] = Field(default_factory=list)
    episode_memory_id: str | None = None

    @field_validator("retrieval_precision_proxy", "negative_transfer_risk")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("retrieval scores must be within [0, 1]")
        return round(float(value), 6)

    @field_validator("prediction_gap")
    @classmethod
    def validate_prediction_gap(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("prediction_gap must be non-negative")
        return round(float(value), 6)


class MemoryMetricSnapshot(BaseModel):
    """Structured metric snapshot carried by one memory evidence record."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    value: float


class DistributionSummary(BaseModel):
    """Distribution summary used by submission-facing memory statistics."""

    model_config = ConfigDict(extra="forbid")

    mean: float
    std: float
    median: float
    iqr: float
    minimum: float
    maximum: float


class MetricDistributionSummary(BaseModel):
    """Distribution summary for one measured metric."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    distribution: DistributionSummary


class MemoryModeSummary(BaseModel):
    """Aggregated summary for one repeated-episode memory mode."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["no_memory", "episodic_retrieval_only", "episodic_plus_reflection", "full_memory"]
    episode_count: int
    feasible_hit_rate: float
    average_real_simulation_calls: float
    average_step_to_first_feasible: float
    average_repeated_failure_count: float
    warm_start_application_rate: float
    average_advice_count: float
    average_advice_consumed_count: float
    advice_consumption_rate: float
    governance_block_rate: float
    average_retrieval_precision: float
    average_negative_transfer_risk: float
    feasible_hit_rate_by_episode: list[float] = Field(default_factory=list)
    median_step_to_first_feasible: float = 0.0
    mean_real_sim_calls_to_first_feasible: float = 0.0
    episode_best_metric_summary: list[MetricDistributionSummary] = Field(default_factory=list)
    repeated_failure_rate: float = 0.0
    dominant_failure_repeat_ratio: float = 0.0
    measurement_failure_repeat_ratio: float = 0.0
    same_failure_mode_consecutive_count: float = 0.0
    retrieval_activation_rate: float = 0.0
    advice_aligned_selection_rate: float = 0.0
    retrieval_to_success_conversion: float = 0.0
    episode_index_vs_sim_calls: list[float] = Field(default_factory=list)
    episode_index_vs_prediction_gap: list[float] = Field(default_factory=list)
    episode_index_vs_feasible_hit_rate: list[float] = Field(default_factory=list)
    episode_index_vs_failure_repeat_ratio: list[float] = Field(default_factory=list)
    simulation_calls_distribution: DistributionSummary
    step_to_feasible_distribution: DistributionSummary
    prediction_gap_distribution: DistributionSummary

    @field_validator(
        "feasible_hit_rate",
        "warm_start_application_rate",
        "advice_consumption_rate",
        "governance_block_rate",
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
    reflection_improves_over_retrieval_only: bool
    governance_preserves_memory_quality: bool
    calibration_and_memory_reduce_prediction_gap: bool = False
    retrieval_advice_improves_decision_alignment: bool = False
    simulation_call_paired_delta: float = 0.0
    repeated_failure_paired_delta: float = 0.0
    prediction_gap_paired_delta: float = 0.0
    simulation_call_effect_size: float = 0.0
    repeated_failure_effect_size: float = 0.0
    prediction_gap_effect_size: float = 0.0
    notes: list[str] = Field(default_factory=list)


class MemoryAblationSuiteResult(BaseModel):
    """Formal repeated-episode ablation result for one vertical slice."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    modes: list[Literal["no_memory", "episodic_retrieval_only", "episodic_plus_reflection", "full_memory"]] = Field(default_factory=list)
    episode_records: list[MemoryEpisodeStatsRecord] = Field(default_factory=list)
    mode_summaries: list[MemoryModeSummary] = Field(default_factory=list)
    summary: MemoryAblationSummary


class MemoryAblationEvidenceBundle(BaseModel):
    """Paper-facing bundle for repeated-episode memory evidence."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    modes: list[Literal["no_memory", "episodic_retrieval_only", "episodic_plus_reflection", "full_memory"]] = Field(default_factory=list)
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
    mode: Literal["no_memory", "governed_transfer", "no_governance", "forced_transfer"]
    source_episode_count: int = 0
    retrieved_episode_count: int = 0
    advice_count: int = 0
    advice_consumed_count: int = 0
    governance_block_count: int = 0
    retrieval_precision_proxy: float = 0.0
    negative_transfer_risk: float = 0.0
    warm_start_applied: bool = False
    warm_start_source: str | None = None
    best_feasible_found: bool = False
    real_simulation_calls: int = 0
    step_to_first_feasible: int | None = None
    repeated_failure_count: int = 0
    harmful_transfer_applied: bool = False
    wrong_family_retrieval: bool = False
    harmful_advice_application: bool = False
    cross_task_failure_after_retrieval: bool = False
    advice_aligned_selection: bool = False
    retrieval_led_to_success: bool = False
    best_metric_values: list[MemoryMetricSnapshot] = Field(default_factory=list)

    @field_validator("retrieval_precision_proxy", "negative_transfer_risk")
    @classmethod
    def validate_transfer_score(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("transfer scores must be within [0, 1]")
        return round(float(value), 6)


class MemoryTransferModeSummary(BaseModel):
    """Aggregated summary for one cross-task transfer mode."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["no_memory", "governed_transfer", "no_governance", "forced_transfer"]
    episode_count: int
    feasible_hit_rate: float
    average_real_simulation_calls: float
    average_step_to_first_feasible: float
    average_repeated_failure_count: float
    warm_start_application_rate: float
    average_advice_count: float
    average_advice_consumed_count: float
    advice_consumption_rate: float
    governance_block_rate: float
    average_retrieval_precision: float
    average_negative_transfer_risk: float
    harmful_transfer_rate: float
    wrong_family_retrieval_rate: float = 0.0
    harmful_advice_application_rate: float = 0.0
    cross_task_failure_after_retrieval_rate: float = 0.0
    governance_rejection_rate: float = 0.0
    advice_aligned_selection_rate: float = 0.0
    retrieval_to_success_conversion: float = 0.0
    simulation_calls_distribution: DistributionSummary
    step_to_feasible_distribution: DistributionSummary

    @field_validator(
        "feasible_hit_rate",
        "warm_start_application_rate",
        "advice_consumption_rate",
        "governance_block_rate",
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
    no_governance_exposes_harmful_transfer: bool
    forced_transfer_exposes_negative_transfer: bool
    governed_vs_no_memory_simulation_delta: float = 0.0
    governed_vs_no_memory_step_delta: float = 0.0
    governed_vs_no_governance_harmful_delta: float = 0.0
    harmful_transfer_effect_size: float = 0.0
    notes: list[str] = Field(default_factory=list)


class MemoryTransferSuiteResult(BaseModel):
    """Formal transfer result between a source and target vertical slice."""

    model_config = ConfigDict(extra="forbid")

    source_task_slug: str
    target_task_slug: str
    transfer_kind: Literal["same_family", "cross_family"]
    modes: list[Literal["no_memory", "governed_transfer", "no_governance", "forced_transfer"]] = Field(default_factory=list)
    transfer_records: list[MemoryTransferStatsRecord] = Field(default_factory=list)
    mode_summaries: list[MemoryTransferModeSummary] = Field(default_factory=list)
    summary: MemoryTransferSummary


class MemoryTransferEvidenceBundle(BaseModel):
    """Paper-facing bundle for cross-task memory transfer evidence."""

    model_config = ConfigDict(extra="forbid")

    source_task_slug: str
    target_task_slug: str
    transfer_kind: Literal["same_family", "cross_family"]
    modes: list[Literal["no_memory", "governed_transfer", "no_governance", "forced_transfer"]] = Field(default_factory=list)
    figures: list[FigureSpec] = Field(default_factory=list)
    tables: list[TableSpec] = Field(default_factory=list)
    summary: MemoryTransferSummary
    json_output_path: str


class MemoryChapterSummary(BaseModel):
    """Top-level conclusions for the memory chapter evidence package."""

    model_config = ConfigDict(extra="forbid")

    repeated_episode_beneficial: bool
    repeated_episode_generalizes_beyond_ota: bool
    same_family_transfer_beneficial: bool
    governance_blocks_cross_family_negative_transfer: bool
    no_governance_exposes_negative_transfer: bool
    forced_transfer_exposes_negative_transfer: bool
    calibration_and_memory_reduce_prediction_gap: bool = False
    retrieval_utility_observable: bool = False
    notes: list[str] = Field(default_factory=list)


class MemoryChapterEvidenceBundle(BaseModel):
    """Paper-facing chapter bundle that organizes repeated and transfer evidence."""

    model_config = ConfigDict(extra="forbid")

    chapter_id: str
    repeated_episode_tasks: list[str] = Field(default_factory=list)
    same_family_pairs: list[str] = Field(default_factory=list)
    cross_family_pairs: list[str] = Field(default_factory=list)
    figures: list[FigureSpec] = Field(default_factory=list)
    tables: list[TableSpec] = Field(default_factory=list)
    summary: MemoryChapterSummary
    json_output_path: str


class MemoryNegativeTransferCaseStudy(BaseModel):
    """Structured case study for a governed vs harmful cross-family transfer pair."""

    model_config = ConfigDict(extra="forbid")

    case_study_id: str
    source_task_slug: str
    target_task_slug: str
    selected_as_primary_case: bool = False
    governed_harmful_transfer_rate: float
    no_governance_harmful_transfer_rate: float
    forced_harmful_transfer_rate: float
    governed_avg_sim_calls: float
    no_governance_avg_sim_calls: float
    forced_avg_sim_calls: float
    governance_block_rate: float
    average_negative_transfer_risk: float
    narrative_summary: str
    markdown_output_path: str
    json_output_path: str

    @field_validator(
        "governed_harmful_transfer_rate",
        "no_governance_harmful_transfer_rate",
        "forced_harmful_transfer_rate",
        "governance_block_rate",
        "average_negative_transfer_risk",
    )
    @classmethod
    def validate_case_rate(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("case-study rates must be within [0, 1]")
        return round(float(value), 6)


class MemoryPaperLayoutBundle(BaseModel):
    """Submission-facing memory chapter package with main/appendix organization."""

    model_config = ConfigDict(extra="forbid")

    layout_id: str
    profile_name: str
    repeated_episode_tasks: list[str] = Field(default_factory=list)
    same_family_pairs: list[str] = Field(default_factory=list)
    cross_family_pairs: list[str] = Field(default_factory=list)
    main_figures: list[str] = Field(default_factory=list)
    appendix_figures: list[str] = Field(default_factory=list)
    main_tables: list[str] = Field(default_factory=list)
    appendix_tables: list[str] = Field(default_factory=list)
    case_studies: list[str] = Field(default_factory=list)
    summary_notes: list[str] = Field(default_factory=list)
    json_output_path: str
    markdown_output_path: str
