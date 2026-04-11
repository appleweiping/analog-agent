"""Formal experiment schemas for Day-4 evaluation and baseline comparisons."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from libs.schema.design_task import DesignTask
from libs.schema.stats import ExperimentStatsRecord, StatsAggregationResult, VerificationStatsRecord

EXPERIMENT_MODES = (
    "full_simulation_baseline",
    "no_world_model_baseline",
    "full_system",
)


class ExperimentBudget(BaseModel):
    """Structured experiment budget."""

    model_config = ConfigDict(extra="forbid")

    max_simulations: int
    max_candidates_per_step: int = 3


class ExperimentLogRecord(BaseModel):
    """Structured per-step or per-simulation log record."""

    model_config = ConfigDict(extra="forbid")

    step_index: int
    mode: Literal["full_simulation_baseline", "no_world_model_baseline", "full_system"]
    candidate_ids: list[str] = Field(default_factory=list)
    predicted_truth_gap: dict[str, float] = Field(default_factory=dict)
    simulation_selection_ratio: float = 0.0
    feasible_hit: bool = False
    failure_type_distribution: dict[str, int] = Field(default_factory=dict)

    @field_validator("simulation_selection_ratio")
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("simulation_selection_ratio must be within [0, 1]")
        return round(float(value), 6)


class ExperimentResult(BaseModel):
    """Formal result for one experiment run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    mode: Literal["full_simulation_baseline", "no_world_model_baseline", "full_system"]
    task_id: str
    simulation_call_count: int
    candidate_count: int
    best_feasible_found: bool
    best_metrics: dict[str, float] = Field(default_factory=dict)
    convergence_step: int | None = None
    predicted_truth_gap: dict[str, float] = Field(default_factory=dict)
    simulation_selection_ratio: float = 0.0
    feasible_hit_rate: float = 0.0
    failure_type_distribution: dict[str, int] = Field(default_factory=dict)
    efficiency_score: float = 0.0
    structured_log: list[ExperimentLogRecord] = Field(default_factory=list)
    verification_stats: list[VerificationStatsRecord] = Field(default_factory=list)
    stats_record: ExperimentStatsRecord | None = None

    @field_validator("simulation_selection_ratio", "feasible_hit_rate")
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("experiment rates must be within [0, 1]")
        return round(float(value), 6)


class ExperimentRunRequest(BaseModel):
    """Formal request for a single experiment run."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    mode: Literal["full_simulation_baseline", "no_world_model_baseline", "full_system"]
    budget: ExperimentBudget
    steps: int = 3
    run_index: int = 0
    fidelity_level: str = "focused_validation"
    backend_preference: str = "ngspice"


class ExperimentSuiteRequest(BaseModel):
    """Formal request for a repeated multi-mode experiment suite."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    modes: list[Literal["full_simulation_baseline", "no_world_model_baseline", "full_system"]] = Field(default_factory=list)
    budget: ExperimentBudget
    steps: int = 3
    repeat_runs: int = 5
    fidelity_level: str = "focused_validation"
    backend_preference: str = "ngspice"


class ExperimentAggregateSummary(BaseModel):
    """Aggregated comparison summary over repeated experiment runs."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["full_simulation_baseline", "no_world_model_baseline", "full_system"]
    run_count: int
    average_simulation_call_count: float
    feasible_hit_rate: float
    average_efficiency_score: float
    average_convergence_step: float
    average_selection_ratio: float
    average_best_metrics: dict[str, float] = Field(default_factory=dict)
    failure_type_distribution: dict[str, int] = Field(default_factory=dict)


class ExperimentSuiteResult(BaseModel):
    """Structured repeated experiment output."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    modes: list[Literal["full_simulation_baseline", "no_world_model_baseline", "full_system"]] = Field(default_factory=list)
    runs: list[ExperimentResult] = Field(default_factory=list)
    summaries: list[ExperimentAggregateSummary] = Field(default_factory=list)
    aggregated_stats: StatsAggregationResult | None = None
