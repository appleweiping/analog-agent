"""Formal experiment schemas for baseline and methodology comparisons."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from libs.schema.design_task import DesignTask
from libs.schema.stats import ExperimentStatsRecord, StatsAggregationResult, VerificationStatsRecord

ExperimentMode = Literal[
    "full_simulation_baseline",
    "random_search_baseline",
    "bayesopt_baseline",
    "cmaes_baseline",
    "rl_baseline",
    "no_world_model_baseline",
    "full_system",
    "no_world_model",
    "no_calibration",
    "no_fidelity_escalation",
]

EXPERIMENT_MODES = (
    "full_simulation_baseline",
    "random_search_baseline",
    "bayesopt_baseline",
    "cmaes_baseline",
    "rl_baseline",
    "no_world_model_baseline",
    "full_system",
    "no_world_model",
    "no_calibration",
    "no_fidelity_escalation",
)


class MethodComponentConfig(BaseModel):
    """Formal execution-component switches for controlled experiments."""

    model_config = ConfigDict(extra="forbid")

    mode: ExperimentMode
    use_world_model: bool
    use_calibration: bool
    use_fidelity_escalation: bool
    use_full_simulation_baseline: bool = False
    use_random_search_baseline: bool = False
    use_bayesopt_baseline: bool = False
    use_cmaes_baseline: bool = False
    use_rl_baseline: bool = False


class ExperimentBudget(BaseModel):
    """Structured experiment budget."""

    model_config = ConfigDict(extra="forbid")

    max_simulations: int
    max_candidates_per_step: int = 3


class ExperimentLogRecord(BaseModel):
    """Structured per-step or per-simulation log record."""

    model_config = ConfigDict(extra="forbid")

    step_index: int
    mode: ExperimentMode
    candidate_ids: list[str] = Field(default_factory=list)
    predicted_truth_gap: dict[str, float] = Field(default_factory=dict)
    simulation_selection_ratio: float = 0.0
    feasible_hit: bool = False
    failure_type_distribution: dict[str, int] = Field(default_factory=dict)
    fidelity_usage: dict[str, int] = Field(default_factory=dict)
    calibration_updates_applied: int = 0
    world_model_enabled: bool = True
    calibration_enabled: bool = True
    fidelity_escalation_enabled: bool = True
    selected_mean_uncertainty: float = 0.0
    selected_mean_confidence: float = 0.0
    selected_mean_simulation_value: float = 0.0
    selected_mean_predicted_feasibility: float = 0.0

    @field_validator(
        "simulation_selection_ratio",
        "selected_mean_uncertainty",
        "selected_mean_confidence",
        "selected_mean_simulation_value",
        "selected_mean_predicted_feasibility",
    )
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("experiment log ratios must be within [0, 1]")
        return round(float(value), 6)


class ExperimentResult(BaseModel):
    """Formal result for one experiment run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    mode: ExperimentMode
    task_id: str
    component_config: MethodComponentConfig
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
    prediction_gap_by_step: list[dict[str, float]] = Field(default_factory=list)
    calibration_update_count: int = 0
    focused_truth_call_count: int = 0
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
    mode: ExperimentMode
    budget: ExperimentBudget
    steps: int = 3
    run_index: int = 0
    fidelity_level: str = "focused_validation"
    backend_preference: str = "ngspice"
    force_full_steps: bool = False


class ExperimentSuiteRequest(BaseModel):
    """Formal request for a repeated multi-mode experiment suite."""

    model_config = ConfigDict(extra="forbid")

    design_task: DesignTask
    modes: list[ExperimentMode] = Field(default_factory=list)
    budget: ExperimentBudget
    steps: int = 3
    repeat_runs: int = 5
    fidelity_level: str = "focused_validation"
    backend_preference: str = "ngspice"
    force_full_steps: bool = False


class ExperimentAggregateSummary(BaseModel):
    """Aggregated comparison summary over repeated experiment runs."""

    model_config = ConfigDict(extra="forbid")

    mode: ExperimentMode
    run_count: int
    average_simulation_call_count: float
    feasible_hit_rate: float
    average_efficiency_score: float
    average_convergence_step: float
    average_selection_ratio: float
    average_best_metrics: dict[str, float] = Field(default_factory=dict)
    failure_type_distribution: dict[str, int] = Field(default_factory=dict)
    average_prediction_gap: dict[str, float] = Field(default_factory=dict)
    average_calibration_update_count: float = 0.0
    average_focused_truth_call_count: float = 0.0


class MethodModeSummary(BaseModel):
    """Aggregated methodology-facing summary for one execution mode."""

    model_config = ConfigDict(extra="forbid")

    mode: ExperimentMode
    component_config: MethodComponentConfig
    run_count: int
    simulation_call_count: float
    feasible_hit_rate: float
    average_prediction_gap: dict[str, float] = Field(default_factory=dict)
    average_best_metrics: dict[str, float] = Field(default_factory=dict)
    average_convergence_step: float
    average_calibration_update_count: float = 0.0
    focused_truth_ratio: float = 0.0
    escalation_count: int = 0
    failure_type_distribution: dict[str, int] = Field(default_factory=dict)


class MethodDeltaSummary(BaseModel):
    """Pairwise delta between two methodology modes."""

    model_config = ConfigDict(extra="forbid")

    baseline_mode: ExperimentMode
    compared_mode: ExperimentMode
    simulation_call_delta: float
    feasible_hit_rate_delta: float
    prediction_gap_delta: dict[str, float] = Field(default_factory=dict)
    best_metric_delta: dict[str, float] = Field(default_factory=dict)
    focused_truth_ratio_delta: float = 0.0
    calibration_update_delta: float = 0.0


class MethodConclusionSummary(BaseModel):
    """Auto-generated methodology conclusions for Day-11 comparisons."""

    model_config = ConfigDict(extra="forbid")

    world_model_effective: bool
    calibration_effective: bool
    fidelity_effective: bool
    conclusion_notes: list[str] = Field(default_factory=list)


class MethodComparisonResult(BaseModel):
    """Structured methodology-comparison result over several modes."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    modes: list[ExperimentMode] = Field(default_factory=list)
    mode_summaries: list[MethodModeSummary] = Field(default_factory=list)
    deltas: list[MethodDeltaSummary] = Field(default_factory=list)
    conclusions: MethodConclusionSummary


class ExperimentSuiteResult(BaseModel):
    """Structured repeated experiment output."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    modes: list[ExperimentMode] = Field(default_factory=list)
    runs: list[ExperimentResult] = Field(default_factory=list)
    summaries: list[ExperimentAggregateSummary] = Field(default_factory=list)
    aggregated_stats: StatsAggregationResult | None = None
    comparison: MethodComparisonResult | None = None
