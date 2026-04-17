"""Formal schemas for paper-facing evidence bundles, figures, and tables."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FigureSeries(BaseModel):
    """One plotted series in a paper-facing figure."""

    model_config = ConfigDict(extra="forbid")

    label: str
    x_values: list[float] = Field(default_factory=list)
    y_values: list[float] = Field(default_factory=list)
    color: str = "#1f77b4"


class FigureSpec(BaseModel):
    """Structured metadata and data for one figure."""

    model_config = ConfigDict(extra="forbid")

    figure_id: str
    title: str
    chart_type: str
    x_label: str
    y_label: str
    series: list[FigureSeries] = Field(default_factory=list)
    caption: str
    output_path: str


class TableColumn(BaseModel):
    """One formal table column definition."""

    model_config = ConfigDict(extra="forbid")

    key: str
    label: str


class TableRow(BaseModel):
    """One structured table row."""

    model_config = ConfigDict(extra="forbid")

    values: dict[str, str | float | int | bool]


class TableSpec(BaseModel):
    """Structured metadata and data for one paper-facing table."""

    model_config = ConfigDict(extra="forbid")

    table_id: str
    title: str
    columns: list[TableColumn] = Field(default_factory=list)
    rows: list[TableRow] = Field(default_factory=list)
    caption: str
    csv_output_path: str
    markdown_output_path: str


class WorldModelUtilitySummary(BaseModel):
    """Top-level conclusions for world-model utility evidence."""

    model_config = ConfigDict(extra="forbid")

    world_model_reduces_simulations: bool
    world_model_preserves_or_improves_feasible_hit_rate: bool
    calibration_reduces_prediction_gap: bool
    prediction_gap_beats_no_world_model: bool
    trust_guides_selection_behavior: bool
    reliability_alignment_improves: bool
    ranking_improves_efficiency: bool
    ranking_preserves_or_improves_feasible_hit_rate: bool
    calibration_improves_convergence: bool
    calibration_updates_observable: bool
    notes: list[str] = Field(default_factory=list)


class WorldModelEvidenceBundle(BaseModel):
    """Formal bundle of figures, tables, and conclusions for world-model utility proof."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    modes: list[str] = Field(default_factory=list)
    figures: list[FigureSpec] = Field(default_factory=list)
    tables: list[TableSpec] = Field(default_factory=list)
    summary: WorldModelUtilitySummary
    json_output_path: str


class PlannerAblationSummary(BaseModel):
    """Top-level conclusions for planner-ablation evidence."""

    model_config = ConfigDict(extra="forbid")

    planner_beats_top_k: bool
    planner_reduces_simulations_vs_top_k: bool
    planner_preserves_or_improves_feasible_hit_rate_vs_top_k: bool
    planner_improves_efficiency_vs_top_k: bool
    fidelity_escalation_effective: bool
    fidelity_escalation_reduces_simulations: bool
    fidelity_escalation_preserves_or_improves_feasible_hit_rate: bool
    phase_updates_effective: bool
    phase_updates_improve_convergence: bool
    phase_updates_observable: bool
    calibration_replanning_effective: bool
    calibration_replanning_improves_convergence: bool
    calibration_replanning_observable: bool
    rollout_guidance_effective: bool
    rollout_guidance_improves_convergence: bool
    rollout_guidance_preserves_or_improves_feasible_hit_rate: bool
    rollout_guidance_observable: bool
    rollout_claim_supported_without_mpc_overclaim: bool
    rollout_claim_limited_to_short_horizon_guidance: bool
    rollout_evidence_real_not_placeholder: bool
    rollout_placeholder_risk: bool
    rollout_claim_scope: str
    rollout_claim_status: str
    dominant_failure_mode: str
    planner_reduces_failure_pressure: bool
    failure_synthesis_ready: bool
    efficiency_synthesis_ready: bool
    efficiency_frontier_consistent: bool
    notes: list[str] = Field(default_factory=list)


class PlannerAblationEvidenceBundle(BaseModel):
    """Formal bundle of figures, tables, and conclusions for planner ablations."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    modes: list[str] = Field(default_factory=list)
    figures: list[FigureSpec] = Field(default_factory=list)
    tables: list[TableSpec] = Field(default_factory=list)
    summary: PlannerAblationSummary
    json_output_path: str


class PlannerPaperLayoutBundle(BaseModel):
    """Submission-facing planner paper layout with main/appendix organization."""

    model_config = ConfigDict(extra="forbid")

    layout_id: str
    profile_name: str
    main_figures: list[str] = Field(default_factory=list)
    appendix_figures: list[str] = Field(default_factory=list)
    main_tables: list[str] = Field(default_factory=list)
    appendix_tables: list[str] = Field(default_factory=list)
    main_figure_captions: dict[str, str] = Field(default_factory=dict)
    appendix_figure_captions: dict[str, str] = Field(default_factory=dict)
    main_table_captions: dict[str, str] = Field(default_factory=dict)
    appendix_table_captions: dict[str, str] = Field(default_factory=dict)
    summary_notes: list[str] = Field(default_factory=list)
    json_output_path: str
    markdown_output_path: str
