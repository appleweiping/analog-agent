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
    trust_guides_selection_behavior: bool
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
