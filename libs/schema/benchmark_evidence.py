"""Formal schemas for benchmark-facing paper evidence and narrative bundles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from libs.schema.paper_evidence import TableSpec


class BenchmarkNarrativeSection(BaseModel):
    """One structured narrative section for benchmark-facing paper packaging."""

    model_config = ConfigDict(extra="forbid")

    section_id: str
    title: str
    body: str


class BenchmarkEvidenceBundle(BaseModel):
    """Formal bundle for benchmark rollups, family tables, and framing narratives."""

    model_config = ConfigDict(extra="forbid")

    bundle_id: str
    suite_id: str
    scope: Literal[
        "multitask_rollup",
        "family_summary",
        "failure_mode_synthesis",
        "robustness_narrative",
        "fidelity_corner_load_framing",
    ]
    tables: list[TableSpec] = Field(default_factory=list)
    narrative_sections: list[BenchmarkNarrativeSection] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    json_output_path: str
    markdown_output_path: str
