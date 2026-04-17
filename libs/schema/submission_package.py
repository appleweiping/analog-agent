"""Schemas for submission-facing package, boundary audits, and freeze manifests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from libs.schema.paper_evidence import TableSpec


class SubmissionAssetEntry(BaseModel):
    """One submission-facing figure or table selection entry."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    asset_kind: Literal["figure", "table"]
    title: str
    section: str
    source_path: str
    target_path: str = ""
    caption: str
    rationale: str
    availability_status: Literal["generated_ready", "manual_curation_required"]


class PhysicalValidityBoundaryBundle(BaseModel):
    """Structured bundle for paper-facing physical-validity boundary audits."""

    model_config = ConfigDict(extra="forbid")

    bundle_id: str
    tables: list[TableSpec] = Field(default_factory=list)
    summary_notes: list[str] = Field(default_factory=list)
    json_output_path: str
    markdown_output_path: str


class SubmissionAssetFreezeBundle(BaseModel):
    """Frozen main-text figure or table selection for the submission package."""

    model_config = ConfigDict(extra="forbid")

    bundle_id: str
    profile_name: str
    asset_kind: Literal["figure", "table"]
    entries: list[SubmissionAssetEntry] = Field(default_factory=list)
    ready_entry_count: int = 0
    pending_entry_count: int = 0
    summary_notes: list[str] = Field(default_factory=list)
    json_output_path: str
    markdown_output_path: str


class SubmissionAppendixAllocationBundle(BaseModel):
    """Appendix allocation for the submission-facing paper package."""

    model_config = ConfigDict(extra="forbid")

    bundle_id: str
    profile_name: str
    main_figure_ids: list[str] = Field(default_factory=list)
    main_table_ids: list[str] = Field(default_factory=list)
    appendix_figures: list[str] = Field(default_factory=list)
    appendix_tables: list[str] = Field(default_factory=list)
    summary_notes: list[str] = Field(default_factory=list)
    json_output_path: str
    markdown_output_path: str


class SubmissionSectionEntry(BaseModel):
    """One frozen manuscript/protocol/limitations section entry."""

    model_config = ConfigDict(extra="forbid")

    section_id: str
    title: str
    source_paths: list[str] = Field(default_factory=list)
    required_asset_ids: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    alignment_status: Literal["ready", "manual_attention_required"]
    notes: list[str] = Field(default_factory=list)


class SubmissionNarrativeFreezeBundle(BaseModel):
    """Frozen protocol, limitations, or manuscript narrative bundle."""

    model_config = ConfigDict(extra="forbid")

    bundle_id: str
    profile_name: str
    bundle_kind: Literal["protocol", "limitations", "manuscript"]
    source_documents: list[str] = Field(default_factory=list)
    frozen_documents: list[str] = Field(default_factory=list)
    sections: list[SubmissionSectionEntry] = Field(default_factory=list)
    tables: list[TableSpec] = Field(default_factory=list)
    ready_section_count: int = 0
    pending_section_count: int = 0
    summary_notes: list[str] = Field(default_factory=list)
    json_output_path: str
    markdown_output_path: str


class SubmissionExperimentAlignmentEntry(BaseModel):
    """One experiments-section alignment entry tied to frozen assets."""

    model_config = ConfigDict(extra="forbid")

    subsection_id: str
    title: str
    primary_claim: str
    main_figure_ids: list[str] = Field(default_factory=list)
    main_table_ids: list[str] = Field(default_factory=list)
    appendix_paths: list[str] = Field(default_factory=list)
    alignment_status: Literal["aligned", "manual_attention_required"]
    notes: list[str] = Field(default_factory=list)


class SubmissionExperimentAlignmentBundle(BaseModel):
    """Experiments-section alignment bundle for the submission package."""

    model_config = ConfigDict(extra="forbid")

    bundle_id: str
    profile_name: str
    entries: list[SubmissionExperimentAlignmentEntry] = Field(default_factory=list)
    tables: list[TableSpec] = Field(default_factory=list)
    aligned_entry_count: int = 0
    pending_entry_count: int = 0
    summary_notes: list[str] = Field(default_factory=list)
    json_output_path: str
    markdown_output_path: str


class InternalSubmissionPackageBundle(BaseModel):
    """Final internal submission package status bundle."""

    model_config = ConfigDict(extra="forbid")

    bundle_id: str
    profile_name: str
    included_bundle_paths: list[str] = Field(default_factory=list)
    primary_document_paths: list[str] = Field(default_factory=list)
    unresolved_manual_asset_ids: list[str] = Field(default_factory=list)
    internal_review_ready: bool
    external_submission_ready: bool
    summary_notes: list[str] = Field(default_factory=list)
    json_output_path: str
    markdown_output_path: str
