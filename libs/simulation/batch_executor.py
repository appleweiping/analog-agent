"""Batch execution helpers for simulator jobs."""

from __future__ import annotations

from libs.schema.simulation import AnalysisStatement


def chunk_jobs(job_ids: list[str], batch_size: int) -> list[list[str]]:
    """Split job ids into fixed-size batches."""
    return [job_ids[index:index + batch_size] for index in range(0, len(job_ids), batch_size)]


def schedule_analysis_batches(analysis_ids: list[str], *, allow_parallel: bool, batch_size: int) -> list[list[str]]:
    """Schedule analyses into execution batches."""

    if not allow_parallel:
        return [[analysis_id] for analysis_id in analysis_ids]
    return chunk_jobs(analysis_ids, max(1, batch_size))


def order_analyses(analyses: list[AnalysisStatement]) -> list[AnalysisStatement]:
    """Return analyses in deterministic execution order."""

    return sorted(analyses, key=lambda item: (item.order, item.analysis_type))
