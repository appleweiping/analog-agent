"""Batch execution helpers for simulator jobs."""

from __future__ import annotations


def chunk_jobs(job_ids: list[str], batch_size: int) -> list[list[str]]:
    """Split job ids into fixed-size batches."""
    return [job_ids[index:index + batch_size] for index in range(0, len(job_ids), batch_size)]
