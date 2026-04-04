"""Job execution record helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def build_run_record(job_id: str, state: str) -> dict[str, str]:
    """Build a small, serializable job status record."""
    return {
        "job_id": job_id,
        "state": state,
        "timestamp": datetime.now(UTC).isoformat(),
    }
