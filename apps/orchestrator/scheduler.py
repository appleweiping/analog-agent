"""Simple scheduling primitives for orchestration jobs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ScheduledJob:
    job_id: str
    priority: int = 0
    tags: list[str] = field(default_factory=list)


def order_jobs(jobs: list[ScheduledJob]) -> list[ScheduledJob]:
    """Return jobs ordered by descending priority."""
    return sorted(jobs, key=lambda job: job.priority, reverse=True)
