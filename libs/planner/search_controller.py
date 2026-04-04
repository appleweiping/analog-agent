"""Search loop coordination helpers."""

from __future__ import annotations


def next_iteration(iteration: int) -> int:
    """Advance the planner iteration counter."""
    return iteration + 1
