"""Workflow state definitions for orchestration runs."""

from __future__ import annotations

from enum import StrEnum


class RunState(StrEnum):
    RECEIVED = "received"
    PARSED = "parsed"
    PLANNED = "planned"
    SIMULATING = "simulating"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATES = {RunState.COMPLETED, RunState.FAILED}


def can_transition(current: RunState, candidate: RunState) -> bool:
    """Allow forward-only transitions except into terminal states twice."""
    if current in TERMINAL_STATES:
        return False
    order = list(RunState)
    return order.index(candidate) >= order.index(current)
