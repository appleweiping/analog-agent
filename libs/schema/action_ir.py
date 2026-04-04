"""Intermediate representation for planner actions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ActionIR:
    action_type: str
    target: str
    params: dict[str, float | str] = field(default_factory=dict)
    rationale: str = ""
