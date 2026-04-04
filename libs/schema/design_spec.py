"""Structured design specification schema."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DesignSpec:
    """Normalized target specification for an analog design task."""

    name: str
    topology_hint: str | None = None
    objectives: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, float] = field(default_factory=dict)
