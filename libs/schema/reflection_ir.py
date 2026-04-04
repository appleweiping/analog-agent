"""Schema for reflection summaries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ReflectionIR:
    summary: str
    failures: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
