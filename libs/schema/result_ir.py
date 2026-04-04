"""Schema for aggregated experiment results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ResultIR:
    benchmark: str
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
