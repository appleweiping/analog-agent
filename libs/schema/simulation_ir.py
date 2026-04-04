"""Simulation request and response schema."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SimulationRequest:
    benchmark: str
    netlist: str
    corners: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SimulationResult:
    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    raw_artifact: str | None = None
