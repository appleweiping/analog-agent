"""Intermediate representation for circuit topology graphs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CircuitNode:
    node_id: str
    kind: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CircuitIR:
    name: str
    nodes: list[CircuitNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
