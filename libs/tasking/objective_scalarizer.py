"""Objective aggregation helpers."""

from __future__ import annotations


def scalarize(metrics: dict[str, float], weights: dict[str, float]) -> float:
    """Compute a simple weighted score."""
    return sum(metrics.get(name, 0.0) * weight for name, weight in weights.items())
