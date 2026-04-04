"""Robustness helpers."""

from __future__ import annotations


def worst_case(metrics: list[float]) -> float:
    """Return the worst observed metric."""
    return min(metrics) if metrics else 0.0
