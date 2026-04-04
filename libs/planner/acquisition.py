"""Acquisition scoring helpers."""

from __future__ import annotations


def upper_confidence_bound(mean: float, uncertainty: float, beta: float = 1.0) -> float:
    """Compute a simple UCB score."""
    return mean + beta * uncertainty
