"""Evaluation metric helpers."""

from __future__ import annotations


def pass_rate(results: list[bool]) -> float:
    """Compute the fraction of successful runs."""
    return sum(results) / len(results) if results else 0.0
