"""Constraint handling helpers."""

from __future__ import annotations


def resolve_feasibility(constraints: dict[str, float], observed: dict[str, float]) -> bool:
    """Return True when observed metrics satisfy all upper-bound constraints."""
    for key, value in constraints.items():
        if observed.get(key, float("inf")) > value:
            return False
    return True
