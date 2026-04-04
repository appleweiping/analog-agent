"""Task graph construction helpers."""

from __future__ import annotations


def build_default_graph() -> list[tuple[str, str]]:
    """Return the canonical task dependency graph."""
    return [
        ("parse", "plan"),
        ("plan", "simulate"),
        ("simulate", "reflect"),
    ]
