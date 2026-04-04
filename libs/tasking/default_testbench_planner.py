"""Default testbench planning helpers."""

from __future__ import annotations


def plan_testbenches(benchmark: str) -> list[str]:
    """Return baseline analysis types for a benchmark."""
    return ["dc", "ac", "transient"] if benchmark else []
