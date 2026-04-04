"""Testbench construction helpers."""

from __future__ import annotations


def build_testbench(benchmark: str, analyses: list[str]) -> str:
    """Return a lightweight testbench summary."""
    joined = ",".join(analyses)
    return f"* testbench benchmark={benchmark} analyses={joined}"
