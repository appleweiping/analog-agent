"""Simulation metric extraction helpers."""

from __future__ import annotations


def extract_metrics(parsed_output: dict[str, object]) -> dict[str, float]:
    """Extract normalized metric values from parser-neutral output."""

    metrics = parsed_output.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return {
        str(metric): float(value)
        for metric, value in metrics.items()
        if isinstance(value, (int, float))
    }
