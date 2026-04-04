"""Waveform post-processing helpers."""

from __future__ import annotations


def summarize_waveform(samples: list[float]) -> dict[str, float]:
    """Return a tiny summary for a numeric trace."""
    if not samples:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": min(samples),
        "max": max(samples),
        "mean": sum(samples) / len(samples),
    }
