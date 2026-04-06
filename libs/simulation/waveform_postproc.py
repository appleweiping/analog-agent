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


def estimate_slew_rate(time_axis: list[float], waveform: list[float]) -> float:
    """Estimate a simple slew rate from a sampled waveform."""

    if len(time_axis) < 2 or len(waveform) < 2:
        return 0.0
    deltas = []
    for index in range(1, min(len(time_axis), len(waveform))):
        dt = time_axis[index] - time_axis[index - 1]
        if dt == 0.0:
            continue
        deltas.append(abs((waveform[index] - waveform[index - 1]) / dt))
    return max(deltas, default=0.0)
