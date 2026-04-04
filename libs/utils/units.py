"""Basic unit conversion helpers."""

from __future__ import annotations


def mhz_to_hz(value: float) -> float:
    """Convert MHz to Hz."""
    return value * 1_000_000.0
