"""Calibration helpers for surrogate uncertainty."""

from __future__ import annotations


def calibrate(model_name: str, calibration_split: str) -> dict[str, str]:
    """Return a stub calibration record."""
    return {"model": model_name, "split": calibration_split, "status": "stub"}
