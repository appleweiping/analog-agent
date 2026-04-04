"""Inference entrypoints for surrogate models."""

from __future__ import annotations


def infer(model_name: str, features: dict) -> dict:
    """Return a stub inference record."""
    return {"model": model_name, "features": features, "status": "stub"}
