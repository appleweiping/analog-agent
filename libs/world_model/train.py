"""Training entrypoints for surrogate models."""

from __future__ import annotations


def train(model_name: str, dataset_path: str) -> dict[str, str]:
    """Return a stub training record."""
    return {"model": model_name, "dataset": dataset_path, "status": "stub"}
