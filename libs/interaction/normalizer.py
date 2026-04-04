"""Normalization helpers for user and benchmark inputs."""

from __future__ import annotations


def normalize_keys(payload: dict[str, object]) -> dict[str, object]:
    """Normalize keys to lower snake-ish names."""
    return {key.strip().lower().replace(" ", "_"): value for key, value in payload.items()}
