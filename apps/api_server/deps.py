"""Dependency helpers for the API server."""

from __future__ import annotations

import os


def get_runtime_summary() -> dict[str, str]:
    """Return a small runtime snapshot for operational checks."""
    return {
        "environment": os.getenv("ANALOG_AGENT_ENV", "development"),
        "config": os.getenv("ANALOG_AGENT_CONFIG", "configs/default.yaml"),
        "simulator": os.getenv("SIMULATOR_BACKEND", "ngspice"),
        "world_model": os.getenv("WORLD_MODEL_BACKEND", "tabular_surrogate"),
    }
