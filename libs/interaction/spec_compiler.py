"""Compile raw user intent into a design specification."""

from __future__ import annotations

from libs.schema.design_spec import DesignSpec


def compile_spec(name: str, raw_objectives: dict[str, float]) -> DesignSpec:
    """Build a minimal DesignSpec from already-parsed objectives."""
    return DesignSpec(name=name, objectives=dict(raw_objectives))
