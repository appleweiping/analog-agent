"""Validation helpers for incoming design specs."""

from __future__ import annotations

from libs.schema.design_spec import DesignSpec


def validate_design_spec(spec: DesignSpec) -> list[str]:
    """Return validation errors for an input spec."""
    errors: list[str] = []
    if not spec.name:
        errors.append("name must not be empty")
    if not spec.objectives:
        errors.append("at least one objective is required")
    return errors
