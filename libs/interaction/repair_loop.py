"""Repair loop helpers for invalid or incomplete specs."""

from __future__ import annotations


def suggest_repairs(errors: list[str]) -> list[str]:
    """Convert validation errors into simple repair hints."""
    return [f"repair:{error}" for error in errors]
