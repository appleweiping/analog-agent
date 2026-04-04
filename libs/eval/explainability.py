"""Explainability helpers."""

from __future__ import annotations


def explain_candidate(candidate: dict) -> str:
    """Render a simple candidate explanation."""
    return f"candidate keys: {', '.join(sorted(candidate.keys()))}"
