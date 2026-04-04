"""Reflection summarization helpers."""

from __future__ import annotations


def summarize_reflections(reflections: list[str]) -> str:
    """Join reflections into a compact summary string."""
    return " | ".join(reflections)
