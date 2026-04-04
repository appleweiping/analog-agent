"""Retrieval policy helpers."""

from __future__ import annotations


def top_k(items: list[dict], k: int) -> list[dict]:
    """Return the first k items as a baseline retrieval policy."""
    return items[:k]
