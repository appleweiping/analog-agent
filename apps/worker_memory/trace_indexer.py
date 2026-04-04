"""Trace indexing placeholder."""

from __future__ import annotations


def index_trace(trace: dict) -> dict:
    """Return a minimal indexing summary."""
    return {"status": "stub", "keys": sorted(trace.keys())}
