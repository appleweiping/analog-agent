"""Trajectory logging helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def build_trajectory_event(step: str, payload: dict) -> dict:
    """Return a timestamped trajectory event."""
    return {"step": step, "payload": payload, "timestamp": datetime.now(UTC).isoformat()}
