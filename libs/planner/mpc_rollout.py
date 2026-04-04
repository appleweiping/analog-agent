"""Model predictive control rollout placeholder."""

from __future__ import annotations


def rollout(plan: list[dict], horizon: int) -> dict:
    """Return a minimal MPC summary."""
    return {"status": "stub", "horizon": horizon, "steps": len(plan)}
