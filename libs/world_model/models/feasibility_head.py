"""Feasibility head placeholder."""

from __future__ import annotations


class FeasibilityHead:
    """Estimate whether a candidate is worth simulating."""

    def predict(self, features: dict) -> dict:
        return {"model": "feasibility_head", "status": "stub", "features": features}
