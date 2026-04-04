"""Uncertainty estimation service placeholder."""

from __future__ import annotations


class UncertaintyService:
    """Estimate confidence for surrogate predictions."""

    def estimate(self, prediction: dict) -> dict:
        return {"service": "uncertainty", "status": "stub", "prediction": prediction}
