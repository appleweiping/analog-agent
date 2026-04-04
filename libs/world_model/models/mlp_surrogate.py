"""MLP surrogate placeholder."""

from __future__ import annotations


class MLPSurrogate:
    """Predict scalar metrics from vectorized features."""

    def predict(self, features: dict) -> dict:
        return {"model": "mlp", "status": "stub", "features": features}
