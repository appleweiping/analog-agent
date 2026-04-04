"""XGBoost-style surrogate placeholder."""

from __future__ import annotations


class XGBSurrogate:
    """Predict scalar metrics from tabular features."""

    def predict(self, features: dict) -> dict:
        return {"model": "xgb", "status": "stub", "features": features}
