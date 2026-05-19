"""XGBoost-based surrogate model for OTA2 circuit performance prediction.

Provides fast inference using trained XGBoost models, with uncertainty
estimation via ensemble variance from cross-validation fold models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

# Default model directory
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "world_model"

FEATURE_NAMES = ["gm1", "gm2", "ro1", "ro2", "cc", "ibias"]


class XGBSurrogate:
    """XGBoost surrogate model for OTA2 metric prediction.

    Loads trained XGBoost models and provides predictions with uncertainty
    estimates derived from ensemble variance across cross-validation folds.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._models: dict[str, xgb.XGBRegressor] = {}
        self._fold_models: dict[str, list[xgb.XGBRegressor]] = {}
        self._summary: dict[str, Any] = {}
        self._loaded = False

    def load(self) -> None:
        """Load all trained models from disk."""
        summary_path = self.model_dir / "ota2_xgb_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Model summary not found at {summary_path}. "
                "Run train_xgb_world_model.py first."
            )

        with open(summary_path, "r", encoding="utf-8") as f:
            self._summary = json.load(f)

        for metric, meta in self._summary["models"].items():
            # Load main model
            model_path = self.model_dir / meta["model_file"]
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            self._models[metric] = model

            # Load fold models for uncertainty
            fold_models = []
            for fold_file in meta["fold_files"]:
                fold_path = self.model_dir / fold_file
                if fold_path.exists():
                    fold_model = xgb.XGBRegressor()
                    fold_model.load_model(str(fold_path))
                    fold_models.append(fold_model)
            self._fold_models[metric] = fold_models

        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _prepare_features(self, params: dict[str, float]) -> np.ndarray:
        """Convert parameter dict to log-transformed feature vector."""
        features = np.array([[params[name] for name in FEATURE_NAMES]])
        return np.log10(features)

    def predict(self, params: dict[str, float]) -> dict[str, float]:
        """Predict circuit metrics from design parameters.

        Args:
            params: Dict with keys gm1, gm2, ro1, ro2, cc, ibias.

        Returns:
            Dict mapping metric names to predicted values.
        """
        self._ensure_loaded()
        X = self._prepare_features(params)
        predictions = {}

        for metric, model in self._models.items():
            pred = float(model.predict(X)[0])
            meta = self._summary["models"][metric]

            # Inverse log transform for log-transformed targets
            if meta.get("log_transformed", False):
                pred = 10 ** pred

            predictions[metric] = pred

        return predictions

    def predict_with_uncertainty(
        self, params: dict[str, float]
    ) -> dict[str, tuple[float, float]]:
        """Predict metrics with uncertainty estimates.

        Uncertainty is estimated from the variance across fold model predictions.

        Args:
            params: Dict with keys gm1, gm2, ro1, ro2, cc, ibias.

        Returns:
            Dict mapping metric names to (mean_prediction, std_deviation) tuples.
        """
        self._ensure_loaded()
        X = self._prepare_features(params)
        results = {}

        for metric, fold_models in self._fold_models.items():
            if not fold_models:
                # Fallback to single model prediction with zero uncertainty
                pred = float(self._models[metric].predict(X)[0])
                meta = self._summary["models"][metric]
                if meta.get("log_transformed", False):
                    pred = 10 ** pred
                results[metric] = (pred, 0.0)
                continue

            # Get predictions from all fold models
            fold_preds = np.array([
                float(m.predict(X)[0]) for m in fold_models
            ])

            meta = self._summary["models"][metric]
            if meta.get("log_transformed", False):
                # Transform to linear space before computing stats
                fold_preds_linear = 10 ** fold_preds
                mean_pred = float(np.mean(fold_preds_linear))
                std_pred = float(np.std(fold_preds_linear))
            else:
                mean_pred = float(np.mean(fold_preds))
                std_pred = float(np.std(fold_preds))

            results[metric] = (mean_pred, std_pred)

        return results

    @property
    def feature_names(self) -> list[str]:
        """Return the expected feature names in order."""
        return list(FEATURE_NAMES)

    @property
    def target_metrics(self) -> list[str]:
        """Return the list of predicted metrics."""
        self._ensure_loaded()
        return list(self._models.keys())

    @property
    def model_summary(self) -> dict[str, Any]:
        """Return the training summary metadata."""
        self._ensure_loaded()
        return dict(self._summary)

    def get_cv_scores(self) -> dict[str, dict[str, float]]:
        """Return cross-validation scores for each metric."""
        self._ensure_loaded()
        scores = {}
        for metric, meta in self._summary["models"].items():
            scores[metric] = {
                "r2_mean": meta["cv_r2_mean"],
                "r2_std": meta["cv_r2_std"],
                "rmse_mean": meta["cv_rmse_mean"],
                "mape_mean": meta["cv_mape_mean"],
            }
        return scores
