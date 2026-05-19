"""Train XGBoost world models for OTA2 circuit metrics.

Loads data from data/ota2_training_data.json, trains one XGBoost regressor per
metric (dc_gain_db, gbw_hz, phase_margin_deg, power_w), evaluates with 5-fold
cross-validation, and saves models to data/world_model/ota2_xgb_*.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "ota2_training_data.json"
MODEL_DIR = REPO_ROOT / "data" / "world_model"

# ---------------------------------------------------------------------------
# Feature and target definitions
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["gm1", "gm2", "ro1", "ro2", "cc", "ibias"]
TARGET_METRICS = ["dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w"]


def load_data() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load training data and return feature matrix and target arrays."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {DATA_PATH}")

    # Filter to records that have all required metrics
    valid_records = [
        r for r in records
        if all(m in r["metrics"] for m in TARGET_METRICS)
    ]
    print(f"Records with all metrics: {len(valid_records)}")

    # Build feature matrix (log-transform for better XGBoost performance)
    X = np.array([
        [r["params"][name] for name in FEATURE_NAMES]
        for r in valid_records
    ])

    # Log-transform features (all are positive and span orders of magnitude)
    X_log = np.log10(X)

    # Build target arrays
    targets = {}
    for metric in TARGET_METRICS:
        values = np.array([r["metrics"][metric] for r in valid_records])
        # Log-transform gbw_hz and power_w for better regression
        if metric in ("gbw_hz", "power_w"):
            values = np.log10(np.clip(values, 1e-15, None))
        targets[metric] = values

    return X_log, targets


def train_and_evaluate(
    X: np.ndarray,
    targets: dict[str, np.ndarray],
    n_folds: int = 5,
) -> dict[str, dict]:
    """Train XGBoost models with cross-validation and return results."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    for metric in TARGET_METRICS:
        y = targets[metric]
        print(f"\n{'='*60}")
        print(f"Training model for: {metric}")
        print(f"  Target range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"  Target mean: {y.mean():.4f}, std: {y.std():.4f}")

        # XGBoost parameters
        xgb_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

        # Cross-validation
        fold_r2 = []
        fold_rmse = []
        fold_mape = []
        fold_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBRegressor(**xgb_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)

            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            # For MAPE, handle log-transformed targets
            if metric in ("gbw_hz", "power_w"):
                # Convert back to linear for MAPE
                y_val_lin = 10 ** y_val
                y_pred_lin = 10 ** y_pred
                mape = mean_absolute_percentage_error(y_val_lin, y_pred_lin)
            else:
                mape = mean_absolute_percentage_error(
                    np.clip(np.abs(y_val), 1e-10, None), y_pred
                )

            fold_r2.append(r2)
            fold_rmse.append(rmse)
            fold_mape.append(mape)
            fold_models.append(model)

        # Report
        mean_r2 = np.mean(fold_r2)
        mean_rmse = np.mean(fold_rmse)
        mean_mape = np.mean(fold_mape)
        std_r2 = np.std(fold_r2)

        print(f"  5-Fold CV Results:")
        print(f"    R2:   {mean_r2:.4f} +/- {std_r2:.4f}")
        print(f"    RMSE: {mean_rmse:.6f}")
        print(f"    MAPE: {mean_mape*100:.2f}%")

        # Train final model on all data
        final_model = xgb.XGBRegressor(**xgb_params)
        final_model.fit(X, y, verbose=False)

        # Feature importance
        importances = final_model.feature_importances_
        print(f"  Feature importance:")
        for fname, imp in sorted(
            zip(FEATURE_NAMES, importances), key=lambda x: -x[1]
        ):
            print(f"    {fname}: {imp:.3f}")

        results[metric] = {
            "model": final_model,
            "fold_models": fold_models,
            "cv_r2_mean": float(mean_r2),
            "cv_r2_std": float(std_r2),
            "cv_rmse_mean": float(mean_rmse),
            "cv_mape_mean": float(mean_mape),
            "feature_importance": {
                name: float(imp) for name, imp in zip(FEATURE_NAMES, importances)
            },
            "xgb_params": xgb_params,
            "log_transformed": metric in ("gbw_hz", "power_w"),
        }

    return results


def save_models(results: dict[str, dict]) -> None:
    """Save trained models and metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "feature_names": FEATURE_NAMES,
        "target_metrics": TARGET_METRICS,
        "feature_transform": "log10",
        "models": {},
    }

    for metric, data in results.items():
        # Save the final model
        model_path = MODEL_DIR / f"ota2_xgb_{metric}.json"
        data["model"].save_model(str(model_path))

        # Save fold models for uncertainty estimation
        for fold_idx, fold_model in enumerate(data["fold_models"]):
            fold_path = MODEL_DIR / f"ota2_xgb_{metric}_fold{fold_idx}.json"
            fold_model.save_model(str(fold_path))

        summary["models"][metric] = {
            "model_file": f"ota2_xgb_{metric}.json",
            "fold_files": [f"ota2_xgb_{metric}_fold{i}.json" for i in range(5)],
            "cv_r2_mean": data["cv_r2_mean"],
            "cv_r2_std": data["cv_r2_std"],
            "cv_rmse_mean": data["cv_rmse_mean"],
            "cv_mape_mean": data["cv_mape_mean"],
            "feature_importance": data["feature_importance"],
            "log_transformed": data["log_transformed"],
            "xgb_params": data["xgb_params"],
        }

    # Save summary
    summary_path = MODEL_DIR / "ota2_xgb_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nModel summary saved to: {summary_path}")


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Training data not found at {DATA_PATH}")
        print("Run collect_training_data.py first.")
        sys.exit(1)

    X, targets = load_data()

    if len(X) < 20:
        print(f"ERROR: Only {len(X)} valid samples. Need at least 20.")
        sys.exit(1)

    results = train_and_evaluate(X, targets)
    save_models(results)

    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'R2':<12} {'RMSE':<12} {'MAPE':<10}")
    print(f"{'-'*54}")
    for metric in TARGET_METRICS:
        r = results[metric]
        print(
            f"{metric:<20} "
            f"{r['cv_r2_mean']:.4f}+/-{r['cv_r2_std']:.3f} "
            f"{r['cv_rmse_mean']:<12.6f} "
            f"{r['cv_mape_mean']*100:<10.2f}%"
        )
    print(f"\nModels saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
