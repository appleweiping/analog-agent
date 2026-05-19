"""Train all 3 GAN modules for analog circuit design optimization.

Trains:
1. Source Critic — identifies trustworthy vs untrustworthy surrogate predictions
2. Inverse Design Generator — generates design params from target specs
3. Pareto GAN — generates designs along the Pareto frontier (GBW vs Power)

Uses data/ota2_training_data.json (500 samples) and XGBoost surrogate predictions.
Saves trained models to data/gan_models/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from libs.world_model.xgb_surrogate import XGBSurrogate
from libs.gan.source_critic import SourceCritic, CriticConfig
from libs.gan.inverse_generator import InverseDesignGenerator, InverseGenConfig
from libs.gan.pareto_gan import ParetoGAN, ParetoGANConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAINING_DATA_PATH = REPO_ROOT / "data" / "ota2_training_data.json"
MODEL_SAVE_DIR = REPO_ROOT / "data" / "gan_models"

PARAM_NAMES = ["gm1", "gm2", "ro1", "ro2", "cc", "ibias"]
METRIC_NAMES = ["dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w"]

PARAM_BOUNDS = np.array([
    [0.5e-3, 5e-3],    # gm1
    [1e-3, 10e-3],     # gm2
    [10e3, 500e3],     # ro1
    [5e3, 200e3],      # ro2
    [0.5e-12, 5e-12],  # cc
    [10e-6, 500e-6],   # ibias
])

CONSTRAINTS = {
    "dc_gain_db": (">=", 60.0),
    "gbw_hz": (">=", 80e6),
    "phase_margin_deg": (">=", 55.0),
    "power_w": ("<=", 1.5e-3),
}

# OTA2 target spec for testing inverse generator
TARGET_SPEC = np.array([70.0, 100e6, 60.0, 0.5e-3])  # gain, gbw, pm, power


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load training data and return params, metrics, feasible arrays."""
    print(f"  Loading training data from {TRAINING_DATA_PATH}")
    with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = len(data)
    params_array = np.zeros((n, len(PARAM_NAMES)))
    metrics_array = np.zeros((n, len(METRIC_NAMES)))
    feasible_array = np.zeros(n, dtype=bool)

    for i, sample in enumerate(data):
        for j, pname in enumerate(PARAM_NAMES):
            params_array[i, j] = sample["params"][pname]
        for j, mname in enumerate(METRIC_NAMES):
            metrics_array[i, j] = sample["metrics"][mname]
        feasible_array[i] = sample["feasible"]

    print(f"  Loaded {n} samples")
    print(f"  Feasible: {feasible_array.sum()} ({feasible_array.mean()*100:.1f}%)")
    return params_array, metrics_array, feasible_array


def get_xgb_predictions(surrogate: XGBSurrogate, params_array: np.ndarray) -> np.ndarray:
    """Get XGBoost surrogate predictions for all samples."""
    n = len(params_array)
    predictions = np.zeros((n, len(METRIC_NAMES)))

    for i in range(n):
        param_dict = {name: params_array[i, j] for j, name in enumerate(PARAM_NAMES)}
        pred = surrogate.predict(param_dict)
        for j, mname in enumerate(METRIC_NAMES):
            predictions[i, j] = pred[mname]

    return predictions


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------
def train_source_critic(
    params_array: np.ndarray,
    predictions: np.ndarray,
    truths: np.ndarray,
) -> tuple[SourceCritic, dict]:
    """Train Source Critic to identify trustworthy surrogate predictions."""
    print("\n" + "=" * 60)
    print("[1/3] TRAINING SOURCE CRITIC")
    print("=" * 60)

    config = CriticConfig(
        hidden_dims=[64, 128, 64],
        learning_rate=1e-3,
        epochs=100,
        batch_size=64,
        trust_threshold=0.15,
    )
    critic = SourceCritic(config=config)

    t0 = time.time()
    stats = critic.train(
        params_array=params_array,
        predictions=predictions,
        truths=truths,
        metric_names=METRIC_NAMES,
    )
    elapsed = time.time() - t0

    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Accuracy: {stats['accuracy']:.4f}")
    print(f"  Trustworthy ratio: {stats['trustworthy_ratio']:.3f}")
    print(f"  Final loss: {stats['final_loss']:.4f}")
    print(f"  Epochs: {stats['epochs']}")

    stats["training_time_s"] = elapsed
    return critic, stats


def train_inverse_generator(
    params_array: np.ndarray,
    metrics_array: np.ndarray,
) -> tuple[InverseDesignGenerator, dict]:
    """Train Inverse Design Generator (metrics -> params)."""
    print("\n" + "=" * 60)
    print("[2/3] TRAINING INVERSE DESIGN GENERATOR")
    print("=" * 60)

    config = InverseGenConfig(
        noise_dim=32,
        hidden_dims_g=[128, 256, 128],
        hidden_dims_d=[128, 256, 128],
        learning_rate_g=1e-4,
        learning_rate_d=1e-4,
        epochs=200,
        batch_size=64,
        n_critic=3,
        gp_lambda=10.0,
        condition_dim=4,
    )
    generator = InverseDesignGenerator(param_dim=len(PARAM_NAMES), config=config)

    t0 = time.time()
    stats = generator.train(
        params_array=params_array,
        metrics_array=metrics_array,
        param_bounds=PARAM_BOUNDS,
    )
    elapsed = time.time() - t0

    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Epochs: {stats['epochs']}")
    print(f"  Final D loss: {stats['final_d_loss']:.4f}")
    print(f"  Final G loss: {stats['final_g_loss']:.4f}")
    print(f"  Samples used: {stats['samples_used']}")

    # Test: generate 10 candidates for target spec
    print(f"\n  Testing: Generate 10 candidates for target spec")
    print(f"    Target: gain={TARGET_SPEC[0]:.0f}dB, GBW={TARGET_SPEC[1]/1e6:.0f}MHz, "
          f"PM={TARGET_SPEC[2]:.0f}deg, Power={TARGET_SPEC[3]*1e3:.2f}mW")

    candidates = generator.generate(TARGET_SPEC, n_candidates=10)
    print(f"    Generated {len(candidates)} candidates")

    # Check if candidates are within bounds
    in_bounds = np.all(
        (candidates >= PARAM_BOUNDS[:, 0]) & (candidates <= PARAM_BOUNDS[:, 1]),
        axis=1,
    )
    print(f"    In bounds: {in_bounds.sum()}/10")

    # Show range of generated params
    print(f"    Parameter ranges of generated candidates:")
    for j, pname in enumerate(PARAM_NAMES):
        lo, hi = candidates[:, j].min(), candidates[:, j].max()
        bound_lo, bound_hi = PARAM_BOUNDS[j]
        print(f"      {pname:6s}: [{lo:.3e}, {hi:.3e}]  (bounds: [{bound_lo:.3e}, {bound_hi:.3e}])")

    stats["training_time_s"] = elapsed
    stats["test_in_bounds"] = int(in_bounds.sum())
    return generator, stats


def train_pareto_gan(
    params_array: np.ndarray,
    metrics_array: np.ndarray,
    feasible_mask: np.ndarray,
) -> tuple[ParetoGAN, dict]:
    """Train Pareto GAN on feasible samples for GBW vs Power tradeoff."""
    print("\n" + "=" * 60)
    print("[3/3] TRAINING PARETO GAN")
    print("=" * 60)

    # Objectives: GBW (maximize) and Power (minimize)
    # Extract GBW (index 1) and Power (index 3)
    objectives = metrics_array[:, [1, 3]]  # GBW, Power
    maximize = [True, False]

    print(f"  Total samples: {len(params_array)}")
    print(f"  Feasible samples: {feasible_mask.sum()}")
    print(f"  Objectives: GBW (maximize), Power (minimize)")

    config = ParetoGANConfig(
        noise_dim=16,
        hidden_dims=[128, 256, 128],
        learning_rate=2e-4,
        epochs=200,
        batch_size=32,
        n_critic=3,
        gp_lambda=10.0,
    )
    pareto_gan = ParetoGAN(param_dim=len(PARAM_NAMES), config=config)

    t0 = time.time()
    stats = pareto_gan.train(
        params_array=params_array,
        objectives=objectives,
        maximize=maximize,
        param_bounds=PARAM_BOUNDS,
        feasible_mask=feasible_mask,
    )
    elapsed = time.time() - t0

    if "error" in stats:
        print(f"  ERROR: {stats['error']}")
        stats["training_time_s"] = elapsed
        return pareto_gan, stats

    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Epochs: {stats['epochs']}")
    print(f"  Pareto points found: {stats['n_pareto_points']}")
    print(f"  Total feasible used: {stats['n_total_feasible']}")
    print(f"  Final D loss: {stats['final_d_loss']:.4f}")
    print(f"  Final G loss: {stats['final_g_loss']:.4f}")

    # Generate a 20-point Pareto front
    if pareto_gan.is_trained:
        print(f"\n  Generating 20-point Pareto front...")
        pareto_designs = pareto_gan.generate_pareto_front(n_points=20, n_samples_per_point=5)
        print(f"    Generated {len(pareto_designs)} designs")

        # Check bounds
        in_bounds = np.all(
            (pareto_designs >= PARAM_BOUNDS[:, 0]) & (pareto_designs <= PARAM_BOUNDS[:, 1]),
            axis=1,
        )
        print(f"    In bounds: {in_bounds.sum()}/20")

        # Show parameter diversity
        for j, pname in enumerate(PARAM_NAMES):
            lo, hi = pareto_designs[:, j].min(), pareto_designs[:, j].max()
            print(f"      {pname:6s}: [{lo:.3e}, {hi:.3e}]")

        stats["pareto_front_in_bounds"] = int(in_bounds.sum())

    stats["training_time_s"] = elapsed
    return pareto_gan, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("#" * 60)
    print("#  GAN MODULE TRAINING FOR ANALOG-AGENT")
    print("#" * 60)
    total_start = time.time()

    # Load data
    print("\n[DATA LOADING]")
    params_array, metrics_array, feasible_array = load_training_data()

    # Load XGBoost surrogate and get predictions
    print("\n[LOADING XGBOOST SURROGATE]")
    surrogate = XGBSurrogate()
    surrogate.load()
    print(f"  Metrics: {surrogate.target_metrics}")

    print("  Computing XGBoost predictions for all 500 samples...")
    t0 = time.time()
    predictions = get_xgb_predictions(surrogate, params_array)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Compute prediction errors for reporting
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_errors = np.abs(predictions - metrics_array) / (np.abs(metrics_array) + 1e-10)
    mean_rel_error = np.mean(rel_errors, axis=0)
    print(f"  Mean relative errors per metric:")
    for j, mname in enumerate(METRIC_NAMES):
        print(f"    {mname:20s}: {mean_rel_error[j]*100:.1f}%")

    # Train all 3 GANs
    all_stats = {}

    # 1. Source Critic
    critic, critic_stats = train_source_critic(params_array, predictions, metrics_array)
    all_stats["source_critic"] = critic_stats

    # 2. Inverse Design Generator
    inv_gen, inv_gen_stats = train_inverse_generator(params_array, metrics_array)
    all_stats["inverse_generator"] = inv_gen_stats

    # 3. Pareto GAN
    pareto_gan, pareto_stats = train_pareto_gan(params_array, metrics_array, feasible_array)
    all_stats["pareto_gan"] = pareto_stats

    # Save models
    print("\n" + "=" * 60)
    print("[SAVING MODELS]")
    print("=" * 60)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    critic.save(MODEL_SAVE_DIR / "source_critic")
    print(f"  Source Critic saved to {MODEL_SAVE_DIR / 'source_critic'}")

    inv_gen.save(MODEL_SAVE_DIR / "inverse_generator")
    print(f"  Inverse Generator saved to {MODEL_SAVE_DIR / 'inverse_generator'}")

    if pareto_gan.is_trained:
        pareto_gan.save(MODEL_SAVE_DIR / "pareto_gan")
        print(f"  Pareto GAN saved to {MODEL_SAVE_DIR / 'pareto_gan'}")
    else:
        print(f"  Pareto GAN NOT saved (training failed)")

    # Save training summary
    summary_path = MODEL_SAVE_DIR / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"  Training summary saved to {summary_path}")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  TOTAL TRAINING TIME: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")

    # Final summary
    print(f"\n  TRAINING RESULTS SUMMARY:")
    print(f"  {'─' * 50}")
    print(f"  Source Critic:      accuracy={critic_stats['accuracy']:.4f}, "
          f"time={critic_stats['training_time_s']:.1f}s")
    print(f"  Inverse Generator:  D_loss={inv_gen_stats['final_d_loss']:.4f}, "
          f"G_loss={inv_gen_stats['final_g_loss']:.4f}, "
          f"time={inv_gen_stats['training_time_s']:.1f}s")
    if "error" not in pareto_stats:
        print(f"  Pareto GAN:         pareto_pts={pareto_stats['n_pareto_points']}, "
              f"D_loss={pareto_stats['final_d_loss']:.4f}, "
              f"time={pareto_stats['training_time_s']:.1f}s")
    else:
        print(f"  Pareto GAN:         ERROR - {pareto_stats['error']}")
    print(f"  {'─' * 50}")

    return all_stats


if __name__ == "__main__":
    main()
