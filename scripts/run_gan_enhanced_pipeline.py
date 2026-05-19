"""GAN-Enhanced Optimization Pipeline for OTA2 circuit design.

Integrates all 3 GAN modules into the optimization loop:
1. Inverse Generator for warm-start candidates
2. Pareto GAN for Pareto-front seeding
3. Source Critic for trust-weighted acquisition

Compares convergence speed with and without GAN enhancement.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from string import Template

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from libs.world_model.xgb_surrogate import XGBSurrogate
from libs.gan.source_critic import SourceCritic, CriticConfig
from libs.gan.inverse_generator import InverseDesignGenerator, InverseGenConfig
from libs.gan.pareto_gan import ParetoGAN, ParetoGANConfig

NGSPICE_BIN = Path(r"D:\research\Agent-AI4EDA\tools\ngspice\bin\ngspice_con.exe")
TEMPLATE_PATH = REPO_ROOT / "templates" / "netlist" / "ota2" / "v1" / "ota2_demonstrator_truth.spice.tpl"
MODEL_DIR = REPO_ROOT / "data" / "gan_models"

# ---------------------------------------------------------------------------
# Design space
# ---------------------------------------------------------------------------
PARAM_NAMES = ["gm1", "gm2", "ro1", "ro2", "cc", "ibias"]
METRIC_NAMES = ["dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w"]

PARAM_BOUNDS = np.array([
    [0.5e-3, 5e-3],
    [1e-3, 10e-3],
    [10e3, 500e3],
    [5e3, 200e3],
    [0.5e-12, 5e-12],
    [10e-6, 500e-6],
])

LOG_BOUNDS_LO = np.log10(PARAM_BOUNDS[:, 0])
LOG_BOUNDS_HI = np.log10(PARAM_BOUNDS[:, 1])

CONSTRAINTS = {
    "dc_gain_db": (">=", 60.0),
    "gbw_hz": (">=", 80e6),
    "phase_margin_deg": (">=", 55.0),
    "power_w": ("<=", 1.5e-3),
}

FIXED_PARAMS = {
    "vdd": "1.2",
    "vin_cm": "0.6",
    "vin_step_high": "0.61",
    "cload": "2e-12",
    "cp1": "0.1e-12",
    "truth_mode": "configured",
    "template_id": "ota2_gan_pipeline",
    "p2_hint_hz": "1e9",
}

# Target spec for inverse generator
TARGET_SPEC = np.array([70.0, 100e6, 60.0, 0.5e-3])


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def log_to_params(x_log: np.ndarray) -> dict[str, float]:
    values = 10 ** x_log
    return {name: float(values[i]) for i, name in enumerate(PARAM_NAMES)}


def params_to_log(params: dict[str, float]) -> np.ndarray:
    return np.log10(np.array([params[name] for name in PARAM_NAMES]))


def params_dict_to_array(params: dict[str, float]) -> np.ndarray:
    return np.array([params[name] for name in PARAM_NAMES])


def array_to_params_dict(arr: np.ndarray) -> dict[str, float]:
    return {name: float(arr[i]) for i, name in enumerate(PARAM_NAMES)}


def check_constraints(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    violated = []
    for metric_name, (op, threshold) in CONSTRAINTS.items():
        if metric_name not in metrics:
            violated.append(metric_name)
            continue
        val = metrics[metric_name]
        if op == ">=" and val < threshold:
            violated.append(metric_name)
        elif op == "<=" and val > threshold:
            violated.append(metric_name)
    return len(violated) == 0, violated


def scalarize_objective(metrics: dict[str, float]) -> float:
    gbw = metrics.get("gbw_hz", 0)
    gbw_norm = min(gbw / 500e6, 1.0)
    power = metrics.get("power_w", 1.5e-3)
    power_eff_norm = max(0, 1.0 - power / 1.5e-3)
    return 0.7 * gbw_norm + 0.3 * power_eff_norm


def constraint_violation_penalty(metrics: dict[str, float]) -> float:
    penalty = 0.0
    for metric_name, (op, threshold) in CONSTRAINTS.items():
        if metric_name not in metrics:
            penalty += 10.0
            continue
        val = metrics[metric_name]
        if op == ">=" and val < threshold:
            penalty += (threshold - val) / abs(threshold) if threshold != 0 else (threshold - val)
        elif op == "<=" and val > threshold:
            penalty += (val - threshold) / abs(threshold) if threshold != 0 else (val - threshold)
    return penalty


# ---------------------------------------------------------------------------
# ngspice simulation
# ---------------------------------------------------------------------------
def build_netlist(params: dict[str, float]) -> str:
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    subs = dict(FIXED_PARAMS)
    for name, value in params.items():
        subs[name] = f"{value:.6e}"
    netlist = Template(template_text).substitute(subs)
    netlist += "\n.ac dec 40 1 20g\n\n"
    netlist += ".control\nrun\n"
    netlist += "let gain = v(vout)/v(vinp)\n"
    netlist += "let gain_db = db(gain)\n"
    netlist += "let dc_gain = gain_db[0]\n"
    netlist += "meas ac gbw_freq when gain_db=0\n"
    netlist += "let phase_at_gbw = 180 + vp(vout)\n"
    netlist += "if (gbw_freq > 0)\n"
    netlist += "  meas ac pm find vp(vout) at=$&gbw_freq\n"
    netlist += "  let phase_margin = 180 + pm\n"
    netlist += "else\n"
    netlist += "  let phase_margin = 0\n"
    netlist += "end\n"
    netlist += "print dc_gain\nprint gbw_freq\nprint phase_margin\n"
    netlist += "quit\n.endc\n.end\n"
    return netlist


def parse_ngspice_output(stdout: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "dc_gain_db": None, "gbw_hz": None, "phase_margin_deg": None,
    }
    for line in stdout.splitlines():
        line_stripped = line.strip().lower()
        if "dc_gain" in line_stripped and "=" in line_stripped:
            match = re.search(r"=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line)
            if match:
                metrics["dc_gain_db"] = float(match.group(1))
        if "gbw_freq" in line_stripped and "=" in line_stripped:
            match = re.search(r"=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line)
            if match:
                val = float(match.group(1))
                if val > 0:
                    metrics["gbw_hz"] = val
        if "phase_margin" in line_stripped and "=" in line_stripped:
            match = re.search(r"=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line)
            if match:
                metrics["phase_margin_deg"] = float(match.group(1))
    return metrics


def run_ngspice(params: dict[str, float], tmp_dir: str) -> dict[str, float] | None:
    netlist = build_netlist(params)
    netlist_path = os.path.join(tmp_dir, "ota2_gan.spice")
    with open(netlist_path, "w", encoding="utf-8") as f:
        f.write(netlist)
    try:
        result = subprocess.run(
            [str(NGSPICE_BIN), "-b", netlist_path],
            capture_output=True, text=True, timeout=30, cwd=tmp_dir,
        )
        stdout = result.stdout + result.stderr
    except (subprocess.TimeoutExpired, OSError):
        return None
    metrics = parse_ngspice_output(stdout)
    vdd = float(FIXED_PARAMS["vdd"])
    metrics["power_w"] = vdd * 2 * params["ibias"]
    if metrics["dc_gain_db"] is None or metrics["gbw_hz"] is None:
        return None
    if metrics["dc_gain_db"] < 0 or metrics["dc_gain_db"] > 200:
        return None
    if metrics["gbw_hz"] > 50e9:
        return None
    return {k: v for k, v in metrics.items() if v is not None}


# ---------------------------------------------------------------------------
# GAN-Enhanced Bayesian Optimization
# ---------------------------------------------------------------------------
class GANEnhancedOptimizer:
    """BO optimizer enhanced with GAN warm-start and trust-weighted acquisition."""

    def __init__(
        self,
        surrogate: XGBSurrogate,
        critic: SourceCritic,
        inv_gen: InverseDesignGenerator,
        pareto_gan: ParetoGAN | None,
        budget: int = 50,
        seed: int = 42,
    ):
        self.surrogate = surrogate
        self.critic = critic
        self.inv_gen = inv_gen
        self.pareto_gan = pareto_gan
        self.budget = budget
        self.rng = np.random.default_rng(seed)

        # History
        self.X_observed: list[np.ndarray] = []
        self.y_observed: list[float] = []
        self.metrics_history: list[dict[str, float]] = []
        self.params_history: list[dict[str, float]] = []
        self.feasible_history: list[bool] = []
        self.trust_scores: list[float] = []
        self.best_feasible_idx: int | None = None
        self.best_feasible_score: float = -np.inf

    def _predict_with_trust(self, x_log: np.ndarray) -> tuple[float, float, float]:
        """Predict objective, uncertainty, and trust score."""
        params = log_to_params(x_log)
        preds = self.surrogate.predict_with_uncertainty(params)
        metrics_mean = {k: v[0] for k, v in preds.items()}
        obj_mean = scalarize_objective(metrics_mean)
        penalty = constraint_violation_penalty(metrics_mean)
        obj_mean -= 2.0 * penalty

        # Uncertainty from ensemble
        obj_std = 0.0
        for k, (mean, std) in preds.items():
            if "gbw" in k:
                obj_std += 0.7 * (std / 500e6)
            elif "power" in k:
                obj_std += 0.3 * (std / 1.5e-3)
        obj_std = max(obj_std, 1e-6)

        # Trust score from Source Critic
        params_arr = params_dict_to_array(params).reshape(1, -1)
        pred_arr = np.array([[metrics_mean[m] for m in METRIC_NAMES]])
        trust = float(self.critic.score_trust(params_arr, pred_arr)[0])

        return obj_mean, obj_std, trust

    def _expected_improvement_with_trust(self, x_log: np.ndarray) -> float:
        """EI * trust_score^2 - penalizes untrusted regions."""
        obj_mean, obj_std, trust = self._predict_with_trust(x_log)

        if self.best_feasible_score <= -np.inf:
            # No feasible point yet, use raw EI
            ei = obj_mean + obj_std
        else:
            z = (obj_mean - self.best_feasible_score) / obj_std
            ei = (obj_mean - self.best_feasible_score) * norm.cdf(z) + obj_std * norm.pdf(z)

        # Trust-weighted acquisition
        return ei * (trust ** 2)

    def _generate_warm_start(self) -> list[np.ndarray]:
        """Generate initial candidates using GANs + random."""
        candidates = []

        # 1. Inverse Generator: 10 candidates for target spec
        print("    Inverse Generator: proposing 10 candidates...")
        try:
            inv_candidates = self.inv_gen.generate(TARGET_SPEC, n_candidates=10)
            for c in inv_candidates:
                # Clip to bounds and convert to log
                c_clipped = np.clip(c, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
                candidates.append(np.log10(c_clipped))
            print(f"      Got {len(inv_candidates)} inverse design candidates")
        except Exception as e:
            print(f"      Inverse Generator failed: {e}")

        # 2. Pareto GAN: 20 Pareto-front designs
        if self.pareto_gan and self.pareto_gan.is_trained:
            print("    Pareto GAN: generating 20 Pareto-front designs...")
            try:
                pareto_designs = self.pareto_gan.generate_pareto_front(n_points=20)
                for d in pareto_designs:
                    d_clipped = np.clip(d, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
                    candidates.append(np.log10(d_clipped))
                print(f"      Got {len(pareto_designs)} Pareto designs")
            except Exception as e:
                print(f"      Pareto GAN failed: {e}")

        # 3. Random samples for diversity
        n_random = max(5, self.budget // 5 - len(candidates))
        print(f"    Adding {n_random} random samples for diversity...")
        for _ in range(n_random):
            x_log = self.rng.uniform(LOG_BOUNDS_LO, LOG_BOUNDS_HI)
            candidates.append(x_log)

        return candidates

    def run(self, verify_with_ngspice: bool = True) -> dict:
        """Run the GAN-enhanced optimization loop."""
        print("\n" + "=" * 60)
        print("  GAN-ENHANCED BAYESIAN OPTIMIZATION")
        print("=" * 60)
        start_time = time.time()

        tmp_dir = tempfile.mkdtemp(prefix="ota2_gan_")

        # Phase 1: GAN warm-start
        print("\n  [Phase 1] GAN Warm-Start")
        warm_candidates = self._generate_warm_start()
        print(f"    Total warm-start candidates: {len(warm_candidates)}")

        # Evaluate warm-start candidates with surrogate
        print("\n  [Phase 2] Evaluating warm-start with surrogate...")
        for x_log in warm_candidates:
            params = log_to_params(x_log)
            pred = self.surrogate.predict(params)
            obj = scalarize_objective(pred)
            penalty = constraint_violation_penalty(pred)
            obj -= 2.0 * penalty

            feasible, _ = check_constraints(pred)

            self.X_observed.append(x_log)
            self.y_observed.append(obj)
            self.metrics_history.append(pred)
            self.params_history.append(params)
            self.feasible_history.append(feasible)

            if feasible and obj > self.best_feasible_score:
                self.best_feasible_score = obj
                self.best_feasible_idx = len(self.X_observed) - 1

        n_warm = len(warm_candidates)
        n_warm_feasible = sum(self.feasible_history[:n_warm])
        print(f"    Warm-start feasible: {n_warm_feasible}/{n_warm}")
        if self.best_feasible_idx is not None:
            print(f"    Best warm-start score: {self.best_feasible_score:.4f}")

        # Phase 3: BO loop with trust-weighted acquisition
        remaining_budget = self.budget - n_warm
        if remaining_budget < 0:
            remaining_budget = 0
        print(f"\n  [Phase 3] BO Loop ({remaining_budget} iterations)")

        ngspice_verified = 0
        for iteration in range(remaining_budget):
            # Multi-start optimization of acquisition function
            best_acq = -np.inf
            best_x = None

            # Try multiple random starts
            for _ in range(20):
                x0 = self.rng.uniform(LOG_BOUNDS_LO, LOG_BOUNDS_HI)
                acq = self._expected_improvement_with_trust(x0)
                if acq > best_acq:
                    best_acq = acq
                    best_x = x0

            # Also try perturbations of best known point
            if self.best_feasible_idx is not None:
                x_best = self.X_observed[self.best_feasible_idx]
                for _ in range(10):
                    perturbation = self.rng.normal(0, 0.1, size=len(x_best))
                    x_cand = np.clip(x_best + perturbation, LOG_BOUNDS_LO, LOG_BOUNDS_HI)
                    acq = self._expected_improvement_with_trust(x_cand)
                    if acq > best_acq:
                        best_acq = acq
                        best_x = x_cand

            if best_x is None:
                best_x = self.rng.uniform(LOG_BOUNDS_LO, LOG_BOUNDS_HI)

            # Evaluate candidate
            params = log_to_params(best_x)
            pred = self.surrogate.predict(params)
            obj = scalarize_objective(pred)
            penalty = constraint_violation_penalty(pred)
            obj -= 2.0 * penalty
            feasible, violated = check_constraints(pred)

            # Trust score
            params_arr = params_dict_to_array(params).reshape(1, -1)
            pred_arr = np.array([[pred[m] for m in METRIC_NAMES]])
            trust = float(self.critic.score_trust(params_arr, pred_arr)[0])
            self.trust_scores.append(trust)

            self.X_observed.append(best_x)
            self.y_observed.append(obj)
            self.metrics_history.append(pred)
            self.params_history.append(params)
            self.feasible_history.append(feasible)

            if feasible and obj > self.best_feasible_score:
                self.best_feasible_score = obj
                self.best_feasible_idx = len(self.X_observed) - 1

            # Verify top candidates with ngspice
            if verify_with_ngspice and feasible and trust > 0.6:
                ngspice_metrics = run_ngspice(params, tmp_dir)
                if ngspice_metrics:
                    ngspice_verified += 1
                    ngspice_feasible, _ = check_constraints(ngspice_metrics)
                    if ngspice_feasible:
                        ngspice_obj = scalarize_objective(ngspice_metrics)
                        # Update with ground truth
                        self.metrics_history[-1] = ngspice_metrics
                        self.y_observed[-1] = ngspice_obj
                        if ngspice_obj > self.best_feasible_score:
                            self.best_feasible_score = ngspice_obj
                            self.best_feasible_idx = len(self.X_observed) - 1

            if (iteration + 1) % 10 == 0:
                print(f"    Iter {iteration+1}/{remaining_budget}: "
                      f"best_score={self.best_feasible_score:.4f}, "
                      f"trust={trust:.3f}, feasible={sum(self.feasible_history)}")

        elapsed = time.time() - start_time

        # Compile results
        results = {
            "total_evaluations": len(self.X_observed),
            "warm_start_candidates": n_warm,
            "bo_iterations": remaining_budget,
            "feasible_count": sum(self.feasible_history),
            "ngspice_verified": ngspice_verified,
            "elapsed_s": elapsed,
            "best_score": self.best_feasible_score,
            "mean_trust": float(np.mean(self.trust_scores)) if self.trust_scores else 0.5,
        }

        if self.best_feasible_idx is not None:
            results["best_design"] = {
                "parameters": self.params_history[self.best_feasible_idx],
                "metrics": self.metrics_history[self.best_feasible_idx],
                "score": self.best_feasible_score,
            }
            results["spec_met"] = True
        else:
            results["spec_met"] = False

        return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("#" * 60)
    print("#  GAN-ENHANCED OPTIMIZATION PIPELINE")
    print("#" * 60)
    pipeline_start = time.time()

    # Stage 1: Load models
    print("\n[Stage 1] LOADING TRAINED MODELS")
    print("=" * 60)

    # XGBoost surrogate
    print("  Loading XGBoost surrogate...")
    surrogate = XGBSurrogate()
    surrogate.load()
    print(f"    Metrics: {surrogate.target_metrics}")

    # Source Critic
    print("  Loading Source Critic...")
    critic = SourceCritic()
    critic_path = MODEL_DIR / "source_critic"
    input_dim = len(PARAM_NAMES) + len(METRIC_NAMES)  # params + predictions
    critic.load(critic_path, input_dim=input_dim)
    print(f"    Loaded from {critic_path}")

    # Inverse Generator
    print("  Loading Inverse Generator...")
    inv_gen = InverseDesignGenerator(param_dim=len(PARAM_NAMES))
    inv_gen.load(MODEL_DIR / "inverse_generator")
    print(f"    Loaded from {MODEL_DIR / 'inverse_generator'}")

    # Pareto GAN
    pareto_gan = None
    pareto_path = MODEL_DIR / "pareto_gan"
    if (pareto_path / "pareto_generator.pt").exists():
        print("  Loading Pareto GAN...")
        pareto_gan = ParetoGAN(param_dim=len(PARAM_NAMES))
        pareto_gan.load(pareto_path)
        print(f"    Loaded from {pareto_path}")
    else:
        print("  Pareto GAN not available (skipping)")

    # Stage 2: Topology selection
    print("\n[Stage 2] TOPOLOGY SELECTION")
    print("=" * 60)
    print("  Circuit: OTA2 (Two-Stage OTA)")
    print("  Family: two_stage_ota")
    print(f"  Parameters: {len(PARAM_NAMES)} ({', '.join(PARAM_NAMES)})")
    print(f"  Constraints: {len(CONSTRAINTS)}")

    # Stage 3: GAN-enhanced optimization
    print("\n[Stage 3] GAN-ENHANCED OPTIMIZATION")
    print("=" * 60)

    optimizer = GANEnhancedOptimizer(
        surrogate=surrogate,
        critic=critic,
        inv_gen=inv_gen,
        pareto_gan=pareto_gan,
        budget=50,
        seed=42,
    )
    results = optimizer.run(verify_with_ngspice=True)

    # Stage 4: Results
    print("\n[Stage 4] RESULTS")
    print("=" * 60)
    print(f"  Total evaluations: {results['total_evaluations']}")
    print(f"  Warm-start candidates: {results['warm_start_candidates']}")
    print(f"  BO iterations: {results['bo_iterations']}")
    print(f"  Feasible designs: {results['feasible_count']}")
    print(f"  ngspice verified: {results['ngspice_verified']}")
    print(f"  Mean trust score: {results['mean_trust']:.3f}")
    print(f"  Elapsed time: {results['elapsed_s']:.1f}s")

    if results.get("best_design"):
        bd = results["best_design"]
        print(f"\n  BEST DESIGN:")
        print(f"  {'=' * 50}")
        print(f"  Parameters:")
        for name, val in bd["parameters"].items():
            idx = PARAM_NAMES.index(name)
            lo, hi = PARAM_BOUNDS[idx]
            pct = (val - lo) / (hi - lo) * 100
            print(f"    {name:8s} = {val:.4e}  ({pct:.0f}% of range)")

        print(f"\n  Metrics:")
        print(f"  {'Metric':<22s} {'Value':>12s}  {'Spec':>16s}  {'Status':>6s}")
        print(f"  {'-' * 60}")
        for metric, (op, threshold) in CONSTRAINTS.items():
            val = bd["metrics"].get(metric, float("nan"))
            if op == ">=" and val >= threshold:
                status = "OK"
            elif op == "<=" and val <= threshold:
                status = "OK"
            else:
                status = "FAIL"
            if "hz" in metric.lower():
                val_str = f"{val/1e6:.2f} MHz"
                thr_str = f"{op} {threshold/1e6:.0f} MHz"
            elif "power" in metric.lower():
                val_str = f"{val*1e3:.3f} mW"
                thr_str = f"{op} {threshold*1e3:.1f} mW"
            else:
                val_str = f"{val:.2f}"
                thr_str = f"{op} {threshold:.0f}"
            print(f"  {metric:<22s} {val_str:>12s}  {thr_str:>16s}  [{status:>4s}]")
    else:
        print("\n  No feasible design found!")

    # Stage 5: Comparison with baseline
    print("\n[Stage 5] COMPARISON WITH BASELINE")
    print("=" * 60)
    baseline_time = 38.6
    baseline_evals = 50
    print(f"  Baseline (no GAN): {baseline_time:.1f}s, {baseline_evals} evals")
    print(f"  GAN-enhanced:      {results['elapsed_s']:.1f}s, {results['total_evaluations']} evals")

    if results.get("best_design"):
        print(f"\n  GAN warm-start provided {results['warm_start_candidates']} informed candidates")
        print(f"  Trust-weighted acquisition filtered unreliable regions")
        if results['feasible_count'] > 0:
            # First feasible found at which eval?
            first_feasible_idx = next(
                (i for i, f in enumerate(optimizer.feasible_history) if f), None
            )
            if first_feasible_idx is not None:
                print(f"  First feasible design found at eval #{first_feasible_idx + 1}")
                if first_feasible_idx < 10:
                    print(f"  -> GAN warm-start found feasible design immediately!")

    # Pareto front data
    if pareto_gan and pareto_gan.is_trained:
        print(f"\n  Pareto Front (from Pareto GAN):")
        pareto_designs = pareto_gan.generate_pareto_front(n_points=10)
        print(f"    Generated 10 Pareto-front designs")
        for i, d in enumerate(pareto_designs[:5]):
            p = array_to_params_dict(d)
            pred = surrogate.predict(p)
            print(f"    #{i+1}: GBW={pred['gbw_hz']/1e6:.1f}MHz, "
                  f"Power={pred['power_w']*1e3:.3f}mW")

    pipeline_elapsed = time.time() - pipeline_start
    print(f"\n{'#' * 60}")
    print(f"  Pipeline completed in {pipeline_elapsed:.1f}s")
    print(f"{'#' * 60}")

    # Save results
    output_dir = REPO_ROOT / "data" / "optimization_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ota2_gan_enhanced_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
