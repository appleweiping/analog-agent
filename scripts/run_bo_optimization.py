"""Bayesian Optimization for OTA2 circuit design using XGBoost surrogate.

Implements Expected Improvement (EI) acquisition function with L-BFGS-B
optimization, multi-objective scalarization, and constraint penalty handling.
Uses the trained XGBoost world model for cheap evaluations and ngspice for
ground-truth verification of top candidates.
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
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from libs.world_model.xgb_surrogate import XGBSurrogate
from libs.memory.episode_store import EpisodeStore

NGSPICE_BIN = Path(r"D:\research\Agent-AI4EDA\tools\ngspice\bin\ngspice_con.exe")
TEMPLATE_PATH = REPO_ROOT / "templates" / "netlist" / "ota2" / "v1" / "ota2_demonstrator_truth.spice.tpl"

# ---------------------------------------------------------------------------
# Design space
# ---------------------------------------------------------------------------
DESIGN_VARS = {
    "gm1": (0.5e-3, 5e-3),
    "gm2": (1e-3, 10e-3),
    "ro1": (10e3, 500e3),
    "ro2": (5e3, 200e3),
    "cc": (0.5e-12, 5e-12),
    "ibias": (10e-6, 500e-6),
}

VAR_NAMES = list(DESIGN_VARS.keys())
BOUNDS_LO = np.array([DESIGN_VARS[v][0] for v in VAR_NAMES])
BOUNDS_HI = np.array([DESIGN_VARS[v][1] for v in VAR_NAMES])

# Work in log10 space for better optimization landscape
LOG_BOUNDS_LO = np.log10(BOUNDS_LO)
LOG_BOUNDS_HI = np.log10(BOUNDS_HI)
LOG_BOUNDS = list(zip(LOG_BOUNDS_LO, LOG_BOUNDS_HI))

# ---------------------------------------------------------------------------
# Spec constraints
# ---------------------------------------------------------------------------
CONSTRAINTS = {
    "dc_gain_db": (">=", 60.0),
    "gbw_hz": (">=", 80e6),
    "phase_margin_deg": (">=", 55.0),
    "power_w": ("<=", 1.5e-3),
}

# Fixed netlist parameters
FIXED_PARAMS = {
    "vdd": "1.2",
    "vin_cm": "0.6",
    "vin_step_high": "0.61",
    "cload": "2e-12",
    "cp1": "0.1e-12",
    "truth_mode": "configured",
    "template_id": "ota2_bo_optimization",
    "p2_hint_hz": "1e9",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def log_to_params(x_log: np.ndarray) -> dict[str, float]:
    """Convert log10 parameter vector to parameter dict."""
    values = 10 ** x_log
    return {name: float(values[i]) for i, name in enumerate(VAR_NAMES)}


def params_to_log(params: dict[str, float]) -> np.ndarray:
    """Convert parameter dict to log10 vector."""
    return np.log10(np.array([params[name] for name in VAR_NAMES]))


def check_constraints(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    """Check if metrics satisfy all constraints.

    Returns:
        (feasible, list_of_violated_constraint_names)
    """
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


def constraint_violation_penalty(metrics: dict[str, float]) -> float:
    """Compute a penalty score for constraint violations (0 = feasible)."""
    penalty = 0.0
    for metric_name, (op, threshold) in CONSTRAINTS.items():
        if metric_name not in metrics:
            penalty += 10.0
            continue
        val = metrics[metric_name]
        if op == ">=" and val < threshold:
            # Normalized violation
            penalty += (threshold - val) / abs(threshold) if threshold != 0 else (threshold - val)
        elif op == "<=" and val > threshold:
            penalty += (val - threshold) / abs(threshold) if threshold != 0 else (val - threshold)
    return penalty


def scalarize_objective(metrics: dict[str, float]) -> float:
    """Scalarize multi-objective into single value (higher is better).

    Weighted sum: 0.7 * normalized_gbw + 0.3 * normalized_power_efficiency
    """
    # Normalize GBW: target range [0, 500MHz]
    gbw = metrics.get("gbw_hz", 0)
    gbw_norm = min(gbw / 500e6, 1.0)

    # Normalize power efficiency (lower power is better): target range [0, 1.5mW]
    power = metrics.get("power_w", 1.5e-3)
    power_eff_norm = max(0, 1.0 - power / 1.5e-3)

    return 0.7 * gbw_norm + 0.3 * power_eff_norm


# ---------------------------------------------------------------------------
# ngspice simulation
# ---------------------------------------------------------------------------
def build_netlist(params: dict[str, float]) -> str:
    """Build ngspice netlist from parameters."""
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

    subs = dict(FIXED_PARAMS)
    for name, value in params.items():
        subs[name] = f"{value:.6e}"

    netlist = Template(template_text).substitute(subs)

    # Append AC analysis
    netlist += "\n"
    netlist += ".ac dec 40 1 20g\n"
    netlist += "\n"
    netlist += ".control\n"
    netlist += "run\n"
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
    netlist += "print dc_gain\n"
    netlist += "print gbw_freq\n"
    netlist += "print phase_margin\n"
    netlist += "quit\n"
    netlist += ".endc\n"
    netlist += ".end\n"

    return netlist


def parse_ngspice_output(stdout: str) -> dict[str, float | None]:
    """Parse ngspice stdout to extract metric values."""
    metrics: dict[str, float | None] = {
        "dc_gain_db": None,
        "gbw_hz": None,
        "phase_margin_deg": None,
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
    """Run ngspice simulation and return metrics, or None on failure."""
    netlist = build_netlist(params)
    netlist_path = os.path.join(tmp_dir, "ota2_bo.spice")

    with open(netlist_path, "w", encoding="utf-8") as f:
        f.write(netlist)

    try:
        result = subprocess.run(
            [str(NGSPICE_BIN), "-b", netlist_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=tmp_dir,
        )
        stdout = result.stdout + result.stderr
    except (subprocess.TimeoutExpired, OSError):
        return None

    metrics = parse_ngspice_output(stdout)

    # Compute power analytically
    vdd = float(FIXED_PARAMS["vdd"])
    power_w = vdd * 2 * params["ibias"]
    metrics["power_w"] = power_w

    # Validate
    if metrics["dc_gain_db"] is None or metrics["gbw_hz"] is None:
        return None
    if metrics["dc_gain_db"] < 0 or metrics["dc_gain_db"] > 200:
        return None
    if metrics["gbw_hz"] > 50e9:
        return None

    return {k: v for k, v in metrics.items() if v is not None}


# ---------------------------------------------------------------------------
# Bayesian Optimization
# ---------------------------------------------------------------------------
class BayesianOptimizer:
    """Bayesian Optimization using XGBoost surrogate with Expected Improvement."""

    def __init__(
        self,
        surrogate: XGBSurrogate,
        episode_store: EpisodeStore,
        budget: int = 50,
        n_initial: int = 10,
        seed: int = 42,
    ):
        self.surrogate = surrogate
        self.episode_store = episode_store
        self.budget = budget
        self.n_initial = n_initial
        self.rng = np.random.default_rng(seed)

        # History
        self.X_observed: list[np.ndarray] = []  # log10 space
        self.y_observed: list[float] = []  # scalarized objective
        self.metrics_history: list[dict[str, float]] = []
        self.params_history: list[dict[str, float]] = []
        self.feasible_history: list[bool] = []
        self.best_feasible_idx: int | None = None
        self.best_feasible_score: float = -np.inf

    def _predict_scalarized(self, x_log: np.ndarray) -> tuple[float, float]:
        """Predict scalarized objective with uncertainty at a point.

        Returns:
            (mean, std) of the scalarized objective.
        """
        params = log_to_params(x_log)
        preds = self.surrogate.predict_with_uncertainty(params)

        # Compute scalarized mean
        metrics_mean = {k: v[0] for k, v in preds.items()}
        obj_mean = scalarize_objective(metrics_mean)

        # Penalty for constraint violations
        penalty = constraint_violation_penalty(metrics_mean)
        obj_mean -= penalty * 2.0  # Strong penalty

        # Estimate uncertainty from GBW (dominant objective)
        gbw_mean, gbw_std = preds.get("gbw_hz", (0, 0))
        # Propagate uncertainty through scalarization
        if gbw_mean > 0:
            obj_std = 0.7 * (gbw_std / 500e6)
        else:
            obj_std = 0.01

        # Ensure positive std
        obj_std = max(obj_std, 1e-6)

        return obj_mean, obj_std

    def expected_improvement(self, x_log: np.ndarray) -> float:
        """Compute Expected Improvement at a point.

        EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
        where z = (f_best - mu(x)) / sigma(x)

        We maximize EI, so we return negative for scipy.minimize.
        """
        mu, sigma = self._predict_scalarized(x_log)

        if sigma < 1e-10:
            return 0.0

        # Use best observed feasible score, or best overall if none feasible
        f_best = self.best_feasible_score if self.best_feasible_score > -np.inf else 0.0

        z = (mu - f_best) / sigma
        ei = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)

        return ei

    def _neg_ei(self, x_log: np.ndarray) -> float:
        """Negative EI for minimization."""
        return -self.expected_improvement(x_log)

    def _optimize_acquisition(self, n_restarts: int = 10) -> np.ndarray:
        """Find the point that maximizes EI using multi-start L-BFGS-B."""
        best_x = None
        best_ei = -np.inf

        for _ in range(n_restarts):
            # Random starting point in log space
            x0 = self.rng.uniform(LOG_BOUNDS_LO, LOG_BOUNDS_HI)

            try:
                result = scipy_minimize(
                    self._neg_ei,
                    x0,
                    method="L-BFGS-B",
                    bounds=LOG_BOUNDS,
                    options={"maxiter": 50, "ftol": 1e-8},
                )
                if -result.fun > best_ei:
                    best_ei = -result.fun
                    best_x = result.x
            except Exception:
                continue

        if best_x is None:
            # Fallback: random point
            best_x = self.rng.uniform(LOG_BOUNDS_LO, LOG_BOUNDS_HI)

        return best_x

    def _generate_initial_points(self) -> list[np.ndarray]:
        """Generate initial Latin Hypercube samples in log space."""
        from scipy.stats.qmc import LatinHypercube

        sampler = LatinHypercube(d=len(VAR_NAMES), seed=int(self.rng.integers(0, 10000)))
        unit_samples = sampler.random(n=self.n_initial)

        points = []
        for i in range(self.n_initial):
            x_log = LOG_BOUNDS_LO + unit_samples[i] * (LOG_BOUNDS_HI - LOG_BOUNDS_LO)
            points.append(x_log)
        return points

    def run(self, verify_with_ngspice: bool = True) -> dict:
        """Run the full BO loop.

        Args:
            verify_with_ngspice: If True, verify top candidates with ngspice.

        Returns:
            Dict with optimization results.
        """
        print("=" * 70)
        print("BAYESIAN OPTIMIZATION - OTA2 Circuit Design")
        print("=" * 70)
        print(f"Budget: {self.budget} iterations")
        print(f"Initial samples: {self.n_initial}")
        print(f"Design variables: {VAR_NAMES}")
        print(f"Verify with ngspice: {verify_with_ngspice}")
        print("-" * 70)

        start_time = time.time()
        tmp_dir = tempfile.mkdtemp(prefix="ota2_bo_")

        try:
            # Phase 1: Initial exploration
            print("\n[Phase 1] Initial exploration...")
            initial_points = self._generate_initial_points()

            for i, x_log in enumerate(initial_points):
                params = log_to_params(x_log)
                metrics = self.surrogate.predict(params)

                # Optionally verify with ngspice
                if verify_with_ngspice:
                    ngspice_metrics = run_ngspice(params, tmp_dir)
                    if ngspice_metrics is not None:
                        metrics = ngspice_metrics

                score = scalarize_objective(metrics)
                feasible, violated = check_constraints(metrics)

                # Penalize infeasible
                if not feasible:
                    score -= constraint_violation_penalty(metrics) * 2.0

                self.X_observed.append(x_log)
                self.y_observed.append(score)
                self.metrics_history.append(metrics)
                self.params_history.append(params)
                self.feasible_history.append(feasible)

                if feasible and score > self.best_feasible_score:
                    self.best_feasible_score = score
                    self.best_feasible_idx = len(self.X_observed) - 1

                # Record in episode store
                self.episode_store.record_episode(
                    params=params,
                    metrics=metrics,
                    feasible=feasible,
                    circuit_family="two_stage_ota",
                    constraints_violated=violated,
                    notes=f"BO initial sample {i+1}/{self.n_initial}",
                )

                status = "FEASIBLE" if feasible else f"violated: {violated}"
                print(f"  [{i+1}/{self.n_initial}] score={score:.4f} | {status}")

            # Phase 2: BO iterations
            remaining = self.budget - self.n_initial
            print(f"\n[Phase 2] BO iterations ({remaining} remaining)...")

            for iteration in range(remaining):
                iter_num = self.n_initial + iteration + 1

                # Optimize acquisition function
                x_next = self._optimize_acquisition(n_restarts=8)
                params = log_to_params(x_next)

                # Predict with surrogate
                metrics = self.surrogate.predict(params)

                # Verify with ngspice
                if verify_with_ngspice:
                    ngspice_metrics = run_ngspice(params, tmp_dir)
                    if ngspice_metrics is not None:
                        metrics = ngspice_metrics

                score = scalarize_objective(metrics)
                feasible, violated = check_constraints(metrics)

                if not feasible:
                    score -= constraint_violation_penalty(metrics) * 2.0

                self.X_observed.append(x_next)
                self.y_observed.append(score)
                self.metrics_history.append(metrics)
                self.params_history.append(params)
                self.feasible_history.append(feasible)

                if feasible and score > self.best_feasible_score:
                    self.best_feasible_score = score
                    self.best_feasible_idx = len(self.X_observed) - 1

                # Record in episode store
                self.episode_store.record_episode(
                    params=params,
                    metrics=metrics,
                    feasible=feasible,
                    circuit_family="two_stage_ota",
                    constraints_violated=violated,
                    notes=f"BO iteration {iter_num}/{self.budget}",
                )

                # Progress report
                if (iteration + 1) % 5 == 0 or iteration == 0:
                    best_str = f"best_score={self.best_feasible_score:.4f}" if self.best_feasible_idx is not None else "no feasible yet"
                    n_feasible = sum(self.feasible_history)
                    print(
                        f"  [{iter_num}/{self.budget}] "
                        f"score={score:.4f} | feasible={n_feasible}/{iter_num} | {best_str}"
                    )

        finally:
            # Cleanup temp dir
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        elapsed = time.time() - start_time

        # Compile results
        results = self._compile_results(elapsed)
        self._print_summary(results)
        return results

    def _compile_results(self, elapsed: float) -> dict:
        """Compile optimization results into a summary dict."""
        results = {
            "optimizer": "bayesian_optimization",
            "budget": self.budget,
            "elapsed_seconds": elapsed,
            "total_evaluations": len(self.X_observed),
            "feasible_count": sum(self.feasible_history),
            "feasibility_rate": sum(self.feasible_history) / len(self.feasible_history) if self.feasible_history else 0,
            "convergence_history": self.y_observed,
        }

        if self.best_feasible_idx is not None:
            best_params = self.params_history[self.best_feasible_idx]
            best_metrics = self.metrics_history[self.best_feasible_idx]
            results["best_design"] = {
                "parameters": best_params,
                "metrics": best_metrics,
                "score": self.best_feasible_score,
                "iteration": self.best_feasible_idx + 1,
            }
            results["spec_met"] = True
        else:
            # Return best infeasible as reference
            if self.y_observed:
                best_idx = int(np.argmax(self.y_observed))
                results["best_design"] = {
                    "parameters": self.params_history[best_idx],
                    "metrics": self.metrics_history[best_idx],
                    "score": self.y_observed[best_idx],
                    "iteration": best_idx + 1,
                    "note": "INFEASIBLE - no design met all constraints",
                }
            results["spec_met"] = False

        return results

    def _print_summary(self, results: dict) -> None:
        """Print a summary of optimization results."""
        print("\n" + "=" * 70)
        print("OPTIMIZATION RESULTS")
        print("=" * 70)
        print(f"Total evaluations: {results['total_evaluations']}")
        print(f"Feasible designs found: {results['feasible_count']}")
        print(f"Feasibility rate: {results['feasibility_rate']:.1%}")
        print(f"Elapsed time: {results['elapsed_seconds']:.1f}s")

        if "best_design" in results:
            bd = results["best_design"]
            print(f"\nBest Design (iteration {bd['iteration']}):")
            print(f"  Score: {bd['score']:.4f}")
            print(f"\n  Parameters:")
            for name, val in bd["parameters"].items():
                print(f"    {name:8s} = {val:.4e}")
            print(f"\n  Metrics vs Spec:")
            for metric, (op, threshold) in CONSTRAINTS.items():
                val = bd["metrics"].get(metric, float("nan"))
                met = ""
                if op == ">=" and val >= threshold:
                    met = "OK"
                elif op == "<=" and val <= threshold:
                    met = "OK"
                else:
                    met = "FAIL"
                print(f"    {metric:20s} = {val:12.4e}  (spec: {op} {threshold:.4e}) [{met}]")

        print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Bayesian Optimization for OTA2")
    parser.add_argument("--budget", type=int, default=50, help="Total evaluation budget")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-ngspice", action="store_true", help="Skip ngspice verification")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    # Load surrogate
    print("Loading XGBoost surrogate model...")
    surrogate = XGBSurrogate()
    surrogate.load()
    print(f"  Metrics: {surrogate.target_metrics}")
    print(f"  CV R² scores: {surrogate.get_cv_scores()}")

    # Initialize episode store
    store_path = REPO_ROOT / "data" / "optimization_results" / "bo_episodes.json"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    episode_store = EpisodeStore(store_path)

    # Run optimization
    optimizer = BayesianOptimizer(
        surrogate=surrogate,
        episode_store=episode_store,
        budget=args.budget,
        seed=args.seed,
    )

    results = optimizer.run(verify_with_ngspice=not args.no_ngspice)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPO_ROOT / "data" / "optimization_results" / f"ota2_bo_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
