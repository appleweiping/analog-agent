"""CMA-ES optimizer for OTA2 circuit design using XGBoost surrogate.

Implements Covariance Matrix Adaptation Evolution Strategy with death penalty
for constraint handling. Uses the world model for cheap evaluations and
ngspice for verification of top candidates.
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
# Design space (same as BO)
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

# Work in log10 space
LOG_BOUNDS_LO = np.log10(BOUNDS_LO)
LOG_BOUNDS_HI = np.log10(BOUNDS_HI)

# ---------------------------------------------------------------------------
# Spec constraints
# ---------------------------------------------------------------------------
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
    "template_id": "ota2_cmaes_optimization",
    "p2_hint_hz": "1e9",
}

DEATH_PENALTY = 1e6


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def log_to_params(x_log: np.ndarray) -> dict[str, float]:
    """Convert log10 parameter vector to parameter dict."""
    values = 10 ** x_log
    return {name: float(values[i]) for i, name in enumerate(VAR_NAMES)}


def check_constraints(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    """Check if metrics satisfy all constraints."""
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
    """Scalarize multi-objective (higher is better)."""
    gbw = metrics.get("gbw_hz", 0)
    gbw_norm = min(gbw / 500e6, 1.0)

    power = metrics.get("power_w", 1.5e-3)
    power_eff_norm = max(0, 1.0 - power / 1.5e-3)

    return 0.7 * gbw_norm + 0.3 * power_eff_norm


# ---------------------------------------------------------------------------
# ngspice simulation (same as BO)
# ---------------------------------------------------------------------------
def build_netlist(params: dict[str, float]) -> str:
    """Build ngspice netlist from parameters."""
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

    subs = dict(FIXED_PARAMS)
    for name, value in params.items():
        subs[name] = f"{value:.6e}"

    netlist = Template(template_text).substitute(subs)

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
    netlist_path = os.path.join(tmp_dir, "ota2_cmaes.spice")

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

    vdd = float(FIXED_PARAMS["vdd"])
    power_w = vdd * 2 * params["ibias"]
    metrics["power_w"] = power_w

    if metrics["dc_gain_db"] is None or metrics["gbw_hz"] is None:
        return None
    if metrics["dc_gain_db"] < 0 or metrics["dc_gain_db"] > 200:
        return None
    if metrics["gbw_hz"] > 50e9:
        return None

    return {k: v for k, v in metrics.items() if v is not None}


# ---------------------------------------------------------------------------
# CMA-ES Optimizer
# ---------------------------------------------------------------------------
class CMAESOptimizer:
    """CMA-ES optimizer for OTA2 circuit design."""

    def __init__(
        self,
        surrogate: XGBSurrogate,
        episode_store: EpisodeStore,
        population_size: int = 20,
        max_generations: int = 50,
        seed: int = 42,
        sigma0: float = 0.3,
    ):
        self.surrogate = surrogate
        self.episode_store = episode_store
        self.population_size = population_size
        self.max_generations = max_generations
        self.seed = seed
        self.sigma0 = sigma0

        # History
        self.all_params: list[dict[str, float]] = []
        self.all_metrics: list[dict[str, float]] = []
        self.all_feasible: list[bool] = []
        self.all_scores: list[float] = []
        self.best_feasible_idx: int | None = None
        self.best_feasible_score: float = -np.inf

    def _objective(self, x_log: np.ndarray) -> float:
        """CMA-ES objective function (to minimize).

        Returns negative scalarized objective (since CMA-ES minimizes).
        Death penalty for infeasible solutions.
        """
        # Clip to bounds
        x_clipped = np.clip(x_log, LOG_BOUNDS_LO, LOG_BOUNDS_HI)

        # Penalty for out-of-bounds (shouldn't happen with bounds, but safety)
        if not np.allclose(x_log, x_clipped):
            return DEATH_PENALTY

        params = log_to_params(x_clipped)
        metrics = self.surrogate.predict(params)

        # Check constraints - death penalty
        feasible, violated = check_constraints(metrics)
        if not feasible:
            # Soft death penalty: large value + distance from feasibility
            penalty = 100.0
            for metric_name, (op, threshold) in CONSTRAINTS.items():
                if metric_name not in metrics:
                    penalty += 10.0
                    continue
                val = metrics[metric_name]
                if op == ">=" and val < threshold:
                    penalty += (threshold - val) / abs(threshold)
                elif op == "<=" and val > threshold:
                    penalty += (val - threshold) / abs(threshold)
            return penalty

        # Feasible: return negative scalarized objective (minimize)
        score = scalarize_objective(metrics)
        return -score

    def run(self, verify_with_ngspice: bool = True) -> dict:
        """Run CMA-ES optimization.

        Args:
            verify_with_ngspice: If True, verify top candidates with ngspice.

        Returns:
            Dict with optimization results.
        """
        import cma

        print("=" * 70)
        print("CMA-ES OPTIMIZATION - OTA2 Circuit Design")
        print("=" * 70)
        print(f"Population size: {self.population_size}")
        print(f"Max generations: {self.max_generations}")
        print(f"Sigma0: {self.sigma0}")
        print(f"Design variables: {VAR_NAMES}")
        print(f"Verify with ngspice: {verify_with_ngspice}")
        print("-" * 70)

        start_time = time.time()

        # Initial point: center of log space
        x0 = (LOG_BOUNDS_LO + LOG_BOUNDS_HI) / 2.0

        # CMA-ES options
        opts = cma.CMAOptions()
        opts.set("seed", self.seed)
        opts.set("popsize", self.population_size)
        opts.set("maxiter", self.max_generations)
        opts.set("bounds", [LOG_BOUNDS_LO.tolist(), LOG_BOUNDS_HI.tolist()])
        opts.set("verbose", -1)  # Suppress CMA internal output
        opts.set("tolfun", 1e-8)

        # Run CMA-ES
        es = cma.CMAEvolutionStrategy(x0.tolist(), self.sigma0, opts)

        generation = 0
        while not es.stop():
            generation += 1
            solutions = es.ask()
            fitnesses = []

            for x in solutions:
                x_arr = np.array(x)
                fitness = self._objective(x_arr)
                fitnesses.append(fitness)

                # Record
                x_clipped = np.clip(x_arr, LOG_BOUNDS_LO, LOG_BOUNDS_HI)
                params = log_to_params(x_clipped)
                metrics = self.surrogate.predict(params)
                feasible, violated = check_constraints(metrics)
                score = scalarize_objective(metrics) if feasible else -fitness

                self.all_params.append(params)
                self.all_metrics.append(metrics)
                self.all_feasible.append(feasible)
                self.all_scores.append(score if feasible else -1.0)

                if feasible and score > self.best_feasible_score:
                    self.best_feasible_score = score
                    self.best_feasible_idx = len(self.all_params) - 1

            es.tell(solutions, fitnesses)

            # Progress
            if generation % 5 == 0 or generation == 1:
                n_feasible = sum(self.all_feasible)
                best_str = f"best={self.best_feasible_score:.4f}" if self.best_feasible_idx is not None else "no feasible"
                print(
                    f"  Gen {generation:3d} | "
                    f"best_fit={es.result.fbest:.4f} | "
                    f"feasible={n_feasible}/{len(self.all_params)} | {best_str}"
                )

        es.result_pretty()

        # Verify best candidates with ngspice
        verified_metrics = None
        if verify_with_ngspice and self.best_feasible_idx is not None:
            print("\n[Verification] Running ngspice on best candidate...")
            tmp_dir = tempfile.mkdtemp(prefix="ota2_cmaes_verify_")
            try:
                best_params = self.all_params[self.best_feasible_idx]
                ngspice_metrics = run_ngspice(best_params, tmp_dir)
                if ngspice_metrics is not None:
                    verified_metrics = ngspice_metrics
                    feasible, violated = check_constraints(ngspice_metrics)
                    print(f"  ngspice verification: {'PASS' if feasible else 'FAIL'}")
                    if violated:
                        print(f"  Violated: {violated}")

                    # Record verification
                    self.episode_store.record_episode(
                        params=best_params,
                        metrics=ngspice_metrics,
                        feasible=feasible,
                        circuit_family="two_stage_ota",
                        constraints_violated=violated,
                        notes="CMA-ES best candidate - ngspice verified",
                    )
                else:
                    print("  ngspice simulation failed!")
            finally:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        elapsed = time.time() - start_time

        # Compile results
        results = self._compile_results(elapsed, verified_metrics)
        self._print_summary(results)
        return results

    def _compile_results(self, elapsed: float, verified_metrics: dict | None = None) -> dict:
        """Compile optimization results."""
        results = {
            "optimizer": "cma_es",
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "elapsed_seconds": elapsed,
            "total_evaluations": len(self.all_params),
            "feasible_count": sum(self.all_feasible),
            "feasibility_rate": sum(self.all_feasible) / len(self.all_feasible) if self.all_feasible else 0,
        }

        if self.best_feasible_idx is not None:
            best_params = self.all_params[self.best_feasible_idx]
            best_metrics = verified_metrics or self.all_metrics[self.best_feasible_idx]
            feasible, _ = check_constraints(best_metrics)

            results["best_design"] = {
                "parameters": best_params,
                "metrics": best_metrics,
                "score": self.best_feasible_score,
                "verified_with_ngspice": verified_metrics is not None,
                "ngspice_feasible": feasible if verified_metrics else None,
            }
            results["spec_met"] = feasible
        else:
            results["best_design"] = None
            results["spec_met"] = False

        return results

    def _print_summary(self, results: dict) -> None:
        """Print summary."""
        print("\n" + "=" * 70)
        print("CMA-ES RESULTS")
        print("=" * 70)
        print(f"Total evaluations: {results['total_evaluations']}")
        print(f"Feasible designs: {results['feasible_count']}")
        print(f"Feasibility rate: {results['feasibility_rate']:.1%}")
        print(f"Elapsed time: {results['elapsed_seconds']:.1f}s")

        if results["best_design"]:
            bd = results["best_design"]
            print(f"\nBest Design:")
            print(f"  Score: {bd['score']:.4f}")
            print(f"  Verified: {bd.get('verified_with_ngspice', False)}")
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
        else:
            print("\nNo feasible design found!")

        print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="CMA-ES Optimization for OTA2")
    parser.add_argument("--popsize", type=int, default=20, help="Population size")
    parser.add_argument("--generations", type=int, default=50, help="Max generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sigma0", type=float, default=0.3, help="Initial step size")
    parser.add_argument("--no-ngspice", action="store_true", help="Skip ngspice verification")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    # Load surrogate
    print("Loading XGBoost surrogate model...")
    surrogate = XGBSurrogate()
    surrogate.load()
    print(f"  Metrics: {surrogate.target_metrics}")

    # Initialize episode store
    store_path = REPO_ROOT / "data" / "optimization_results" / "cmaes_episodes.json"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    episode_store = EpisodeStore(store_path)

    # Run optimization
    optimizer = CMAESOptimizer(
        surrogate=surrogate,
        episode_store=episode_store,
        population_size=args.popsize,
        max_generations=args.generations,
        seed=args.seed,
        sigma0=args.sigma0,
    )

    results = optimizer.run(verify_with_ngspice=not args.no_ngspice)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPO_ROOT / "data" / "optimization_results" / f"ota2_cmaes_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
