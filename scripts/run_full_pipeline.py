"""End-to-end analog-agent optimization pipeline.

Orchestrates the full flow:
1. Topology selection
2. Design space definition
3. World model loading
4. Memory warm start
5. Optimization loop (BO or CMA-ES)
6. Results reporting and memory consolidation
"""

from __future__ import annotations

import argparse
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
from libs.topology.catalog import TOPOLOGY_CATALOG

NGSPICE_BIN = Path(r"D:\research\Agent-AI4EDA\tools\ngspice\bin\ngspice_con.exe")
TEMPLATE_PATH = REPO_ROOT / "templates" / "netlist" / "ota2" / "v1" / "ota2_demonstrator_truth.spice.tpl"

# ---------------------------------------------------------------------------
# Circuit configurations
# ---------------------------------------------------------------------------
CIRCUIT_CONFIGS = {
    "ota2": {
        "family": "two_stage_ota",
        "topology_key": "two_stage_ota",
        "design_vars": {
            "gm1": (0.5e-3, 5e-3),
            "gm2": (1e-3, 10e-3),
            "ro1": (10e3, 500e3),
            "ro2": (5e3, 200e3),
            "cc": (0.5e-12, 5e-12),
            "ibias": (10e-6, 500e-6),
        },
        "constraints": {
            "dc_gain_db": (">=", 60.0),
            "gbw_hz": (">=", 80e6),
            "phase_margin_deg": (">=", 55.0),
            "power_w": ("<=", 1.5e-3),
        },
        "objectives": {
            "gbw_hz": "maximize",
            "power_w": "minimize",
        },
        "fixed_params": {
            "vdd": "1.2",
            "vin_cm": "0.6",
            "vin_step_high": "0.61",
            "cload": "2e-12",
            "cp1": "0.1e-12",
            "truth_mode": "configured",
            "template_id": "ota2_pipeline",
            "p2_hint_hz": "1e9",
        },
    }
}


# ---------------------------------------------------------------------------
# ngspice simulation
# ---------------------------------------------------------------------------
def build_netlist(params: dict[str, float], fixed_params: dict[str, str]) -> str:
    """Build ngspice netlist from parameters."""
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

    subs = dict(fixed_params)
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


def run_ngspice(params: dict[str, float], fixed_params: dict[str, str], tmp_dir: str) -> dict[str, float] | None:
    """Run ngspice simulation and return metrics."""
    netlist = build_netlist(params, fixed_params)
    netlist_path = os.path.join(tmp_dir, "ota2_pipeline.spice")

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

    vdd = float(fixed_params["vdd"])
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
# Pipeline stages
# ---------------------------------------------------------------------------
def stage_topology_selection(circuit: str) -> dict:
    """Stage 1: Select topology for the given circuit."""
    print("\n" + "=" * 70)
    print("[Stage 1] TOPOLOGY SELECTION")
    print("=" * 70)

    config = CIRCUIT_CONFIGS[circuit]
    topology_key = config["topology_key"]

    if topology_key in TOPOLOGY_CATALOG:
        topology = TOPOLOGY_CATALOG[topology_key]()
        print(f"  Circuit: {circuit}")
        print(f"  Topology: {topology.name}")
        print(f"  Family: {topology.family}")
        print(f"  Nodes: {len(topology.nodes)}")
        print(f"  Edges: {len(topology.edges)}")
        return {"topology": topology_key, "name": topology.name, "family": topology.family}
    else:
        print(f"  WARNING: Topology '{topology_key}' not in catalog, using config directly")
        return {"topology": topology_key, "name": topology_key, "family": config["family"]}


def stage_design_space(circuit: str) -> dict:
    """Stage 2: Define design space."""
    print("\n" + "=" * 70)
    print("[Stage 2] DESIGN SPACE DEFINITION")
    print("=" * 70)

    config = CIRCUIT_CONFIGS[circuit]
    design_vars = config["design_vars"]

    print(f"  Variables: {len(design_vars)}")
    for name, (lo, hi) in design_vars.items():
        print(f"    {name:8s}: [{lo:.2e}, {hi:.2e}]  (range: {hi/lo:.0f}x)")

    print(f"\n  Constraints:")
    for metric, (op, threshold) in config["constraints"].items():
        print(f"    {metric:20s} {op} {threshold:.4e}")

    print(f"\n  Objectives:")
    for metric, direction in config["objectives"].items():
        print(f"    {metric:20s} -> {direction}")

    return {"design_vars": design_vars, "constraints": config["constraints"]}


def stage_world_model_loading() -> XGBSurrogate:
    """Stage 3: Load world model."""
    print("\n" + "=" * 70)
    print("[Stage 3] WORLD MODEL LOADING")
    print("=" * 70)

    surrogate = XGBSurrogate()
    surrogate.load()

    print(f"  Model directory: {surrogate.model_dir}")
    print(f"  Features: {surrogate.feature_names}")
    print(f"  Target metrics: {surrogate.target_metrics}")

    cv_scores = surrogate.get_cv_scores()
    print(f"\n  Cross-validation R^2 scores:")
    for metric, scores in cv_scores.items():
        print(f"    {metric:20s}: R²={scores['r2_mean']:.4f} ± {scores['r2_std']:.4f}")

    return surrogate


def stage_memory_warm_start(circuit: str, episode_store: EpisodeStore) -> dict:
    """Stage 4: Check memory for prior knowledge."""
    print("\n" + "=" * 70)
    print("[Stage 4] MEMORY WARM START")
    print("=" * 70)

    config = CIRCUIT_CONFIGS[circuit]
    family = config["family"]

    history = episode_store.get_history(circuit_family=family)
    total = episode_store.get_total_count()
    feasible = episode_store.get_feasible_count()

    print(f"  Episode store: {episode_store.store_path}")
    print(f"  Total episodes: {total}")
    print(f"  Feasible episodes: {feasible}")
    print(f"  Episodes for '{family}': {len(history)}")

    warm_start_info = {
        "total_episodes": total,
        "feasible_episodes": feasible,
        "family_episodes": len(history),
    }

    if history:
        # Get best prior designs
        best_designs = episode_store.get_best_designs(n=3, metric="gbw_hz", maximize=True)
        if best_designs:
            print(f"\n  Best prior designs (by GBW):")
            for i, ep in enumerate(best_designs[:3]):
                gbw = ep["metrics"].get("gbw_hz", 0)
                print(f"    #{i+1}: GBW={gbw:.2e} Hz, feasible={ep['feasible']}")
            warm_start_info["best_prior_gbw"] = best_designs[0]["metrics"].get("gbw_hz", 0)

        # Get feasible region bounds
        bounds = episode_store.get_feasible_region_bounds()
        if bounds:
            print(f"\n  Feasible region bounds:")
            for param, b in bounds.items():
                print(f"    {param:8s}: [{b['min']:.3e}, {b['max']:.3e}] (mean={b['mean']:.3e})")
            warm_start_info["feasible_bounds"] = bounds
    else:
        print("  No prior episodes found - starting fresh")

    return warm_start_info


def stage_optimization(
    circuit: str,
    optimizer_type: str,
    surrogate: XGBSurrogate,
    episode_store: EpisodeStore,
    budget: int,
    seed: int,
) -> dict:
    """Stage 5: Run optimization loop."""
    print("\n" + "=" * 70)
    print(f"[Stage 5] OPTIMIZATION ({optimizer_type.upper()})")
    print("=" * 70)

    if optimizer_type == "bo":
        from scripts.run_bo_optimization import BayesianOptimizer

        optimizer = BayesianOptimizer(
            surrogate=surrogate,
            episode_store=episode_store,
            budget=budget,
            seed=seed,
        )
        results = optimizer.run(verify_with_ngspice=True)

    elif optimizer_type == "cmaes":
        from scripts.run_cmaes_optimization import CMAESOptimizer

        # Convert budget to generations (budget / popsize)
        popsize = 20
        generations = max(budget // popsize, 10)

        optimizer = CMAESOptimizer(
            surrogate=surrogate,
            episode_store=episode_store,
            population_size=popsize,
            max_generations=generations,
            seed=seed,
        )
        results = optimizer.run(verify_with_ngspice=True)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}. Use 'bo' or 'cmaes'.")

    return results


def stage_results(
    circuit: str,
    optimizer_type: str,
    results: dict,
    episode_store: EpisodeStore,
) -> dict:
    """Stage 6: Report results and consolidate memory."""
    print("\n" + "=" * 70)
    print("[Stage 6] RESULTS & MEMORY CONSOLIDATION")
    print("=" * 70)

    config = CIRCUIT_CONFIGS[circuit]
    constraints = config["constraints"]

    # Final summary
    print(f"\n  Optimizer: {optimizer_type}")
    print(f"  Total evaluations: {results.get('total_evaluations', 'N/A')}")
    print(f"  Feasible designs: {results.get('feasible_count', 0)}")
    print(f"  Spec met: {results.get('spec_met', False)}")

    if results.get("best_design"):
        bd = results["best_design"]
        print(f"\n  {'='*50}")
        print(f"  BEST DESIGN FOUND")
        print(f"  {'='*50}")
        print(f"\n  Parameters:")
        for name, val in bd["parameters"].items():
            lo, hi = config["design_vars"][name]
            pct = (val - lo) / (hi - lo) * 100
            print(f"    {name:8s} = {val:.4e}  ({pct:.0f}% of range)")

        print(f"\n  {'─'*50}")
        print(f"  {'Metric':<22s} {'Value':>12s}  {'Spec':>16s}  {'Status':>6s}")
        print(f"  {'─'*50}")

        all_met = True
        for metric, (op, threshold) in constraints.items():
            val = bd["metrics"].get(metric, float("nan"))
            if op == ">=" and val >= threshold:
                status = "OK"
            elif op == "<=" and val <= threshold:
                status = "OK"
            else:
                status = "FAIL"
                all_met = False

            # Format value
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

        print(f"  {'─'*50}")
        print(f"\n  OVERALL: {'ALL SPECS MET' if all_met else 'SOME SPECS NOT MET'}")
    else:
        print("\n  No feasible design found!")

    # Memory consolidation
    print(f"\n  Memory consolidation:")
    print(f"    Total episodes in store: {episode_store.get_total_count()}")
    print(f"    Feasible episodes: {episode_store.get_feasible_count()}")
    print(f"    Feasibility rate: {episode_store.get_feasibility_rate():.1%}")

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = REPO_ROOT / "data" / "optimization_results" / f"{circuit}_{optimizer_type}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_results = {
        "circuit": circuit,
        "optimizer": optimizer_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "memory_stats": {
            "total_episodes": episode_store.get_total_count(),
            "feasible_episodes": episode_store.get_feasible_count(),
            "feasibility_rate": episode_store.get_feasibility_rate(),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    return final_results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(circuit: str, optimizer: str, budget: int, seed: int) -> dict:
    """Run the complete analog-agent optimization pipeline."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  ANALOG-AGENT: End-to-End Optimization Pipeline".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print(f"\n  Circuit: {circuit}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Budget: {budget}")
    print(f"  Seed: {seed}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if circuit not in CIRCUIT_CONFIGS:
        print(f"\nERROR: Unknown circuit '{circuit}'. Available: {list(CIRCUIT_CONFIGS.keys())}")
        sys.exit(1)

    pipeline_start = time.time()

    # Stage 1: Topology
    topo_info = stage_topology_selection(circuit)

    # Stage 2: Design space
    space_info = stage_design_space(circuit)

    # Stage 3: World model
    surrogate = stage_world_model_loading()

    # Stage 4: Memory warm start
    store_path = REPO_ROOT / "data" / "optimization_results" / f"{circuit}_{optimizer}_episodes.json"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    episode_store = EpisodeStore(store_path)
    warm_start_info = stage_memory_warm_start(circuit, episode_store)

    # Stage 5: Optimization
    opt_results = stage_optimization(
        circuit=circuit,
        optimizer_type=optimizer,
        surrogate=surrogate,
        episode_store=episode_store,
        budget=budget,
        seed=seed,
    )

    # Stage 6: Results
    final_results = stage_results(circuit, optimizer, opt_results, episode_store)

    pipeline_elapsed = time.time() - pipeline_start
    print(f"\n{'#'*70}")
    print(f"  Pipeline completed in {pipeline_elapsed:.1f}s")
    print(f"{'#'*70}\n")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Analog-Agent End-to-End Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_full_pipeline.py --circuit ota2 --optimizer bo --budget 50
  python scripts/run_full_pipeline.py --circuit ota2 --optimizer cmaes --budget 100
        """,
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default="ota2",
        choices=list(CIRCUIT_CONFIGS.keys()),
        help="Circuit to optimize (default: ota2)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="bo",
        choices=["bo", "cmaes"],
        help="Optimization algorithm (default: bo)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=50,
        help="Evaluation budget (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    results = run_pipeline(
        circuit=args.circuit,
        optimizer=args.optimizer,
        budget=args.budget,
        seed=args.seed,
    )

    # Exit code based on spec met
    spec_met = results.get("results", {}).get("spec_met", False)
    sys.exit(0 if spec_met else 1)


if __name__ == "__main__":
    main()
