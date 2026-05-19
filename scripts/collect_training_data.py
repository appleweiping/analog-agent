"""Collect training data for OTA2 XGBoost world model via ngspice simulation.

Generates Latin Hypercube samples across the OTA2 design space, runs AC analysis
in ngspice, and extracts key performance metrics (dc_gain_db, gbw_hz,
phase_margin_deg, power_w). Results are saved to data/ota2_training_data.json.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from string import Template

import numpy as np
from scipy.stats.qmc import LatinHypercube

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
NGSPICE_BIN = Path(r"D:\research\Agent-AI4EDA\tools\ngspice\bin\ngspice_con.exe")
TEMPLATE_PATH = REPO_ROOT / "templates" / "netlist" / "ota2" / "v1" / "ota2_demonstrator_truth.spice.tpl"
OUTPUT_PATH = REPO_ROOT / "data" / "ota2_training_data.json"

# ---------------------------------------------------------------------------
# Design variable ranges
# ---------------------------------------------------------------------------
DESIGN_VARS = {
    "gm1": (0.5e-3, 5e-3),
    "gm2": (1e-3, 10e-3),
    "ro1": (10e3, 500e3),
    "ro2": (5e3, 200e3),
    "cc": (0.5e-12, 5e-12),
    "ibias": (10e-6, 500e-6),
}

# Fixed parameters
FIXED_PARAMS = {
    "vdd": "1.2",
    "vin_cm": "0.6",
    "vin_step_high": "0.61",
    "cload": "2e-12",
    "cp1": "0.1e-12",
    "truth_mode": "configured",
    "template_id": "ota2_xgb_data_collection",
    "p2_hint_hz": "1e9",
}

N_SAMPLES = 500


def generate_samples(n: int) -> list[dict[str, float]]:
    """Generate n parameter combinations using Latin Hypercube Sampling."""
    dim = len(DESIGN_VARS)
    sampler = LatinHypercube(d=dim, seed=42)
    unit_samples = sampler.random(n=n)

    var_names = list(DESIGN_VARS.keys())
    samples = []
    for i in range(n):
        params = {}
        for j, name in enumerate(var_names):
            lo, hi = DESIGN_VARS[name]
            # Use log-uniform for parameters spanning orders of magnitude
            if hi / lo > 10:
                params[name] = float(np.exp(
                    np.log(lo) + unit_samples[i, j] * (np.log(hi) - np.log(lo))
                ))
            else:
                params[name] = float(lo + unit_samples[i, j] * (hi - lo))
        samples.append(params)
    return samples


def build_netlist(params: dict[str, float]) -> str:
    """Substitute parameters into the template and append AC analysis."""
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

    # Build substitution dict
    subs = dict(FIXED_PARAMS)
    for name, value in params.items():
        subs[name] = f"{value:.6e}"

    # Use Template substitution
    netlist = Template(template_text).substitute(subs)

    # Append AC analysis and control block
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

        # Parse "dc_gain = ..." or "gain_db[0] = ..."
        if "dc_gain" in line_stripped and "=" in line_stripped:
            match = re.search(r"=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line)
            if match:
                metrics["dc_gain_db"] = float(match.group(1))

        # Parse "gbw_freq = ..."
        if "gbw_freq" in line_stripped and "=" in line_stripped:
            match = re.search(r"=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line)
            if match:
                val = float(match.group(1))
                if val > 0:
                    metrics["gbw_hz"] = val

        # Parse "phase_margin = ..."
        if "phase_margin" in line_stripped and "=" in line_stripped:
            match = re.search(r"=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line)
            if match:
                metrics["phase_margin_deg"] = float(match.group(1))

    return metrics


def run_simulation(params: dict[str, float], tmp_dir: str) -> dict | None:
    """Run a single ngspice simulation and return results."""
    netlist = build_netlist(params)
    netlist_path = os.path.join(tmp_dir, "ota2_sim.spice")

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
    except (subprocess.TimeoutExpired, OSError) as e:
        return None

    metrics = parse_ngspice_output(stdout)

    # Compute power analytically: P = vdd * 2*ibias
    vdd = float(FIXED_PARAMS["vdd"])
    power_w = vdd * 2 * params["ibias"]
    metrics["power_w"] = power_w

    # Check if simulation produced valid results
    if metrics["dc_gain_db"] is None or metrics["gbw_hz"] is None:
        return None

    # Sanity checks
    if metrics["dc_gain_db"] < 0 or metrics["dc_gain_db"] > 200:
        return None
    if metrics["gbw_hz"] is not None and metrics["gbw_hz"] > 50e9:
        return None

    return {
        "params": params,
        "metrics": {k: v for k, v in metrics.items() if v is not None},
        "feasible": True,
    }


def main():
    print(f"Generating {N_SAMPLES} LHS samples...")
    samples = generate_samples(N_SAMPLES)

    print(f"Running ngspice simulations...")
    print(f"  ngspice: {NGSPICE_BIN}")
    print(f"  template: {TEMPLATE_PATH}")

    results = []
    failed = 0
    start_time = time.time()

    with tempfile.TemporaryDirectory(prefix="ota2_data_") as tmp_dir:
        for i, params in enumerate(samples):
            result = run_simulation(params, tmp_dir)
            if result is not None:
                results.append(result)
            else:
                failed += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (N_SAMPLES - i - 1) / rate
                print(
                    f"  [{i+1}/{N_SAMPLES}] "
                    f"success={len(results)}, failed={failed}, "
                    f"rate={rate:.1f} sim/s, ETA={eta:.0f}s"
                )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total samples: {N_SAMPLES}")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {len(results)/N_SAMPLES*100:.1f}%")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
