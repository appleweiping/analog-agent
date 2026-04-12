"""Run structured benchmark experiments for Day-4 baseline comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.eval.stats import export_stats_csv, export_stats_json
from libs.eval.benchmark_registry import list_benchmark_definitions, load_benchmark_task_definition, load_benchmark_suite_definition
from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.folded_cascode import run_folded_cascode_experiment_suite
from libs.vertical_slices.ldo import run_ldo_experiment_suite
from libs.vertical_slices.ota2 import run_ota_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="ota2")
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeat-runs", type=int, default=5)
    parser.add_argument("--max-simulations", type=int, default=6)
    parser.add_argument("--max-candidates-per-step", type=int, default=3)
    parser.add_argument("--output", default="")
    parser.add_argument("--stats-json", default="")
    parser.add_argument("--stats-csv", default="")
    args = parser.parse_args()
    if args.list_benchmarks:
        suite = load_benchmark_suite_definition()
        payload = {
            "suite_id": suite.suite_id,
            "primary_benchmark_id": suite.primary_benchmark_id,
            "benchmarks": [
                {
                    "benchmark_id": benchmark.benchmark_id,
                    "family": benchmark.family,
                    "role": benchmark.benchmark_role,
                    "execution_readiness": benchmark.execution_readiness,
                }
                for benchmark in list_benchmark_definitions()
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    benchmark_definition = load_benchmark_task_definition(args.benchmark)
    if benchmark_definition.execution_readiness != "frozen_runnable":
        raise ValueError(
            f"benchmark '{args.benchmark}' is defined but not yet runnable; current readiness is {benchmark_definition.execution_readiness}"
        )
    budget = ExperimentBudget(
        max_simulations=args.max_simulations,
        max_candidates_per_step=args.max_candidates_per_step,
    )
    benchmark_name = (
        "ota2"
        if benchmark_definition.benchmark_id == "ota2_v1"
        else "folded_cascode"
        if benchmark_definition.benchmark_id == "folded_cascode_v1"
        else "ldo"
    )
    if benchmark_definition.benchmark_id == "ota2_v1":
        suite = run_ota_experiment_suite(
            budget=budget,
            steps=args.steps,
            repeat_runs=args.repeat_runs,
            task_id="benchmark-ota2-v1",
        )
    elif benchmark_definition.benchmark_id == "folded_cascode_v1":
        suite = run_folded_cascode_experiment_suite(
            budget=budget,
            steps=args.steps,
            repeat_runs=args.repeat_runs,
            task_id="benchmark-folded-cascode-v1",
        )
    elif benchmark_definition.benchmark_id == "ldo_v1":
        suite = run_ldo_experiment_suite(
            budget=budget,
            steps=args.steps,
            repeat_runs=args.repeat_runs,
            task_id="benchmark-ldo-v1",
        )
    else:
        raise ValueError(f"benchmark '{benchmark_definition.benchmark_id}' is frozen_runnable but not yet wired into run_benchmark.py")

    output_path = Path(args.output or f"research/benchmarks/{benchmark_name}_experiment_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(suite.model_dump(), indent=2), encoding="utf-8")
    stats_json = export_stats_json(suite, args.stats_json or f"research/benchmarks/{benchmark_name}_stats_summary.json")
    stats_csv = export_stats_csv(suite, args.stats_csv or f"research/benchmarks/{benchmark_name}_stats_summary.csv")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "stats_json": str(stats_json),
                "stats_csv": str(stats_csv),
                "run_count": len(suite.runs),
                "mode_count": len(suite.modes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
