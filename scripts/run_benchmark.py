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
from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.ota2 import run_ota_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="ota2")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeat-runs", type=int, default=5)
    parser.add_argument("--max-simulations", type=int, default=6)
    parser.add_argument("--max-candidates-per-step", type=int, default=3)
    parser.add_argument("--output", default="research/benchmarks/ota2_experiment_results.json")
    parser.add_argument("--stats-json", default="research/benchmarks/ota2_stats_summary.json")
    parser.add_argument("--stats-csv", default="research/benchmarks/ota2_stats_summary.csv")
    args = parser.parse_args()
    if args.benchmark != "ota2":
        raise ValueError("only ota2 is currently supported in the Day-4 benchmark runner")

    budget = ExperimentBudget(
        max_simulations=args.max_simulations,
        max_candidates_per_step=args.max_candidates_per_step,
    )
    suite = run_ota_experiment_suite(
        budget=budget,
        steps=args.steps,
        repeat_runs=args.repeat_runs,
        task_id="benchmark-ota2-v1",
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(suite.model_dump(), indent=2), encoding="utf-8")
    stats_json = export_stats_json(suite, args.stats_json)
    stats_csv = export_stats_csv(suite, args.stats_csv)
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
