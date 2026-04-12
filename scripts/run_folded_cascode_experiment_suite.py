"""Run the frozen folded-cascode v1 experiment suite and export benchmark stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.folded_cascode import run_folded_cascode_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeat-runs", type=int, default=5)
    parser.add_argument("--max-simulations", type=int, default=6)
    parser.add_argument("--max-candidates-per-step", type=int, default=3)
    parser.add_argument("--comparison-profile", choices=["baseline", "methodology"], default="baseline")
    parser.add_argument("--output", default="research/benchmarks/folded_cascode_experiment_results.json")
    parser.add_argument("--export-dir", default="research/benchmarks")
    args = parser.parse_args()

    suite = run_folded_cascode_experiment_suite(
        steps=args.steps,
        repeat_runs=args.repeat_runs,
        budget=ExperimentBudget(
            max_simulations=args.max_simulations,
            max_candidates_per_step=args.max_candidates_per_step,
        ),
        comparison_profile=args.comparison_profile,
        export_directory=args.export_dir,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(suite.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "stats_dir": str(Path(args.export_dir)), "run_count": len(suite.runs)}, indent=2))


if __name__ == "__main__":
    main()
