"""Generate paper-facing figures and tables for world-model utility evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.eval.paper_evidence import build_world_model_evidence_bundle
from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.ota2 import run_ota_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeat-runs", type=int, default=5)
    parser.add_argument("--max-simulations", type=int, default=6)
    parser.add_argument("--max-candidates-per-step", type=int, default=3)
    parser.add_argument("--output-root", default="research/papers")
    parser.add_argument("--task-id", default="benchmark-ota2-v1-world-model-evidence")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    figs_root = output_root / "figs"
    tables_root = output_root / "tables"
    figs_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    suite = run_ota_experiment_suite(
        steps=args.steps,
        repeat_runs=args.repeat_runs,
        comparison_profile="methodology",
        budget=ExperimentBudget(
            max_simulations=args.max_simulations,
            max_candidates_per_step=args.max_candidates_per_step,
        ),
        task_id=args.task_id,
        export_directory=output_root / "methodology",
        force_full_steps=True,
    )
    baseline_suite = run_ota_experiment_suite(
        steps=args.steps,
        repeat_runs=args.repeat_runs,
        comparison_profile="baseline",
        budget=ExperimentBudget(
            max_simulations=args.max_simulations,
            max_candidates_per_step=args.max_candidates_per_step,
        ),
        task_id=f"{args.task_id}-baseline",
        export_directory=output_root / "baseline",
        force_full_steps=False,
        modes=["full_simulation_baseline", "no_world_model_baseline", "full_system"],
    )

    bundle = build_world_model_evidence_bundle(
        suite,
        baseline_suite=baseline_suite,
        figures_dir=figs_root,
        tables_dir=tables_root,
        json_output_path=output_root / "world_model_evidence_bundle.json",
    )
    print(bundle.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
