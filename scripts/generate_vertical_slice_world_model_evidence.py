"""Generate world-model paper evidence for any frozen benchmark vertical slice."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.bandgap import run_bandgap_world_model_evidence
from libs.vertical_slices.folded_cascode import run_folded_cascode_world_model_evidence
from libs.vertical_slices.ldo import run_ldo_world_model_evidence
from libs.vertical_slices.ota2 import run_ota_experiment_suite
from libs.vertical_slices.world_model_evidence import run_vertical_slice_world_model_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["ota2", "folded_cascode", "ldo", "bandgap"], required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeat-runs", type=int, default=5)
    parser.add_argument("--max-simulations", type=int, default=6)
    parser.add_argument("--max-candidates-per-step", type=int, default=3)
    parser.add_argument("--output-root", default="research/papers")
    args = parser.parse_args()

    budget = ExperimentBudget(
        max_simulations=args.max_simulations,
        max_candidates_per_step=args.max_candidates_per_step,
    )
    family_root = Path(args.output_root) / f"{args.family}_v1"

    if args.family == "ota2":
        bundle = run_vertical_slice_world_model_evidence(
            task_slug="ota2-v1",
            suite_runner=run_ota_experiment_suite,
            steps=args.steps,
            repeat_runs=args.repeat_runs,
            budget=budget,
            output_root=family_root,
        )
    elif args.family == "folded_cascode":
        bundle = run_folded_cascode_world_model_evidence(
            steps=args.steps,
            repeat_runs=args.repeat_runs,
            budget=budget,
            output_root=family_root,
        )
    elif args.family == "ldo":
        bundle = run_ldo_world_model_evidence(
            steps=args.steps,
            repeat_runs=args.repeat_runs,
            budget=budget,
            output_root=family_root,
        )
    else:
        bundle = run_bandgap_world_model_evidence(
            steps=args.steps,
            repeat_runs=args.repeat_runs,
            budget=budget,
            output_root=family_root,
        )

    print(bundle.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
