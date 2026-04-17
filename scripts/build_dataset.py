"""Build a structured world-model dataset from runnable benchmark slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.schema.experiment import ExperimentBudget
from libs.world_model.dataset_builder import build_world_model_dataset


def _suite_runner(task_name: str):
    if task_name == "ota2":
        from libs.vertical_slices.ota2 import run_ota_experiment_suite

        return run_ota_experiment_suite
    if task_name == "folded_cascode":
        from libs.vertical_slices.folded_cascode import run_folded_cascode_experiment_suite

        return run_folded_cascode_experiment_suite
    if task_name == "ldo":
        from libs.vertical_slices.ldo import run_ldo_experiment_suite

        return run_ldo_experiment_suite
    if task_name == "bandgap":
        from libs.vertical_slices.bandgap import run_bandgap_experiment_suite

        return run_bandgap_experiment_suite
    raise ValueError(f"unsupported task: {task_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="ota2,folded_cascode,ldo,bandgap")
    parser.add_argument("--modes", default="full_system,top_k_baseline,random_search_baseline,full_simulation_baseline")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--repeat-runs", type=int, default=2)
    parser.add_argument("--max-simulations", type=int, default=4)
    parser.add_argument("--max-candidates-per-step", type=int, default=2)
    parser.add_argument("--max-records-per-family", type=int, default=8)
    parser.add_argument("--dataset-name", default="world_model_multitask_v1")
    parser.add_argument("--output", default="research/datasets/world_model_multitask_v1.json")
    args = parser.parse_args()

    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    suites = []
    budget = ExperimentBudget(
        max_simulations=args.max_simulations,
        max_candidates_per_step=args.max_candidates_per_step,
    )
    for task_name in task_names:
        runner = _suite_runner(task_name)
        suites.append(
            runner(
                steps=args.steps,
                repeat_runs=args.repeat_runs,
                budget=budget,
                modes=modes,
                task_id=f"dataset-{task_name}-v1",
            )
        )

    bundle = build_world_model_dataset(
        suites,
        dataset_name=args.dataset_name,
        max_records_per_family=args.max_records_per_family,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_id": bundle.dataset_id,
                "record_count": len(bundle.records),
                "family_coverage": bundle.family_coverage,
                "output": str(output_path),
                "sampling_policy": bundle.sampling_policy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
