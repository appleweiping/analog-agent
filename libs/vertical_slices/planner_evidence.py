"""Shared planner-ablation evidence runner for frozen benchmark vertical slices."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from libs.eval.planner_evidence import build_planner_ablation_evidence_bundle
from libs.schema.experiment import ExperimentBudget, ExperimentSuiteResult
from libs.schema.paper_evidence import PlannerAblationEvidenceBundle

SuiteRunner = Callable[..., ExperimentSuiteResult]


def run_vertical_slice_planner_evidence(
    *,
    task_slug: str,
    suite_runner: SuiteRunner,
    steps: int = 3,
    repeat_runs: int = 5,
    budget: ExperimentBudget | None = None,
    backend_preference: str = "ngspice",
    fidelity_level: str = "focused_truth",
    output_root: str | Path = "research/papers",
) -> PlannerAblationEvidenceBundle:
    output_root = Path(output_root)
    methodology_root = output_root / "planner_methodology"
    figures_root = output_root / "planner_figs"
    tables_root = output_root / "planner_tables"

    methodology_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    experiment_budget = budget or ExperimentBudget(max_simulations=6, max_candidates_per_step=3)
    suite = suite_runner(
        steps=steps,
        repeat_runs=repeat_runs,
        comparison_profile="planner_ablation",
        budget=experiment_budget,
        task_id=f"benchmark-{task_slug}-planner-ablation",
        backend_preference=backend_preference,
        fidelity_level=fidelity_level,
        export_directory=methodology_root,
        force_full_steps=True,
    )
    return build_planner_ablation_evidence_bundle(
        suite,
        figures_dir=figures_root,
        tables_dir=tables_root,
        json_output_path=output_root / "planner_evidence_bundle.json",
    )
