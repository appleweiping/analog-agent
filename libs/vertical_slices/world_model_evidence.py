"""Shared world-model evidence runner for frozen benchmark vertical slices."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from libs.eval.paper_evidence import build_world_model_evidence_bundle
from libs.schema.experiment import ExperimentBudget, ExperimentSuiteResult
from libs.schema.paper_evidence import WorldModelEvidenceBundle

SuiteRunner = Callable[..., ExperimentSuiteResult]


def run_vertical_slice_world_model_evidence(
    *,
    task_slug: str,
    suite_runner: SuiteRunner,
    measurement_targets: list[str] | None = None,
    steps: int = 3,
    repeat_runs: int = 5,
    budget: ExperimentBudget | None = None,
    backend_preference: str = "ngspice",
    fidelity_level: str = "focused_truth",
    output_root: str | Path = "research/papers",
) -> WorldModelEvidenceBundle:
    """Generate aligned methodology and baseline evidence for one frozen vertical slice."""

    output_root = Path(output_root)
    methodology_root = output_root / "methodology"
    baseline_root = output_root / "baseline"
    figures_root = output_root / "figs"
    tables_root = output_root / "tables"

    methodology_root.mkdir(parents=True, exist_ok=True)
    baseline_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    experiment_budget = budget or ExperimentBudget(max_simulations=6, max_candidates_per_step=3)

    methodology_suite = suite_runner(
        steps=steps,
        repeat_runs=repeat_runs,
        comparison_profile="methodology",
        budget=experiment_budget,
        task_id=f"benchmark-{task_slug}-world-model-methodology",
        backend_preference=backend_preference,
        fidelity_level=fidelity_level,
        export_directory=methodology_root,
        force_full_steps=True,
    )
    baseline_suite = suite_runner(
        steps=steps,
        repeat_runs=repeat_runs,
        comparison_profile="baseline",
        modes=["full_simulation_baseline", "no_world_model_baseline", "full_system"],
        budget=experiment_budget,
        task_id=f"benchmark-{task_slug}-world-model-baseline",
        backend_preference=backend_preference,
        fidelity_level=fidelity_level,
        export_directory=baseline_root,
    )
    return build_world_model_evidence_bundle(
        methodology_suite,
        baseline_suite=baseline_suite,
        figures_dir=figures_root,
        tables_dir=tables_root,
        json_output_path=output_root / "world_model_evidence_bundle.json",
        core_gap_metrics=measurement_targets,
    )
