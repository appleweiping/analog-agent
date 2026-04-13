"""Frozen LDO v1 experiment and acceptance entry points."""

from __future__ import annotations

import json
from pathlib import Path

from apps.orchestrator.job_runner import run_full_system_acceptance
from libs.eval.memory_evidence import (
    build_memory_ablation_evidence_bundle,
    run_repeated_episode_memory_ablation,
)
from libs.eval.experiment_runner import run_experiment_suite
from libs.eval.stats import export_stats_csv, export_stats_json
from libs.schema.experiment import ExperimentBudget, ExperimentSuiteResult
from libs.schema.memory_evidence import MemoryAblationEvidenceBundle, MemoryAblationSuiteResult
from libs.schema.paper_evidence import PlannerAblationEvidenceBundle, WorldModelEvidenceBundle
from libs.schema.system_binding import AcceptanceTaskConfig, SystemAcceptanceResult
from libs.vertical_slices.ldo_spec import build_ldo_v1_design_task, load_ldo_v1_config
from libs.vertical_slices.planner_evidence import run_vertical_slice_planner_evidence
from libs.vertical_slices.world_model_evidence import run_vertical_slice_world_model_evidence


def run_ldo_acceptance(
    *,
    max_steps: int = 3,
    backend_preference: str | None = None,
    default_fidelity: str | None = None,
    task_id: str = "ldo-v1-acceptance",
) -> SystemAcceptanceResult:
    """Run the frozen LDO v1 end-to-end acceptance path."""

    config = load_ldo_v1_config()
    return run_full_system_acceptance(
        AcceptanceTaskConfig(
            design_task=build_ldo_v1_design_task(task_id=task_id),
            max_steps=max_steps,
            default_fidelity=default_fidelity or config.defaults.fidelity_policy.default_fidelity,
            backend_preference=backend_preference or config.defaults.backend_preference,
            escalation_reason=f"{config.version}:ldo_acceptance",
        )
    )


def run_ldo_experiment_suite(
    *,
    steps: int = 3,
    repeat_runs: int = 5,
    budget: ExperimentBudget | None = None,
    backend_preference: str | None = None,
    fidelity_level: str | None = None,
    comparison_profile: str = "baseline",
    modes: list[str] | None = None,
    export_directory: str | Path | None = None,
    task_id: str = "benchmark-ldo-v1",
    force_full_steps: bool = False,
) -> ExperimentSuiteResult:
    """Run the frozen LDO v1 experiment suite and optionally export stats."""

    config = load_ldo_v1_config()
    selected_modes = modes
    if selected_modes is None:
        if comparison_profile == "methodology":
            selected_modes = ["full_system", "no_world_model", "no_calibration", "no_fidelity_escalation"]
        elif comparison_profile == "planner_ablation":
            selected_modes = [
                "full_system",
                "top_k_baseline",
                "no_fidelity_escalation",
                "no_phase_updates",
                "no_calibration_replanning",
                "no_rollout_planning",
            ]
        else:
            selected_modes = ["full_simulation_baseline", "top_k_baseline", "random_search_baseline", "bayesopt_baseline", "cmaes_baseline", "rl_baseline", "no_world_model_baseline", "full_system"]
    suite = run_experiment_suite(
        build_ldo_v1_design_task(task_id=task_id),
        modes=selected_modes,
        budget=budget or ExperimentBudget(max_simulations=6, max_candidates_per_step=3),
        steps=steps,
        repeat_runs=repeat_runs,
        fidelity_level=fidelity_level or config.defaults.fidelity_policy.promoted_fidelity,
        backend_preference=backend_preference or config.defaults.backend_preference,
        force_full_steps=force_full_steps,
    )
    if suite.aggregated_stats is not None and task_id.startswith("benchmark-"):
        suite = suite.model_copy(
            update={
                "aggregated_stats": suite.aggregated_stats.model_copy(
                    update={"aggregation_scope": "benchmark_suite"}
                )
            }
        )
    if export_directory is not None:
        output_root = Path(export_directory)
        output_root.mkdir(parents=True, exist_ok=True)
        export_stats_json(suite, output_root / "ldo_stats_summary.json")
        export_stats_csv(suite, output_root / "ldo_stats_summary.csv")
        if suite.comparison is not None:
            (output_root / "ldo_method_comparison.json").write_text(
                json.dumps(suite.comparison.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
    return suite


def run_ldo_world_model_evidence(
    *,
    steps: int = 3,
    repeat_runs: int = 5,
    budget: ExperimentBudget | None = None,
    backend_preference: str | None = None,
    fidelity_level: str | None = None,
    output_root: str | Path = "research/papers/ldo_v1",
) -> WorldModelEvidenceBundle:
    """Generate paper-facing world-model evidence for ldo_v1."""

    config = load_ldo_v1_config()
    return run_vertical_slice_world_model_evidence(
        task_slug="ldo-v1",
        suite_runner=run_ldo_experiment_suite,
        measurement_targets=list(config.measurement_targets),
        steps=steps,
        repeat_runs=repeat_runs,
        budget=budget,
        backend_preference=backend_preference or config.defaults.backend_preference,
        fidelity_level=fidelity_level or config.defaults.fidelity_policy.promoted_fidelity,
        output_root=output_root,
    )


def run_ldo_planner_evidence(
    *,
    steps: int = 3,
    repeat_runs: int = 5,
    budget: ExperimentBudget | None = None,
    backend_preference: str | None = None,
    fidelity_level: str | None = None,
    output_root: str | Path = "research/papers/ldo_v1",
) -> PlannerAblationEvidenceBundle:
    config = load_ldo_v1_config()
    return run_vertical_slice_planner_evidence(
        task_slug="ldo-v1",
        suite_runner=run_ldo_experiment_suite,
        steps=steps,
        repeat_runs=repeat_runs,
        budget=budget,
        backend_preference=backend_preference or config.defaults.backend_preference,
        fidelity_level=fidelity_level or config.defaults.fidelity_policy.promoted_fidelity,
        output_root=output_root,
    )


def run_ldo_memory_ablation_suite(
    *,
    episodes: int = 5,
    max_steps: int = 3,
    backend_preference: str | None = None,
    fidelity_level: str | None = None,
) -> MemoryAblationSuiteResult:
    """Run repeated-episode memory ablation on the frozen ldo v1 path."""

    config = load_ldo_v1_config()
    return run_repeated_episode_memory_ablation(
        task_slug="ldo-v1",
        task_builder=build_ldo_v1_design_task,
        episodes=episodes,
        max_steps=max_steps,
        fidelity_level=fidelity_level or config.defaults.fidelity_policy.promoted_fidelity,
        backend_preference=backend_preference or config.defaults.backend_preference,
    )


def run_ldo_memory_evidence(
    *,
    episodes: int = 5,
    max_steps: int = 3,
    backend_preference: str | None = None,
    fidelity_level: str | None = None,
    output_root: str | Path = "research/papers/ldo_v1",
) -> MemoryAblationEvidenceBundle:
    """Generate repeated-episode memory evidence bundle for ldo_v1."""

    suite = run_ldo_memory_ablation_suite(
        episodes=episodes,
        max_steps=max_steps,
        backend_preference=backend_preference,
        fidelity_level=fidelity_level,
    )
    root = Path(output_root)
    return build_memory_ablation_evidence_bundle(
        suite,
        figures_dir=root / "memory_figs",
        tables_dir=root / "memory_tables",
        json_output_path=root / "memory_evidence_bundle.json",
    )
