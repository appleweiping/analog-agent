"""Frozen bandgap v1 experiment and acceptance entry points."""

from __future__ import annotations

import json
from pathlib import Path

from apps.orchestrator.job_runner import run_full_system_acceptance
from libs.eval.benchmark_protocol import benchmark_modes_for_profile, default_benchmark_budget
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
from libs.vertical_slices.bandgap_spec import (
    build_bandgap_v1_design_task,
    load_bandgap_v1_config,
)
from libs.vertical_slices.planner_evidence import run_vertical_slice_planner_evidence
from libs.vertical_slices.world_model_evidence import run_vertical_slice_world_model_evidence


def run_bandgap_acceptance(
    *,
    max_steps: int = 3,
    backend_preference: str | None = None,
    default_fidelity: str | None = None,
    task_id: str = "bandgap-v1-acceptance",
) -> SystemAcceptanceResult:
    """Run the frozen bandgap v1 end-to-end acceptance path."""

    config = load_bandgap_v1_config()
    return run_full_system_acceptance(
        AcceptanceTaskConfig(
            design_task=build_bandgap_v1_design_task(task_id=task_id),
            max_steps=max_steps,
            default_fidelity=default_fidelity or config.defaults.fidelity_policy.default_fidelity,
            backend_preference=backend_preference or config.defaults.backend_preference,
            escalation_reason=f"{config.version}:bandgap_acceptance",
        )
    )


def run_bandgap_experiment_suite(
    *,
    steps: int = 3,
    repeat_runs: int = 5,
    budget: ExperimentBudget | None = None,
    backend_preference: str | None = None,
    fidelity_level: str | None = None,
    comparison_profile: str = "baseline",
    modes: list[str] | None = None,
    export_directory: str | Path | None = None,
    task_id: str = "benchmark-bandgap-v1",
    force_full_steps: bool = False,
) -> ExperimentSuiteResult:
    """Run the frozen bandgap v1 experiment suite and optionally export stats."""

    config = load_bandgap_v1_config()
    selected_modes = modes
    if selected_modes is None:
        selected_modes = benchmark_modes_for_profile(comparison_profile)
    suite = run_experiment_suite(
        build_bandgap_v1_design_task(task_id=task_id),
        modes=selected_modes,
        budget=budget or default_benchmark_budget(),
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
        export_stats_json(suite, output_root / "bandgap_stats_summary.json")
        export_stats_csv(suite, output_root / "bandgap_stats_summary.csv")
        if suite.comparison is not None:
            (output_root / "bandgap_method_comparison.json").write_text(
                json.dumps(suite.comparison.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
    return suite


def run_bandgap_world_model_evidence(
    *,
    steps: int = 3,
    repeat_runs: int = 5,
    budget: ExperimentBudget | None = None,
    backend_preference: str | None = None,
    fidelity_level: str | None = None,
    output_root: str | Path = "research/papers/bandgap_v1",
) -> WorldModelEvidenceBundle:
    """Generate paper-facing world-model evidence for bandgap_v1."""

    config = load_bandgap_v1_config()
    return run_vertical_slice_world_model_evidence(
        task_slug="bandgap-v1",
        suite_runner=run_bandgap_experiment_suite,
        measurement_targets=list(config.measurement_targets),
        steps=steps,
        repeat_runs=repeat_runs,
        budget=budget,
        backend_preference=backend_preference or config.defaults.backend_preference,
        fidelity_level=fidelity_level or config.defaults.fidelity_policy.promoted_fidelity,
        output_root=output_root,
    )


def run_bandgap_planner_evidence(
    *,
    steps: int = 3,
    repeat_runs: int = 5,
    budget: ExperimentBudget | None = None,
    backend_preference: str | None = None,
    fidelity_level: str | None = None,
    output_root: str | Path = "research/papers/bandgap_v1",
) -> PlannerAblationEvidenceBundle:
    config = load_bandgap_v1_config()
    return run_vertical_slice_planner_evidence(
        task_slug="bandgap-v1",
        suite_runner=run_bandgap_experiment_suite,
        steps=steps,
        repeat_runs=repeat_runs,
        budget=budget,
        backend_preference=backend_preference or config.defaults.backend_preference,
        fidelity_level=fidelity_level or config.defaults.fidelity_policy.promoted_fidelity,
        output_root=output_root,
    )


def run_bandgap_memory_ablation_suite(
    *,
    episodes: int = 5,
    max_steps: int = 3,
    backend_preference: str | None = None,
    fidelity_level: str | None = None,
) -> MemoryAblationSuiteResult:
    """Run repeated-episode memory ablation on the frozen bandgap v1 path."""

    config = load_bandgap_v1_config()
    return run_repeated_episode_memory_ablation(
        task_slug="bandgap-v1",
        task_builder=build_bandgap_v1_design_task,
        episodes=episodes,
        max_steps=max_steps,
        fidelity_level=fidelity_level or config.defaults.fidelity_policy.promoted_fidelity,
        backend_preference=backend_preference or config.defaults.backend_preference,
    )


def run_bandgap_memory_evidence(
    *,
    episodes: int = 5,
    max_steps: int = 3,
    backend_preference: str | None = None,
    fidelity_level: str | None = None,
    output_root: str | Path = "research/papers/bandgap_v1",
) -> MemoryAblationEvidenceBundle:
    """Generate repeated-episode memory evidence bundle for bandgap_v1."""

    suite = run_bandgap_memory_ablation_suite(
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
