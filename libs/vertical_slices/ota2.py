"""Frozen OTA v1 experiment and acceptance entry points."""

from __future__ import annotations

from pathlib import Path

from apps.orchestrator.job_runner import run_full_system_acceptance
from libs.eval.experiment_runner import run_experiment_suite
from libs.eval.stats import export_stats_csv, export_stats_json
from libs.schema.experiment import ExperimentBudget, ExperimentSuiteResult
from libs.schema.system_binding import AcceptanceTaskConfig, SystemAcceptanceResult
from libs.vertical_slices.ota2_spec import build_ota2_v1_design_task, load_ota2_v1_config


def run_ota_acceptance(
    *,
    max_steps: int = 3,
    backend_preference: str | None = None,
    default_fidelity: str | None = None,
    task_id: str = "ota2-v1-acceptance",
) -> SystemAcceptanceResult:
    """Run the frozen OTA v1 end-to-end acceptance path."""

    config = load_ota2_v1_config()
    return run_full_system_acceptance(
        AcceptanceTaskConfig(
            design_task=build_ota2_v1_design_task(task_id=task_id),
            max_steps=max_steps,
            default_fidelity=default_fidelity or config.defaults.fidelity_policy.default_fidelity,
            backend_preference=backend_preference or config.defaults.backend_preference,
            escalation_reason=f"{config.version}:ota_acceptance",
        )
    )


def run_ota_experiment_suite(
    *,
    steps: int = 3,
    repeat_runs: int = 5,
    budget: ExperimentBudget | None = None,
    backend_preference: str | None = None,
    fidelity_level: str | None = None,
    export_directory: str | Path | None = None,
    task_id: str = "benchmark-ota2-v1",
) -> ExperimentSuiteResult:
    """Run the frozen OTA v1 experiment suite and optionally export stats."""

    config = load_ota2_v1_config()
    suite = run_experiment_suite(
        build_ota2_v1_design_task(task_id=task_id),
        modes=["full_simulation_baseline", "no_world_model_baseline", "full_system"],
        budget=budget or ExperimentBudget(max_simulations=6, max_candidates_per_step=3),
        steps=steps,
        repeat_runs=repeat_runs,
        fidelity_level=fidelity_level or config.defaults.fidelity_policy.promoted_fidelity,
        backend_preference=backend_preference or config.defaults.backend_preference,
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
        export_stats_json(suite, output_root / "ota2_stats_summary.json")
        export_stats_csv(suite, output_root / "ota2_stats_summary.csv")
    return suite
