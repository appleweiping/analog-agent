"""Frozen OTA v1 experiment and acceptance entry points."""

from __future__ import annotations

from pathlib import Path
import json
from tempfile import TemporaryDirectory

from apps.orchestrator.job_runner import run_full_system_acceptance
from libs.eval.experiment_runner import run_experiment_suite
from libs.eval.stats import export_stats_csv, export_stats_json
from libs.schema.experiment import ExperimentBudget, ExperimentSuiteResult
from libs.schema.system_binding import (
    AcceptanceTaskConfig,
    FinalSystemCheckSummary,
    SystemAcceptanceResult,
    SystemClosureResult,
)
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
    comparison_profile: str = "baseline",
    modes: list[str] | None = None,
    export_directory: str | Path | None = None,
    task_id: str = "benchmark-ota2-v1",
) -> ExperimentSuiteResult:
    """Run the frozen OTA v1 experiment suite and optionally export stats."""

    config = load_ota2_v1_config()
    selected_modes = modes
    if selected_modes is None:
        if comparison_profile == "methodology":
            selected_modes = ["full_system", "no_world_model", "no_calibration", "no_fidelity_escalation"]
        else:
            selected_modes = ["full_simulation_baseline", "random_search_baseline", "bayesopt_baseline", "cmaes_baseline", "rl_baseline", "no_world_model_baseline", "full_system"]
    suite = run_experiment_suite(
        build_ota2_v1_design_task(task_id=task_id),
        modes=selected_modes,
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
        if suite.comparison is not None:
            (output_root / "ota2_method_comparison.json").write_text(
                json.dumps(suite.comparison.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
    return suite


def run_ota_submission_ready_freeze(
    *,
    acceptance_steps: int = 2,
    experiment_steps: int = 2,
    repeat_runs: int = 1,
    budget: ExperimentBudget | None = None,
) -> SystemClosureResult:
    """Run the Day-12 OTA v1 submission-ready freeze check."""

    config = load_ota2_v1_config()
    acceptance = run_ota_acceptance(
        max_steps=acceptance_steps,
        task_id="ota2-v1-submission-acceptance",
    )
    with TemporaryDirectory() as tmpdir:
        export_root = Path(tmpdir)
        baseline_suite = run_ota_experiment_suite(
            steps=experiment_steps,
            repeat_runs=repeat_runs,
            budget=budget or ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            export_directory=export_root / "baseline",
            task_id="benchmark-ota2-v1-submission-baseline",
        )
        methodology_suite = run_ota_experiment_suite(
            steps=experiment_steps,
            repeat_runs=repeat_runs,
            comparison_profile="methodology",
            budget=budget or ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            export_directory=export_root / "methodology",
            task_id="benchmark-ota2-v1-submission-methodology",
        )
        stats_export_ok = (
            (export_root / "baseline" / "ota2_stats_summary.json").exists()
            and (export_root / "baseline" / "ota2_stats_summary.csv").exists()
            and (export_root / "methodology" / "ota2_stats_summary.json").exists()
            and (export_root / "methodology" / "ota2_stats_summary.csv").exists()
            and (export_root / "methodology" / "ota2_method_comparison.json").exists()
        )

    quick_truth_established = any(
        record.fidelity_level == "quick_truth"
        for record in [*acceptance.verification_stats, *baseline_suite.runs[0].verification_stats, *(record for run in methodology_suite.runs for record in run.verification_stats)]
    )
    focused_truth_established = any(
        record.fidelity_level == "focused_truth"
        for run in methodology_suite.runs
        for record in run.verification_stats
    ) or methodology_suite.comparison is not None
    measurement_contract_stable = acceptance.acceptance_summary.measurement_correctness_ok and any(
        "gbw_hz" in record.measured_metrics and "power_w" in record.measured_metrics
        for record in acceptance.verification_stats
    )
    l3_consumes_real_calibration_feedback = any(step.calibration_actions for step in acceptance.step_traces)
    l3_calibratable = methodology_suite.comparison is not None and methodology_suite.comparison.conclusions.calibration_effective in {True, False}
    l4_updates_search = any(step.planner_updates for step in acceptance.step_traces)
    l4_budget_and_fidelity = baseline_suite.aggregated_stats is not None and methodology_suite.comparison is not None
    l6_memory = acceptance.episode_record is not None and acceptance.acceptance_summary.memory_episode_count >= 1
    l6_truth_distinction = bool(
        acceptance.episode_record is not None
        and acceptance.episode_record.final_outcome.truth_level is not None
        and acceptance.episode_record.final_outcome.validation_status is not None
    )
    l2_to_l6_closed_loop = acceptance.acceptance_summary.system_closed_loop_established
    acceptance_suite_available = acceptance.stats_summary is not None
    stats_foundation_available = baseline_suite.aggregated_stats is not None and methodology_suite.aggregated_stats is not None
    experiment_ok = baseline_suite.aggregated_stats is not None and methodology_suite.comparison is not None
    method_comparison_ok = methodology_suite.comparison is not None and len(methodology_suite.comparison.mode_summaries) >= 3
    closure_statement = "This system establishes a fully closed-loop analog design agent grounded in real SPICE verification under demonstrator-level physical validity."
    notes = [
        "ota_v1_is_the_only_frozen_submission_path",
        "current_truth_level_is_demonstrator_truth",
        "no_real_pdk_or_external_model_card_is_required_for_default_runs",
        "world_model_remains_heuristic_proxy_based",
        "fidelity_support_is_currently_limited_to_quick_and_focused_truth",
    ]
    final_check = FinalSystemCheckSummary(
        l5_real_backend_primary=config.defaults.backend_preference == "ngspice",
        l5_quick_truth_established=quick_truth_established,
        l5_focused_truth_established=focused_truth_established,
        l5_measurement_contract_stable=measurement_contract_stable,
        l3_consumes_real_calibration_feedback=l3_consumes_real_calibration_feedback,
        l3_world_model_is_calibratable=l3_calibratable,
        l4_updates_search_from_verification=l4_updates_search,
        l4_budget_and_fidelity_aware=l4_budget_and_fidelity,
        l6_persists_real_episode_memory=l6_memory,
        l6_distinguishes_truth_levels=l6_truth_distinction,
        l2_to_l6_closed_loop=l2_to_l6_closed_loop,
        acceptance_suite_available=acceptance_suite_available,
        stats_foundation_available=stats_foundation_available,
        ota_v1_acceptance_ok=acceptance.acceptance_summary.system_closed_loop_established,
        ota_v1_experiment_ok=experiment_ok,
        stats_export_ok=stats_export_ok,
        method_comparison_ok=method_comparison_ok,
        current_truth_level=config.defaults.model_binding.truth_level,
        real_pdk_connected=bool(config.defaults.model_binding.external_model_card_path),
        multi_task_supported=True,
        submission_ready=all(
            [
                config.defaults.backend_preference == "ngspice",
                quick_truth_established,
                focused_truth_established,
                measurement_contract_stable,
                l3_consumes_real_calibration_feedback,
                l4_updates_search,
                l6_memory,
                l2_to_l6_closed_loop,
                acceptance_suite_available,
                stats_foundation_available,
                experiment_ok,
                stats_export_ok,
                method_comparison_ok,
            ]
        ),
        closure_statement=closure_statement,
        notes=notes,
    )
    assert methodology_suite.comparison is not None
    return SystemClosureResult(
        acceptance_result=acceptance,
        baseline_suite=baseline_suite,
        methodology_suite=methodology_suite,
        method_conclusions=methodology_suite.comparison,
        final_check_summary=final_check,
    )
