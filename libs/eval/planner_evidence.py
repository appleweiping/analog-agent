"""Paper-facing figure and table generation for planner ablation evidence."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from libs.eval.paper_evidence import _mean, _write_svg_bar_chart, _write_table_csv, _write_table_markdown
from libs.schema.experiment import ExperimentSuiteResult
from libs.schema.paper_evidence import (
    FigureSeries,
    FigureSpec,
    PlannerAblationEvidenceBundle,
    PlannerPaperLayoutBundle,
    PlannerAblationSummary,
    TableColumn,
    TableRow,
    TableSpec,
)


def _mode_runs(suite: ExperimentSuiteResult, mode: str):
    return [run for run in suite.runs if run.mode == mode]


def _mode_summary_map(suite: ExperimentSuiteResult):
    return {summary.mode: summary for summary in suite.comparison.mode_summaries} if suite.comparison else {}


def _aggregate_summary_map(suite: ExperimentSuiteResult):
    return {summary.mode: summary for summary in suite.summaries}


def _phase_change_rate(runs) -> float:
    values = [1.0 if record.phase_changed else 0.0 for run in runs for record in run.structured_log]
    return _mean(values)


def _calibration_replanning_rate(runs) -> float:
    values = [1.0 if record.calibration_required_after_step else 0.0 for run in runs for record in run.structured_log]
    return _mean(values)


def _rollout_guidance_rate(runs) -> float:
    values = [1.0 if record.rollout_guidance_applied else 0.0 for run in runs for record in run.structured_log]
    return _mean(values)


def _average_rollout_guidance_value(runs) -> float:
    values = [record.rollout_guidance_value for run in runs for record in run.structured_log if record.rollout_guidance_applied]
    return _mean(values)


def _mode_log_rate(runs, attr: str) -> float:
    values = [1.0 if getattr(record, attr) else 0.0 for run in runs for record in run.structured_log]
    return _mean(values)


def _aggregate_metric(aggregate_map, mode: str, attr: str) -> float:
    summary = aggregate_map.get(mode)
    return float(getattr(summary, attr, 0.0)) if summary is not None else 0.0


def _summary_metric(summary_map, mode: str, attr: str) -> float:
    summary = summary_map.get(mode)
    return float(getattr(summary, attr, 0.0)) if summary is not None else 0.0


def _comparison_modes(*modes: str, available: set[str]) -> list[str]:
    return [mode for mode in modes if mode in available]


def _failure_distribution(summary_map, mode: str) -> dict[str, int]:
    summary = summary_map.get(mode)
    if summary is None:
        return {}
    return {key: int(value) for key, value in summary.failure_type_distribution.items()}


def _failure_pressure(summary_map, mode: str) -> float:
    summary = summary_map.get(mode)
    if summary is None or summary.run_count <= 0:
        return 0.0
    total = sum(_failure_distribution(summary_map, mode).values())
    return round(total / max(1, int(summary.run_count)), 6)


def _dominant_failure_mode(summary_map, mode: str) -> str:
    distribution = _failure_distribution(summary_map, mode)
    if not distribution:
        return "none"
    return sorted(distribution.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _dominant_failure_share(summary_map, mode: str) -> float:
    distribution = _failure_distribution(summary_map, mode)
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    dominant = max(distribution.values())
    return round(dominant / total, 6)


def _copy_file(source: str | Path, target: str | Path) -> str:
    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return str(target_path)


def build_planner_ablation_evidence_bundle(
    suite: ExperimentSuiteResult,
    *,
    figures_dir: str | Path,
    tables_dir: str | Path,
    json_output_path: str | Path,
) -> PlannerAblationEvidenceBundle:
    if suite.comparison is None:
        raise ValueError("planner evidence requires a comparison suite")

    figures_root = Path(figures_dir)
    tables_root = Path(tables_dir)
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    modes = ["full_system", "top_k_baseline", "no_fidelity_escalation", "no_phase_updates", "no_calibration_replanning", "no_rollout_planning"]
    summary_map = _mode_summary_map(suite)
    aggregate_map = _aggregate_summary_map(suite)
    present_modes = [mode for mode in modes if mode in summary_map]
    present_mode_set = set(present_modes)

    simulation_figure = FigureSpec(
        figure_id="fig_planner_simulation_calls",
        title="Planner Ablation: Real Simulation Calls",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Average Real Simulation Calls",
        series=[
            FigureSeries(label=mode, x_values=[float(index)], y_values=[summary_map[mode].simulation_call_count], color=color)
            for index, (mode, color) in enumerate(
                zip(
                    present_modes,
                    ["#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c"],
                )
            )
        ],
        caption="Planner-side ablation on real SPICE call count. Lower is better.",
        output_path=str(figures_root / "planner_simulation_calls.svg"),
    )

    feasible_figure = FigureSpec(
        figure_id="fig_planner_feasible_hit_rate",
        title="Planner Ablation: Feasible Hit Rate",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Feasible Hit Rate",
        series=[
            FigureSeries(label=mode, x_values=[float(index)], y_values=[summary_map[mode].feasible_hit_rate], color=color)
            for index, (mode, color) in enumerate(
                zip(
                    present_modes,
                    ["#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c"],
                )
            )
        ],
        caption="Planner-side ablation on feasible-solution hit rate. Higher is better.",
        output_path=str(figures_root / "planner_feasible_hit_rate.svg"),
    )

    fidelity_figure = FigureSpec(
        figure_id="fig_planner_focused_truth_ratio",
        title="Planner Ablation: Focused-Truth Usage",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Focused Truth Ratio",
        series=[
            FigureSeries(label=mode, x_values=[float(index)], y_values=[summary_map[mode].focused_truth_ratio], color=color)
            for index, (mode, color) in enumerate(
                zip(
                    present_modes,
                    ["#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c"],
                )
            )
        ],
        caption="Planner-side ablation on focused-truth utilization.",
        output_path=str(figures_root / "planner_focused_truth_ratio.svg"),
    )

    efficiency_figure = FigureSpec(
        figure_id="fig_planner_efficiency_score",
        title="Planner Ablation: Efficiency Score",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Feasible / Real Simulation Call",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(index)],
                y_values=[aggregate_map[mode].average_efficiency_score if mode in aggregate_map else 0.0],
                color=color,
            )
            for index, (mode, color) in enumerate(
                zip(
                    present_modes,
                    ["#1f77b4", "#8c564b", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c"],
                )
            )
        ],
        caption="Efficiency score under planner ablation. Higher is better.",
        output_path=str(figures_root / "planner_efficiency_score.svg"),
    )

    phase_figure = FigureSpec(
        figure_id="fig_planner_phase_change_rate",
        title="Planner Ablation: Phase Change Rate",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Phase Change Rate",
        series=[
            FigureSeries(label=mode, x_values=[float(index)], y_values=[_phase_change_rate(_mode_runs(suite, mode))], color=color)
            for index, (mode, color) in enumerate(
                zip(
                    present_modes,
                    ["#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c"],
                )
            )
        ],
        caption="Fraction of logged steps that trigger a formal phase transition.",
        output_path=str(figures_root / "planner_phase_change_rate.svg"),
    )

    rollout_figure = FigureSpec(
        figure_id="fig_planner_rollout_guidance_rate",
        title="Planner Ablation: Rollout Guidance Rate",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Rollout Guidance Rate",
        series=[
            FigureSeries(label=mode, x_values=[float(index)], y_values=[_rollout_guidance_rate(_mode_runs(suite, mode))], color=color)
            for index, (mode, color) in enumerate(
                zip(
                    present_modes,
                    ["#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c"],
                )
            )
        ],
        caption="Fraction of logged steps that receive explicit rollout guidance.",
        output_path=str(figures_root / "planner_rollout_guidance_rate.svg"),
    )

    calibration_replanning_figure = FigureSpec(
        figure_id="fig_planner_calibration_replanning_rate",
        title="Planner Ablation: Calibration-Replanning Rate",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Calibration Replanning Rate",
        series=[
            FigureSeries(label=mode, x_values=[float(index)], y_values=[_calibration_replanning_rate(_mode_runs(suite, mode))], color=color)
            for index, (mode, color) in enumerate(
                zip(
                    present_modes,
                    ["#1f77b4", "#8c564b", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c"],
                )
            )
        ],
        caption="Fraction of logged steps that still require calibration-driven replanning.",
        output_path=str(figures_root / "planner_calibration_replanning_rate.svg"),
    )

    topk_modes = _comparison_modes("full_system", "top_k_baseline", available=present_mode_set)
    fidelity_modes = _comparison_modes("full_system", "no_fidelity_escalation", available=present_mode_set)
    phase_modes = _comparison_modes("full_system", "no_phase_updates", available=present_mode_set)
    calibration_modes = _comparison_modes("full_system", "no_calibration_replanning", available=present_mode_set)
    rollout_modes = _comparison_modes("full_system", "no_rollout_planning", available=present_mode_set)

    topk_utility_figure = FigureSpec(
        figure_id="fig_planner_topk_efficiency",
        title="Planner vs Top-K: Efficiency",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Efficiency Score",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(index)],
                y_values=[_aggregate_metric(aggregate_map, mode, "average_efficiency_score")],
                color=color,
            )
            for index, (mode, color) in enumerate(zip(topk_modes, ["#1f77b4", "#9467bd"]))
        ],
        caption="Direct planner-versus-top-k efficiency comparison for the current slice.",
        output_path=str(figures_root / "planner_topk_efficiency.svg"),
    )

    fidelity_tradeoff_figure = FigureSpec(
        figure_id="fig_planner_fidelity_tradeoff",
        title="Planner Fidelity Escalation: Real Simulation Calls",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Average Real Simulation Calls",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(index)],
                y_values=[_summary_metric(summary_map, mode, "simulation_call_count")],
                color=color,
            )
            for index, (mode, color) in enumerate(zip(fidelity_modes, ["#1f77b4", "#ff7f0e"]))
        ],
        caption="Focused-truth escalation should reduce real simulation calls without hurting search quality.",
        output_path=str(figures_root / "planner_fidelity_tradeoff.svg"),
    )

    phase_convergence_figure = FigureSpec(
        figure_id="fig_planner_phase_convergence",
        title="Planner Phase Updates: Convergence",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Average Convergence Step",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(index)],
                y_values=[_summary_metric(summary_map, mode, "average_convergence_step")],
                color=color,
            )
            for index, (mode, color) in enumerate(zip(phase_modes, ["#1f77b4", "#d62728"]))
        ],
        caption="Phase-aware updates should converge earlier than the no-phase ablation.",
        output_path=str(figures_root / "planner_phase_convergence.svg"),
    )

    calibration_utility_figure = FigureSpec(
        figure_id="fig_planner_calibration_utility",
        title="Planner Calibration Replanning: Convergence",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Average Convergence Step",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(index)],
                y_values=[_summary_metric(summary_map, mode, "average_convergence_step")],
                color=color,
            )
            for index, (mode, color) in enumerate(zip(calibration_modes, ["#1f77b4", "#2ca02c"]))
        ],
        caption="Calibration-driven replanning is evaluated by convergence speed under the same planner budget.",
        output_path=str(figures_root / "planner_calibration_utility.svg"),
    )

    rollout_claim_audit_figure = FigureSpec(
        figure_id="fig_planner_rollout_claim_audit",
        title="Planner Rollout Audit: Convergence",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Average Convergence Step",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(index)],
                y_values=[_summary_metric(summary_map, mode, "average_convergence_step")],
                color=color,
            )
            for index, (mode, color) in enumerate(zip(rollout_modes, ["#1f77b4", "#8c564b"]))
        ],
        caption="Rollout evidence is kept at the short-horizon guidance level rather than a general MPC-optimality claim.",
        output_path=str(figures_root / "planner_rollout_claim_audit.svg"),
    )

    failure_pressure_figure = FigureSpec(
        figure_id="fig_planner_failure_pressure",
        title="Planner Failure Pressure by Mode",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Failure Events per Run",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(index)],
                y_values=[_failure_pressure(summary_map, mode)],
                color=color,
            )
            for index, (mode, color) in enumerate(
                zip(
                    present_modes,
                    ["#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c", "#8c564b"],
                )
            )
        ],
        caption="Observed failure-event pressure aggregated over planner runs. Lower is better.",
        output_path=str(figures_root / "planner_failure_pressure.svg"),
    )

    efficiency_frontier_figure = FigureSpec(
        figure_id="fig_planner_efficiency_frontier",
        title="Planner Efficiency Frontier",
        chart_type="bar",
        x_label="Planner Mode",
        y_label="Efficiency Score",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(index)],
                y_values=[_aggregate_metric(aggregate_map, mode, "average_efficiency_score")],
                color=color,
            )
            for index, (mode, color) in enumerate(
                zip(
                    present_modes,
                    ["#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c", "#8c564b"],
                )
            )
        ],
        caption="Compact efficiency synthesis across planner ablations used for paper-level figure selection.",
        output_path=str(figures_root / "planner_efficiency_frontier.svg"),
    )

    figures = [
        simulation_figure,
        feasible_figure,
        fidelity_figure,
        efficiency_figure,
        phase_figure,
        rollout_figure,
        calibration_replanning_figure,
        topk_utility_figure,
        fidelity_tradeoff_figure,
        phase_convergence_figure,
        calibration_utility_figure,
        rollout_claim_audit_figure,
        failure_pressure_figure,
        efficiency_frontier_figure,
    ]

    for figure in figures:
        _write_svg_bar_chart(figure)

    comparison_table = TableSpec(
        table_id="tbl_planner_ablation_comparison",
        title="Planner Ablation Comparison",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="simulation_call_count", label="Avg Sim Calls"),
            TableColumn(key="feasible_hit_rate", label="Feasible Hit Rate"),
            TableColumn(key="average_efficiency_score", label="Efficiency Score"),
            TableColumn(key="average_convergence_step", label="Avg Convergence Step"),
            TableColumn(key="focused_truth_ratio", label="Focused Truth Ratio"),
            TableColumn(key="phase_change_rate", label="Phase Change Rate"),
            TableColumn(key="calibration_replanning_rate", label="Calib Replan Rate"),
            TableColumn(key="rollout_guidance_rate", label="Rollout Guidance Rate"),
            TableColumn(key="rollout_guidance_value", label="Avg Rollout Guidance"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": mode,
                    "simulation_call_count": summary_map[mode].simulation_call_count,
                    "feasible_hit_rate": summary_map[mode].feasible_hit_rate,
                    "average_efficiency_score": aggregate_map[mode].average_efficiency_score if mode in aggregate_map else 0.0,
                    "average_convergence_step": summary_map[mode].average_convergence_step,
                    "focused_truth_ratio": summary_map[mode].focused_truth_ratio,
                    "phase_change_rate": _phase_change_rate(_mode_runs(suite, mode)),
                    "calibration_replanning_rate": _calibration_replanning_rate(_mode_runs(suite, mode)),
                    "rollout_guidance_rate": _rollout_guidance_rate(_mode_runs(suite, mode)),
                    "rollout_guidance_value": _average_rollout_guidance_value(_mode_runs(suite, mode)),
                }
            )
            for mode in present_modes
        ],
        caption="Primary planner-ablation table for the current vertical slice.",
        csv_output_path=str(tables_root / "planner_ablation_comparison.csv"),
        markdown_output_path=str(tables_root / "planner_ablation_comparison.md"),
    )

    top_k_reference = aggregate_map.get("top_k_baseline")
    delta_vs_top_k_table = TableSpec(
        table_id="tbl_planner_delta_vs_topk",
        title="Planner Delta Against Simple Top-K Baseline",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="delta_simulation_calls", label="Delta Sim Calls"),
            TableColumn(key="delta_feasible_hit_rate", label="Delta Feasible Hit"),
            TableColumn(key="delta_efficiency_score", label="Delta Efficiency"),
            TableColumn(key="delta_focused_truth_ratio", label="Delta Focused Ratio"),
            TableColumn(key="delta_phase_change_rate", label="Delta Phase Change"),
            TableColumn(key="delta_calibration_replanning_rate", label="Delta Calib Replan"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": mode,
                    "delta_simulation_calls": round(summary_map[mode].simulation_call_count - top_k_reference.average_simulation_call_count, 6) if top_k_reference else 0.0,
                    "delta_feasible_hit_rate": round(summary_map[mode].feasible_hit_rate - (aggregate_map["top_k_baseline"].feasible_hit_rate if "top_k_baseline" in aggregate_map else 0.0), 6) if top_k_reference else 0.0,
                    "delta_efficiency_score": round((aggregate_map[mode].average_efficiency_score if mode in aggregate_map else 0.0) - top_k_reference.average_efficiency_score, 6) if top_k_reference and mode in aggregate_map else 0.0,
                    "delta_focused_truth_ratio": round(summary_map[mode].focused_truth_ratio - summary_map["top_k_baseline"].focused_truth_ratio, 6) if top_k_reference and "top_k_baseline" in summary_map else 0.0,
                    "delta_phase_change_rate": round(_phase_change_rate(_mode_runs(suite, mode)) - _phase_change_rate(_mode_runs(suite, "top_k_baseline")), 6) if top_k_reference else 0.0,
                    "delta_calibration_replanning_rate": round(_calibration_replanning_rate(_mode_runs(suite, mode)) - _calibration_replanning_rate(_mode_runs(suite, "top_k_baseline")), 6) if top_k_reference else 0.0,
                }
            )
            for mode in present_modes
            if mode != "top_k_baseline"
        ],
        caption="Primary paper-facing delta table against the simple top-K planner baseline.",
        csv_output_path=str(tables_root / "planner_delta_vs_topk.csv"),
        markdown_output_path=str(tables_root / "planner_delta_vs_topk.md"),
    )

    step_behavior_table = TableSpec(
        table_id="tbl_planner_step_behavior",
        title="Planner Step Behavior Profile",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="mean_selected_uncertainty", label="Mean Selected Uncertainty"),
            TableColumn(key="mean_selected_confidence", label="Mean Selected Confidence"),
            TableColumn(key="mean_selected_simulation_value", label="Mean Sim Value"),
            TableColumn(key="mean_selected_predicted_feasibility", label="Mean Pred Feasibility"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": mode,
                    "mean_selected_uncertainty": _mean(
                        [record.selected_mean_uncertainty for run in _mode_runs(suite, mode) for record in run.structured_log]
                    ),
                    "mean_selected_confidence": _mean(
                        [record.selected_mean_confidence for run in _mode_runs(suite, mode) for record in run.structured_log]
                    ),
                    "mean_selected_simulation_value": _mean(
                        [record.selected_mean_simulation_value for run in _mode_runs(suite, mode) for record in run.structured_log]
                    ),
                    "mean_selected_predicted_feasibility": _mean(
                        [record.selected_mean_predicted_feasibility for run in _mode_runs(suite, mode) for record in run.structured_log]
                    ),
                }
            )
            for mode in present_modes
        ],
        caption="Stepwise selection behavior across planner-ablation modes.",
        csv_output_path=str(tables_root / "planner_step_behavior.csv"),
        markdown_output_path=str(tables_root / "planner_step_behavior.md"),
    )

    topk_utility_table = TableSpec(
        table_id="tbl_planner_topk_utility",
        title="Planner vs Top-K Utility",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="simulation_call_count", label="Avg Sim Calls"),
            TableColumn(key="feasible_hit_rate", label="Feasible Hit Rate"),
            TableColumn(key="efficiency_score", label="Efficiency Score"),
            TableColumn(key="convergence_step", label="Avg Convergence Step"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": mode,
                    "simulation_call_count": _summary_metric(summary_map, mode, "simulation_call_count"),
                    "feasible_hit_rate": _summary_metric(summary_map, mode, "feasible_hit_rate"),
                    "efficiency_score": _aggregate_metric(aggregate_map, mode, "average_efficiency_score"),
                    "convergence_step": _summary_metric(summary_map, mode, "average_convergence_step"),
                }
            )
            for mode in topk_modes
        ],
        caption="Sharpened direct comparison between the full planner and the simple top-k baseline.",
        csv_output_path=str(tables_root / "planner_topk_utility.csv"),
        markdown_output_path=str(tables_root / "planner_topk_utility.md"),
    )

    fidelity_tradeoff_table = TableSpec(
        table_id="tbl_planner_fidelity_tradeoff",
        title="Planner Fidelity Escalation Tradeoff",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="simulation_call_count", label="Avg Sim Calls"),
            TableColumn(key="focused_truth_ratio", label="Focused Truth Ratio"),
            TableColumn(key="feasible_hit_rate", label="Feasible Hit Rate"),
            TableColumn(key="efficiency_score", label="Efficiency Score"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": mode,
                    "simulation_call_count": _summary_metric(summary_map, mode, "simulation_call_count"),
                    "focused_truth_ratio": _summary_metric(summary_map, mode, "focused_truth_ratio"),
                    "feasible_hit_rate": _summary_metric(summary_map, mode, "feasible_hit_rate"),
                    "efficiency_score": _aggregate_metric(aggregate_map, mode, "average_efficiency_score"),
                }
            )
            for mode in fidelity_modes
        ],
        caption="Focused-truth escalation comparison used for the planner-side cost-quality tradeoff claim.",
        csv_output_path=str(tables_root / "planner_fidelity_tradeoff.csv"),
        markdown_output_path=str(tables_root / "planner_fidelity_tradeoff.md"),
    )

    phase_utility_table = TableSpec(
        table_id="tbl_planner_phase_utility",
        title="Planner Phase-Update Utility",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="convergence_step", label="Avg Convergence Step"),
            TableColumn(key="feasible_hit_rate", label="Feasible Hit Rate"),
            TableColumn(key="phase_change_rate", label="Phase Change Rate"),
            TableColumn(key="efficiency_score", label="Efficiency Score"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": mode,
                    "convergence_step": _summary_metric(summary_map, mode, "average_convergence_step"),
                    "feasible_hit_rate": _summary_metric(summary_map, mode, "feasible_hit_rate"),
                    "phase_change_rate": _phase_change_rate(_mode_runs(suite, mode)),
                    "efficiency_score": _aggregate_metric(aggregate_map, mode, "average_efficiency_score"),
                }
            )
            for mode in phase_modes
        ],
        caption="Formal convergence and phase-transition evidence for the phase-aware planner update claim.",
        csv_output_path=str(tables_root / "planner_phase_utility.csv"),
        markdown_output_path=str(tables_root / "planner_phase_utility.md"),
    )

    calibration_utility_table = TableSpec(
        table_id="tbl_planner_calibration_utility",
        title="Planner Calibration Replanning Utility",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="convergence_step", label="Avg Convergence Step"),
            TableColumn(key="simulation_call_count", label="Avg Sim Calls"),
            TableColumn(key="calibration_replanning_rate", label="Calib Replan Rate"),
            TableColumn(key="calibration_updates", label="Avg Calibration Updates"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": mode,
                    "convergence_step": _summary_metric(summary_map, mode, "average_convergence_step"),
                    "simulation_call_count": _summary_metric(summary_map, mode, "simulation_call_count"),
                    "calibration_replanning_rate": _calibration_replanning_rate(_mode_runs(suite, mode)),
                    "calibration_updates": _summary_metric(summary_map, mode, "average_calibration_update_count"),
                }
            )
            for mode in calibration_modes
        ],
        caption="Calibration-driven replanning comparison used for the planner convergence claim.",
        csv_output_path=str(tables_root / "planner_calibration_utility.csv"),
        markdown_output_path=str(tables_root / "planner_calibration_utility.md"),
    )

    rollout_claim_audit_table = TableSpec(
        table_id="tbl_planner_rollout_claim_audit",
        title="Planner Rollout Claim Audit",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="convergence_step", label="Avg Convergence Step"),
            TableColumn(key="feasible_hit_rate", label="Feasible Hit Rate"),
            TableColumn(key="rollout_guidance_rate", label="Rollout Guidance Rate"),
            TableColumn(key="rollout_guidance_value", label="Avg Rollout Guidance"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": mode,
                    "convergence_step": _summary_metric(summary_map, mode, "average_convergence_step"),
                    "feasible_hit_rate": _summary_metric(summary_map, mode, "feasible_hit_rate"),
                    "rollout_guidance_rate": _rollout_guidance_rate(_mode_runs(suite, mode)),
                    "rollout_guidance_value": _average_rollout_guidance_value(_mode_runs(suite, mode)),
                }
            )
            for mode in rollout_modes
        ],
        caption="Rollout evidence is audited at the observable short-horizon-guidance level to avoid overclaiming a general MPC result.",
        csv_output_path=str(tables_root / "planner_rollout_claim_audit.csv"),
        markdown_output_path=str(tables_root / "planner_rollout_claim_audit.md"),
    )

    failure_mode_table = TableSpec(
        table_id="tbl_planner_failure_mode_summary",
        title="Planner Failure-Mode Synthesis",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="failure_pressure", label="Failure Events per Run"),
            TableColumn(key="dominant_failure_mode", label="Dominant Failure"),
            TableColumn(key="dominant_failure_share", label="Dominant Failure Share"),
            TableColumn(key="feasible_hit_rate", label="Feasible Hit Rate"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": mode,
                    "failure_pressure": _failure_pressure(summary_map, mode),
                    "dominant_failure_mode": _dominant_failure_mode(summary_map, mode),
                    "dominant_failure_share": _dominant_failure_share(summary_map, mode),
                    "feasible_hit_rate": _summary_metric(summary_map, mode, "feasible_hit_rate"),
                }
            )
            for mode in present_modes
        ],
        caption="Failure-mode synthesis for planner ablations, exposing whether improved efficiency simply hides failure pressure.",
        csv_output_path=str(tables_root / "planner_failure_mode_summary.csv"),
        markdown_output_path=str(tables_root / "planner_failure_mode_summary.md"),
    )

    efficiency_synthesis_table = TableSpec(
        table_id="tbl_planner_efficiency_synthesis",
        title="Planner Efficiency Synthesis",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="efficiency_score", label="Efficiency Score"),
            TableColumn(key="simulation_call_count", label="Avg Sim Calls"),
            TableColumn(key="convergence_step", label="Avg Convergence Step"),
            TableColumn(key="focused_truth_ratio", label="Focused Truth Ratio"),
            TableColumn(key="failure_pressure", label="Failure Events per Run"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": mode,
                    "efficiency_score": _aggregate_metric(aggregate_map, mode, "average_efficiency_score"),
                    "simulation_call_count": _summary_metric(summary_map, mode, "simulation_call_count"),
                    "convergence_step": _summary_metric(summary_map, mode, "average_convergence_step"),
                    "focused_truth_ratio": _summary_metric(summary_map, mode, "focused_truth_ratio"),
                    "failure_pressure": _failure_pressure(summary_map, mode),
                }
            )
            for mode in present_modes
        ],
        caption="Compact planner efficiency synthesis balancing simulation calls, convergence, and failure pressure.",
        csv_output_path=str(tables_root / "planner_efficiency_synthesis.csv"),
        markdown_output_path=str(tables_root / "planner_efficiency_synthesis.md"),
    )

    tables = [
        comparison_table,
        delta_vs_top_k_table,
        step_behavior_table,
        topk_utility_table,
        fidelity_tradeoff_table,
        phase_utility_table,
        calibration_utility_table,
        rollout_claim_audit_table,
        failure_mode_table,
        efficiency_synthesis_table,
    ]

    for table in tables:
        _write_table_csv(table)
        _write_table_markdown(table)

    conclusions = suite.comparison.conclusions
    full_summary = summary_map.get("full_system")
    top_k_summary = summary_map.get("top_k_baseline")
    no_fidelity_summary = summary_map.get("no_fidelity_escalation")
    no_phase_summary = summary_map.get("no_phase_updates")
    no_replanning_summary = summary_map.get("no_calibration_replanning")
    no_rollout_summary = summary_map.get("no_rollout_planning")

    planner_reduces_simulations_vs_top_k = top_k_summary is None or (
        full_summary is not None and full_summary.simulation_call_count <= top_k_summary.simulation_call_count
    )
    planner_preserves_or_improves_feasible_hit_rate_vs_top_k = top_k_summary is None or (
        full_summary is not None and full_summary.feasible_hit_rate >= top_k_summary.feasible_hit_rate
    )
    planner_improves_efficiency_vs_top_k = top_k_summary is None or (
        _aggregate_metric(aggregate_map, "full_system", "average_efficiency_score")
        >= _aggregate_metric(aggregate_map, "top_k_baseline", "average_efficiency_score")
    )
    fidelity_escalation_reduces_simulations = no_fidelity_summary is None or (
        full_summary is not None and full_summary.simulation_call_count <= no_fidelity_summary.simulation_call_count
    )
    fidelity_escalation_preserves_or_improves_feasible_hit_rate = no_fidelity_summary is None or (
        full_summary is not None and full_summary.feasible_hit_rate >= no_fidelity_summary.feasible_hit_rate
    )
    phase_updates_improve_convergence = no_phase_summary is None or (
        full_summary is not None and full_summary.average_convergence_step <= no_phase_summary.average_convergence_step
    )
    phase_updates_observable = no_phase_summary is None or (
        _phase_change_rate(_mode_runs(suite, "full_system")) > _phase_change_rate(_mode_runs(suite, "no_phase_updates"))
    )
    calibration_replanning_improves_convergence = no_replanning_summary is None or (
        full_summary is not None and full_summary.average_convergence_step <= no_replanning_summary.average_convergence_step
    )
    calibration_replanning_observable = no_replanning_summary is None or (
        _calibration_replanning_rate(_mode_runs(suite, "full_system"))
        > _calibration_replanning_rate(_mode_runs(suite, "no_calibration_replanning"))
    )
    rollout_guidance_improves_convergence = no_rollout_summary is None or (
        full_summary is not None and full_summary.average_convergence_step <= no_rollout_summary.average_convergence_step
    )
    rollout_guidance_preserves_or_improves_feasible_hit_rate = no_rollout_summary is None or (
        full_summary is not None and full_summary.feasible_hit_rate >= no_rollout_summary.feasible_hit_rate
    )
    rollout_guidance_observable = no_rollout_summary is None or (
        _rollout_guidance_rate(_mode_runs(suite, "full_system"))
        > _rollout_guidance_rate(_mode_runs(suite, "no_rollout_planning"))
    )
    rollout_claim_status = (
        "supported_short_horizon_rollout_guidance"
        if rollout_guidance_observable and rollout_guidance_improves_convergence and rollout_guidance_preserves_or_improves_feasible_hit_rate
        else "observable_but_not_strong_enough_for_broad_mpc_claim"
        if rollout_guidance_observable
        else "do_not_claim_rollout_utility"
    )
    rollout_evidence_real_not_placeholder = (
        "full_system" in present_mode_set
        and "no_rollout_planning" in present_mode_set
        and rollout_guidance_observable
        and _average_rollout_guidance_value(_mode_runs(suite, "full_system")) > 0.0
    )
    full_failure_mode = _dominant_failure_mode(summary_map, "full_system")
    planner_reduces_failure_pressure = "top_k_baseline" not in present_mode_set or (
        _failure_pressure(summary_map, "full_system") <= _failure_pressure(summary_map, "top_k_baseline")
    )
    efficiency_frontier_consistent = all(
        _aggregate_metric(aggregate_map, "full_system", "average_efficiency_score")
        >= _aggregate_metric(aggregate_map, mode, "average_efficiency_score")
        for mode in present_modes
        if mode != "full_system"
    )
    summary = PlannerAblationSummary(
        planner_beats_top_k=conclusions.top_k_baseline_effective,
        planner_reduces_simulations_vs_top_k=planner_reduces_simulations_vs_top_k,
        planner_preserves_or_improves_feasible_hit_rate_vs_top_k=planner_preserves_or_improves_feasible_hit_rate_vs_top_k,
        planner_improves_efficiency_vs_top_k=planner_improves_efficiency_vs_top_k,
        fidelity_escalation_effective=conclusions.fidelity_effective,
        fidelity_escalation_reduces_simulations=fidelity_escalation_reduces_simulations,
        fidelity_escalation_preserves_or_improves_feasible_hit_rate=fidelity_escalation_preserves_or_improves_feasible_hit_rate,
        phase_updates_effective=conclusions.phase_updates_effective,
        phase_updates_improve_convergence=phase_updates_improve_convergence,
        phase_updates_observable=phase_updates_observable,
        calibration_replanning_effective=conclusions.calibration_replanning_effective,
        calibration_replanning_improves_convergence=calibration_replanning_improves_convergence,
        calibration_replanning_observable=calibration_replanning_observable,
        rollout_guidance_effective=conclusions.rollout_effective,
        rollout_guidance_improves_convergence=rollout_guidance_improves_convergence,
        rollout_guidance_preserves_or_improves_feasible_hit_rate=rollout_guidance_preserves_or_improves_feasible_hit_rate,
        rollout_guidance_observable=rollout_guidance_observable,
        rollout_claim_supported_without_mpc_overclaim=rollout_claim_status == "supported_short_horizon_rollout_guidance",
        rollout_claim_limited_to_short_horizon_guidance=True,
        rollout_evidence_real_not_placeholder=rollout_evidence_real_not_placeholder,
        rollout_placeholder_risk=not rollout_evidence_real_not_placeholder,
        rollout_claim_scope="short_horizon_world_model_guidance",
        rollout_claim_status=rollout_claim_status,
        dominant_failure_mode=full_failure_mode,
        planner_reduces_failure_pressure=planner_reduces_failure_pressure,
        failure_synthesis_ready=True,
        efficiency_synthesis_ready=True,
        efficiency_frontier_consistent=efficiency_frontier_consistent,
        notes=[
            *conclusions.conclusion_notes,
            f"planner_vs_topk_delta_efficiency={round(_aggregate_metric(aggregate_map, 'full_system', 'average_efficiency_score') - _aggregate_metric(aggregate_map, 'top_k_baseline', 'average_efficiency_score'), 6)}",
            f"fidelity_delta_sim_calls={round(_summary_metric(summary_map, 'no_fidelity_escalation', 'simulation_call_count') - _summary_metric(summary_map, 'full_system', 'simulation_call_count'), 6)}",
            f"phase_delta_convergence={round(_summary_metric(summary_map, 'no_phase_updates', 'average_convergence_step') - _summary_metric(summary_map, 'full_system', 'average_convergence_step'), 6)}",
            f"calibration_replanning_delta_convergence={round(_summary_metric(summary_map, 'no_calibration_replanning', 'average_convergence_step') - _summary_metric(summary_map, 'full_system', 'average_convergence_step'), 6)}",
            f"rollout_delta_convergence={round(_summary_metric(summary_map, 'no_rollout_planning', 'average_convergence_step') - _summary_metric(summary_map, 'full_system', 'average_convergence_step'), 6)}",
            f"full_system_failure_pressure={_failure_pressure(summary_map, 'full_system')}",
            f"dominant_failure_mode={full_failure_mode}",
            f"rollout_claim_scope=short_horizon_world_model_guidance_only",
            f"modes={','.join(present_modes)}",
        ],
    )

    bundle = PlannerAblationEvidenceBundle(
        task_id=suite.task_id,
        modes=present_modes,
        figures=figures,
        tables=tables,
        summary=summary,
        json_output_path=str(json_output_path),
    )
    Path(json_output_path).write_text(json.dumps(bundle.model_dump(mode="json"), indent=2), encoding="utf-8")
    return bundle


def build_planner_paper_layout_bundle(
    *,
    profile_name: str,
    bundle: PlannerAblationEvidenceBundle,
    output_root: str | Path,
) -> PlannerPaperLayoutBundle:
    """Assemble main/appendix organization for the planner paper section."""

    root = Path(output_root)
    main_figs_root = root / "main_figs"
    appendix_figs_root = root / "appendix_figs"
    main_tables_root = root / "main_tables"
    appendix_tables_root = root / "appendix_tables"
    for directory in (main_figs_root, appendix_figs_root, main_tables_root, appendix_tables_root):
        directory.mkdir(parents=True, exist_ok=True)

    figure_map = {figure.figure_id: figure for figure in bundle.figures}
    table_map = {table.table_id: table for table in bundle.tables}
    main_figure_ids = [
        "fig_planner_topk_efficiency",
        "fig_planner_fidelity_tradeoff",
        "fig_planner_phase_convergence",
        "fig_planner_failure_pressure",
        "fig_planner_efficiency_frontier",
    ]
    main_table_ids = [
        "tbl_planner_topk_utility",
        "tbl_planner_failure_mode_summary",
        "tbl_planner_efficiency_synthesis",
    ]

    main_figure_captions = {
        Path(figure_map[figure_id].output_path).name: figure_map[figure_id].caption
        for figure_id in main_figure_ids
        if figure_id in figure_map
    }
    appendix_figure_captions = {
        Path(figure.output_path).name: figure.caption
        for figure in bundle.figures
        if figure.figure_id not in set(main_figure_ids)
    }
    main_figure_paths = [
        _copy_file(figure_map[figure_id].output_path, main_figs_root / Path(figure_map[figure_id].output_path).name)
        for figure_id in main_figure_ids
        if figure_id in figure_map
    ]
    appendix_figure_paths = [
        _copy_file(figure.output_path, appendix_figs_root / Path(figure.output_path).name)
        for figure in bundle.figures
        if figure.figure_id not in set(main_figure_ids)
    ]

    main_table_paths: list[str] = []
    main_table_captions: dict[str, str] = {}
    for table_id in main_table_ids:
        if table_id not in table_map:
            continue
        table = table_map[table_id]
        main_table_paths.append(_copy_file(table.csv_output_path, main_tables_root / Path(table.csv_output_path).name))
        main_table_paths.append(_copy_file(table.markdown_output_path, main_tables_root / Path(table.markdown_output_path).name))
        main_table_captions[Path(table.csv_output_path).name] = table.caption
        main_table_captions[Path(table.markdown_output_path).name] = table.caption

    appendix_table_paths: list[str] = []
    appendix_table_captions: dict[str, str] = {}
    for table in bundle.tables:
        if table.table_id in set(main_table_ids):
            continue
        appendix_table_paths.append(_copy_file(table.csv_output_path, appendix_tables_root / Path(table.csv_output_path).name))
        appendix_table_paths.append(_copy_file(table.markdown_output_path, appendix_tables_root / Path(table.markdown_output_path).name))
        appendix_table_captions[Path(table.csv_output_path).name] = table.caption
        appendix_table_captions[Path(table.markdown_output_path).name] = table.caption

    markdown_output_path = root / "planner_paper_layout.md"
    json_output_path = root / "planner_paper_layout_bundle.json"
    caption_manifest_path = root / "planner_caption_manifest.md"
    notes = [
        f"profile={profile_name}",
        f"task_id={bundle.task_id}",
        "main_text prioritizes planner-vs-top-k utility, fidelity tradeoff, phase convergence, failure pressure, and overall efficiency synthesis.",
        "appendix carries the broader ablation overview, rollout boundary audit, and per-mode tables.",
        "captions are exported alongside the layout bundle so table/figure wording stays stable during paper assembly.",
    ]
    layout = PlannerPaperLayoutBundle(
        layout_id=f"planner_paper_layout_{profile_name}",
        profile_name=profile_name,
        main_figures=main_figure_paths,
        appendix_figures=appendix_figure_paths,
        main_tables=main_table_paths,
        appendix_tables=appendix_table_paths,
        main_figure_captions=main_figure_captions,
        appendix_figure_captions=appendix_figure_captions,
        main_table_captions=main_table_captions,
        appendix_table_captions=appendix_table_captions,
        summary_notes=notes,
        json_output_path=str(json_output_path),
        markdown_output_path=str(markdown_output_path),
    )
    markdown_lines = [
        "# Planner Paper Layout Bundle",
        "",
        f"- Profile: `{profile_name}`",
        f"- Task: `{bundle.task_id}`",
        "",
        "## Main Figures",
        "",
        *[f"- `{Path(path).name}`" for path in main_figure_paths],
        "",
        "## Appendix Figures",
        "",
        *[f"- `{Path(path).name}`" for path in appendix_figure_paths],
        "",
        "## Main Tables",
        "",
        *[f"- `{Path(path).name}`" for path in main_table_paths],
        "",
        "## Appendix Tables",
        "",
        *[f"- `{Path(path).name}`" for path in appendix_table_paths],
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in notes],
    ]
    caption_lines = [
        "# Planner Caption Manifest",
        "",
        "## Main Figure Captions",
        "",
        *[f"- `{name}`: {caption}" for name, caption in sorted(main_figure_captions.items())],
        "",
        "## Appendix Figure Captions",
        "",
        *[f"- `{name}`: {caption}" for name, caption in sorted(appendix_figure_captions.items())],
        "",
        "## Main Table Captions",
        "",
        *[f"- `{name}`: {caption}" for name, caption in sorted(main_table_captions.items())],
        "",
        "## Appendix Table Captions",
        "",
        *[f"- `{name}`: {caption}" for name, caption in sorted(appendix_table_captions.items())],
    ]
    markdown_output_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    caption_manifest_path.write_text("\n".join(caption_lines), encoding="utf-8")
    json_output_path.write_text(json.dumps(layout.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    return layout
