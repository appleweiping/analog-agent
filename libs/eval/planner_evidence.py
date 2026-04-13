"""Paper-facing figure and table generation for planner ablation evidence."""

from __future__ import annotations

import json
from pathlib import Path

from libs.eval.paper_evidence import _mean, _write_svg_bar_chart, _write_table_csv, _write_table_markdown
from libs.schema.experiment import ExperimentSuiteResult
from libs.schema.paper_evidence import (
    FigureSeries,
    FigureSpec,
    PlannerAblationEvidenceBundle,
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

    for figure in [simulation_figure, feasible_figure, fidelity_figure, efficiency_figure, phase_figure, rollout_figure, calibration_replanning_figure]:
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

    for table in [comparison_table, delta_vs_top_k_table, step_behavior_table]:
        _write_table_csv(table)
        _write_table_markdown(table)

    conclusions = suite.comparison.conclusions
    summary = PlannerAblationSummary(
        planner_beats_top_k=conclusions.top_k_baseline_effective,
        fidelity_escalation_effective=conclusions.fidelity_effective,
        phase_updates_effective=conclusions.phase_updates_effective,
        calibration_replanning_effective=conclusions.calibration_replanning_effective,
        rollout_guidance_effective=conclusions.rollout_effective,
        notes=[
            *conclusions.conclusion_notes,
            f"modes={','.join(present_modes)}",
        ],
    )

    bundle = PlannerAblationEvidenceBundle(
        task_id=suite.task_id,
        modes=present_modes,
        figures=[simulation_figure, feasible_figure, fidelity_figure, efficiency_figure, phase_figure, rollout_figure, calibration_replanning_figure],
        tables=[comparison_table, delta_vs_top_k_table, step_behavior_table],
        summary=summary,
        json_output_path=str(json_output_path),
    )
    Path(json_output_path).write_text(json.dumps(bundle.model_dump(mode="json"), indent=2), encoding="utf-8")
    return bundle
