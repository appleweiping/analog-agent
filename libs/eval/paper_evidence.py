"""Paper-facing figure and table generation for methodology evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from libs.schema.experiment import ExperimentAggregateSummary, ExperimentSuiteResult
from libs.schema.paper_evidence import (
    FigureSeries,
    FigureSpec,
    TableColumn,
    TableRow,
    TableSpec,
    WorldModelEvidenceBundle,
    WorldModelUtilitySummary,
)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _mode_runs(suite: ExperimentSuiteResult, mode: str):
    return [run for run in suite.runs if run.mode == mode]


def _mode_summary_map(suite: ExperimentSuiteResult):
    return {summary.mode: summary for summary in suite.comparison.mode_summaries} if suite.comparison else {}


def _aggregate_summary_map(suite: ExperimentSuiteResult) -> dict[str, ExperimentAggregateSummary]:
    return {summary.mode: summary for summary in suite.summaries}


def _average_step_series(runs, metric_getter):
    if not runs:
        return []
    max_steps = max(len(run.structured_log) for run in runs)
    values: list[float] = []
    for step_index in range(max_steps):
        step_values = []
        for run in runs:
            if step_index < len(run.structured_log):
                step_values.append(float(metric_getter(run.structured_log[step_index])))
        values.append(_mean(step_values))
    return values


def _average_gap_series(runs) -> list[float]:
    if not runs:
        return []
    max_steps = max(len(run.prediction_gap_by_step) for run in runs)
    values: list[float] = []
    for step_index in range(max_steps):
        step_values = []
        for run in runs:
            if step_index >= len(run.prediction_gap_by_step):
                continue
            gaps = run.prediction_gap_by_step[step_index]
            core = [abs(float(gaps.get(metric, 0.0))) for metric in ("dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w") if metric in gaps]
            if core:
                step_values.append(_mean(core))
        values.append(_mean(step_values))
    return values


def _write_svg_line_chart(spec: FigureSpec) -> None:
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 90, 30, 60, 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_x = [value for series in spec.series for value in series.x_values] or [0.0, 1.0]
    all_y = [value for series in spec.series for value in series.y_values] or [0.0, 1.0]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0

    def map_x(value: float) -> float:
        return margin_left + (value - min_x) / (max_x - min_x) * plot_width

    def map_y(value: float) -> float:
        return margin_top + plot_height - (value - min_y) / (max_y - min_y) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{spec.title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black" stroke-width="2"/>',
    ]

    for index in range(5):
        ratio = index / 4
        y_value = min_y + (max_y - min_y) * ratio
        y = map_y(y_value)
        lines.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#dddddd" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 10}" y="{y + 5}" text-anchor="end" font-size="12" font-family="Arial">{y_value:.3g}</text>')
    for index in range(int(max_x - min_x) + 1):
        x_value = min_x + index
        x = map_x(x_value)
        lines.append(f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{margin_top + plot_height}" stroke="#f0f0f0" stroke-width="1"/>')
        lines.append(f'<text x="{x}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="12" font-family="Arial">{x_value:.0f}</text>')

    for series in spec.series:
        points = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in zip(series.x_values, series.y_values))
        lines.append(f'<polyline fill="none" stroke="{series.color}" stroke-width="3" points="{points}"/>')
        for x, y in zip(series.x_values, series.y_values):
            lines.append(f'<circle cx="{map_x(x):.2f}" cy="{map_y(y):.2f}" r="4" fill="{series.color}"/>')

    legend_x = width - margin_right - 180
    legend_y = margin_top + 10
    for index, series in enumerate(spec.series):
        y = legend_y + index * 24
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{series.color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-size="13" font-family="Arial">{series.label}</text>')

    lines.extend(
        [
            f'<text x="{width/2}" y="{height - 18}" text-anchor="middle" font-size="15" font-family="Arial">{spec.x_label}</text>',
            f'<text x="24" y="{height/2}" text-anchor="middle" font-size="15" font-family="Arial" transform="rotate(-90 24 {height/2})">{spec.y_label}</text>',
            '</svg>',
        ]
    )
    Path(spec.output_path).write_text("\n".join(lines), encoding="utf-8")


def _write_svg_bar_chart(spec: FigureSpec) -> None:
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 90, 30, 60, 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    labels = [series.label for series in spec.series]
    values = [series.y_values[0] if series.y_values else 0.0 for series in spec.series]
    max_y = max(values or [1.0])
    if max_y <= 0.0:
        max_y = 1.0
    bar_width = plot_width / max(1, len(values)) * 0.55
    step = plot_width / max(1, len(values))

    def map_y(value: float) -> float:
        return margin_top + plot_height - (value / max_y) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{spec.title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black" stroke-width="2"/>',
    ]
    for index in range(5):
        ratio = index / 4
        y_value = max_y * ratio
        y = map_y(y_value)
        lines.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#dddddd" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 10}" y="{y + 5}" text-anchor="end" font-size="12" font-family="Arial">{y_value:.3g}</text>')

    for index, (label, value, series) in enumerate(zip(labels, values, spec.series)):
        x = margin_left + index * step + (step - bar_width) / 2
        y = map_y(value)
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{margin_top + plot_height - y:.2f}" fill="{series.color}" opacity="0.9"/>')
        lines.append(f'<text x="{x + bar_width/2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="12" font-family="Arial">{value:.3g}</text>')
        lines.append(f'<text x="{x + bar_width/2:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>')

    lines.extend(
        [
            f'<text x="{width/2}" y="{height - 18}" text-anchor="middle" font-size="15" font-family="Arial">{spec.x_label}</text>',
            f'<text x="24" y="{height/2}" text-anchor="middle" font-size="15" font-family="Arial" transform="rotate(-90 24 {height/2})">{spec.y_label}</text>',
            '</svg>',
        ]
    )
    Path(spec.output_path).write_text("\n".join(lines), encoding="utf-8")


def _write_table_csv(table: TableSpec) -> None:
    with Path(table.csv_output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([column.label for column in table.columns])
        for row in table.rows:
            writer.writerow([row.values.get(column.key, "") for column in table.columns])


def _write_table_markdown(table: TableSpec) -> None:
    lines = [f"# {table.title}", "", table.caption, ""]
    headers = [column.label for column in table.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in table.rows:
        lines.append("| " + " | ".join(str(row.values.get(column.key, "")) for column in table.columns) + " |")
    Path(table.markdown_output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_world_model_evidence_bundle(
    suite: ExperimentSuiteResult,
    *,
    baseline_suite: ExperimentSuiteResult | None = None,
    figures_dir: str | Path,
    tables_dir: str | Path,
    json_output_path: str | Path,
) -> WorldModelEvidenceBundle:
    """Build formal figure/table evidence for world-model utility claims."""

    if suite.comparison is None:
        raise ValueError("world-model evidence requires a methodology comparison suite")

    figures_root = Path(figures_dir)
    tables_root = Path(tables_dir)
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    full_runs = _mode_runs(suite, "full_system")
    no_world_model_runs = _mode_runs(suite, "no_world_model")
    no_calibration_runs = _mode_runs(suite, "no_calibration")
    no_fidelity_runs = _mode_runs(suite, "no_fidelity_escalation")
    summary_map = _mode_summary_map(suite)
    baseline_suite = baseline_suite or suite
    baseline_summary_map = _aggregate_summary_map(baseline_suite)

    figure_gap = FigureSpec(
        figure_id="fig_world_model_prediction_gap_vs_step",
        title="Prediction Gap vs Step",
        chart_type="line",
        x_label="Optimization Step",
        y_label="Average Absolute Prediction Gap",
        series=[
            FigureSeries(
                label="full_system",
                x_values=list(range(len(_average_gap_series(full_runs)))),
                y_values=_average_gap_series(full_runs),
                color="#1f77b4",
            ),
            FigureSeries(
                label="no_calibration",
                x_values=list(range(len(_average_gap_series(no_calibration_runs)))),
                y_values=_average_gap_series(no_calibration_runs),
                color="#d62728",
            ),
            FigureSeries(
                label="no_world_model",
                x_values=list(range(len(_average_gap_series(no_world_model_runs)))),
                y_values=_average_gap_series(no_world_model_runs),
                color="#2ca02c",
            ),
        ],
        caption="Average core-metric prediction gap across optimization steps. Lower is better.",
        output_path=str(figures_root / "world_model_prediction_gap_vs_step.svg"),
    )

    figure_sim_calls = FigureSpec(
        figure_id="fig_world_model_simulation_calls",
        title="Real Simulation Calls by Method Mode",
        chart_type="bar",
        x_label="Method Mode",
        y_label="Average Real Simulation Calls",
        series=[
            FigureSeries(label="full_simulation_baseline", x_values=[0.0], y_values=[baseline_summary_map["full_simulation_baseline"].average_simulation_call_count], color="#7f7f7f"),
            FigureSeries(label="no_world_model_baseline", x_values=[1.0], y_values=[baseline_summary_map["no_world_model_baseline"].average_simulation_call_count], color="#2ca02c"),
            FigureSeries(label="full_system", x_values=[2.0], y_values=[baseline_summary_map["full_system"].average_simulation_call_count], color="#1f77b4"),
        ],
        caption="Average number of real SPICE verifications per run. Lower is better.",
        output_path=str(figures_root / "world_model_simulation_calls.svg"),
    )

    figure_feasible_hit = FigureSpec(
        figure_id="fig_world_model_feasible_hit_rate",
        title="Feasible Hit Rate by Method Mode",
        chart_type="bar",
        x_label="Method Mode",
        y_label="Feasible Hit Rate",
        series=[
            FigureSeries(label="full_simulation_baseline", x_values=[0.0], y_values=[baseline_summary_map["full_simulation_baseline"].feasible_hit_rate], color="#7f7f7f"),
            FigureSeries(label="no_world_model_baseline", x_values=[1.0], y_values=[baseline_summary_map["no_world_model_baseline"].feasible_hit_rate], color="#2ca02c"),
            FigureSeries(label="full_system", x_values=[2.0], y_values=[baseline_summary_map["full_system"].feasible_hit_rate], color="#1f77b4"),
        ],
        caption="Fraction of runs that find a feasible solution.",
        output_path=str(figures_root / "world_model_feasible_hit_rate.svg"),
    )

    figure_trust = FigureSpec(
        figure_id="fig_world_model_trust_guided_selection",
        title="Selected Candidate Uncertainty vs Step",
        chart_type="line",
        x_label="Optimization Step",
        y_label="Mean Selected Uncertainty",
        series=[
            FigureSeries(
                label="full_system",
                x_values=list(range(len(_average_step_series(full_runs, lambda log: log.selected_mean_uncertainty)))),
                y_values=_average_step_series(full_runs, lambda log: log.selected_mean_uncertainty),
                color="#1f77b4",
            ),
            FigureSeries(
                label="no_world_model",
                x_values=list(range(len(_average_step_series(no_world_model_runs, lambda log: log.selected_mean_uncertainty)))),
                y_values=_average_step_series(no_world_model_runs, lambda log: log.selected_mean_uncertainty),
                color="#2ca02c",
            ),
            FigureSeries(
                label="no_fidelity_escalation",
                x_values=list(range(len(_average_step_series(no_fidelity_runs, lambda log: log.selected_mean_uncertainty)))),
                y_values=_average_step_series(no_fidelity_runs, lambda log: log.selected_mean_uncertainty),
                color="#9467bd",
            ),
        ],
        caption="Selection-time uncertainty profile of candidates chosen for real simulation.",
        output_path=str(figures_root / "world_model_trust_guided_selection.svg"),
    )

    for figure in (figure_gap, figure_sim_calls, figure_feasible_hit, figure_trust):
        if figure.chart_type == "line":
            _write_svg_line_chart(figure)
        else:
            _write_svg_bar_chart(figure)

    comparison_table = TableSpec(
        table_id="tbl_world_model_method_comparison",
        title="World Model Utility Comparison",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="simulation_call_count", label="Avg Sim Calls"),
            TableColumn(key="feasible_hit_rate", label="Feasible Hit Rate"),
            TableColumn(key="average_convergence_step", label="Avg Convergence Step"),
            TableColumn(key="focused_truth_ratio", label="Focused Truth Ratio"),
            TableColumn(key="gbw_gap", label="GBW Gap"),
            TableColumn(key="pm_gap", label="PM Gap"),
            TableColumn(key="power_gap", label="Power Gap"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": summary.mode,
                    "simulation_call_count": summary.simulation_call_count,
                    "feasible_hit_rate": summary.feasible_hit_rate,
                    "average_convergence_step": summary.average_convergence_step,
                    "focused_truth_ratio": summary.focused_truth_ratio,
                    "gbw_gap": summary.average_prediction_gap.get("gbw_hz", 0.0),
                    "pm_gap": summary.average_prediction_gap.get("phase_margin_deg", 0.0),
                    "power_gap": summary.average_prediction_gap.get("power_w", 0.0),
                }
            )
            for summary in suite.comparison.mode_summaries
        ],
        caption="Primary methodology comparison table for OTA v1 world-model utility evidence.",
        csv_output_path=str(tables_root / "world_model_method_comparison.csv"),
        markdown_output_path=str(tables_root / "world_model_method_comparison.md"),
    )

    step_table = TableSpec(
        table_id="tbl_prediction_gap_by_step",
        title="Prediction Gap by Step",
        columns=[
            TableColumn(key="step", label="Step"),
            TableColumn(key="full_system", label="Full System"),
            TableColumn(key="no_calibration", label="No Calibration"),
            TableColumn(key="no_world_model", label="No World Model"),
        ],
        rows=[
            TableRow(
                values={
                    "step": step_index,
                    "full_system": figure_gap.series[0].y_values[step_index] if step_index < len(figure_gap.series[0].y_values) else "",
                    "no_calibration": figure_gap.series[1].y_values[step_index] if step_index < len(figure_gap.series[1].y_values) else "",
                    "no_world_model": figure_gap.series[2].y_values[step_index] if step_index < len(figure_gap.series[2].y_values) else "",
                }
            )
            for step_index in range(max(len(series.y_values) for series in figure_gap.series))
        ],
        caption="Average core-metric prediction gap per optimization step.",
        csv_output_path=str(tables_root / "prediction_gap_by_step.csv"),
        markdown_output_path=str(tables_root / "prediction_gap_by_step.md"),
    )

    trust_table = TableSpec(
        table_id="tbl_selection_behavior",
        title="Trust-Guided Selection Profile",
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
                    "mean_selected_uncertainty": _mean([log.selected_mean_uncertainty for run in _mode_runs(suite, mode) for log in run.structured_log]),
                    "mean_selected_confidence": _mean([log.selected_mean_confidence for run in _mode_runs(suite, mode) for log in run.structured_log]),
                    "mean_selected_simulation_value": _mean([log.selected_mean_simulation_value for run in _mode_runs(suite, mode) for log in run.structured_log]),
                    "mean_selected_predicted_feasibility": _mean([log.selected_mean_predicted_feasibility for run in _mode_runs(suite, mode) for log in run.structured_log]),
                }
            )
            for mode in ("full_system", "no_world_model", "no_calibration", "no_fidelity_escalation")
        ],
        caption="Planner-selection profile aggregated over experiment steps and runs.",
        csv_output_path=str(tables_root / "trust_guided_selection_profile.csv"),
        markdown_output_path=str(tables_root / "trust_guided_selection_profile.md"),
    )

    baseline_table = TableSpec(
        table_id="tbl_world_model_budget_comparison",
        title="World Model Budget Comparison",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="avg_sim_calls", label="Avg Sim Calls"),
            TableColumn(key="feasible_hit_rate", label="Feasible Hit Rate"),
            TableColumn(key="efficiency_score", label="Efficiency Score"),
            TableColumn(key="selection_ratio", label="Selection Ratio"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": summary.mode,
                    "avg_sim_calls": summary.average_simulation_call_count,
                    "feasible_hit_rate": summary.feasible_hit_rate,
                    "efficiency_score": summary.average_efficiency_score,
                    "selection_ratio": summary.average_selection_ratio,
                }
            )
            for summary in baseline_suite.summaries
            if summary.mode in {"full_simulation_baseline", "no_world_model_baseline", "full_system"}
        ],
        caption="Budget-facing comparison for with/without world model under natural execution.",
        csv_output_path=str(tables_root / "world_model_budget_comparison.csv"),
        markdown_output_path=str(tables_root / "world_model_budget_comparison.md"),
    )

    for table in (comparison_table, step_table, trust_table, baseline_table):
        _write_table_csv(table)
        _write_table_markdown(table)

    full_summary = summary_map["full_system"]
    no_world_summary = summary_map["no_world_model"]
    no_cal_summary = summary_map["no_calibration"]
    budget_full_summary = baseline_summary_map["full_system"]
    budget_no_world_summary = baseline_summary_map["no_world_model_baseline"]
    summary = WorldModelUtilitySummary(
        world_model_reduces_simulations=budget_full_summary.average_simulation_call_count < budget_no_world_summary.average_simulation_call_count,
        world_model_preserves_or_improves_feasible_hit_rate=budget_full_summary.feasible_hit_rate >= budget_no_world_summary.feasible_hit_rate,
        calibration_reduces_prediction_gap=full_summary.average_prediction_gap.get("gbw_hz", 0.0) < no_cal_summary.average_prediction_gap.get("gbw_hz", float("inf")),
        trust_guides_selection_behavior=_mean([log.selected_mean_uncertainty for run in full_runs for log in run.structured_log]) != _mean([log.selected_mean_uncertainty for run in no_world_model_runs for log in run.structured_log]),
        notes=[
            f"system_delta_vs_full_sim_baseline_calls={round(baseline_summary_map['full_simulation_baseline'].average_simulation_call_count - budget_full_summary.average_simulation_call_count, 6)}",
            f"world_model_delta_sim_calls={round(budget_no_world_summary.average_simulation_call_count - budget_full_summary.average_simulation_call_count, 6)}",
            f"world_model_delta_feasible_hit={round(budget_full_summary.feasible_hit_rate - budget_no_world_summary.feasible_hit_rate, 6)}",
            f"calibration_delta_gbw_gap={round(no_cal_summary.average_prediction_gap.get('gbw_hz', 0.0) - full_summary.average_prediction_gap.get('gbw_hz', 0.0), 6)}",
            f"trust_selection_delta_uncertainty={round(_mean([log.selected_mean_uncertainty for run in no_world_model_runs for log in run.structured_log]) - _mean([log.selected_mean_uncertainty for run in full_runs for log in run.structured_log]), 6)}",
        ],
    )

    bundle = WorldModelEvidenceBundle(
        task_id=suite.task_id,
        modes=sorted({*suite.modes, *baseline_suite.modes}),
        figures=[figure_gap, figure_sim_calls, figure_feasible_hit, figure_trust],
        tables=[comparison_table, step_table, trust_table, baseline_table],
        summary=summary,
        json_output_path=str(json_output_path),
    )
    Path(json_output_path).write_text(json.dumps(bundle.model_dump(mode="json"), indent=2), encoding="utf-8")
    return bundle
