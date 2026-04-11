"""Statistics builders, aggregators, and exporters for research reporting."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from libs.eval.metrics import simulation_reduction_ratio
from libs.schema.stats import (
    ExperimentStatsRecord,
    FailureModeStatsSummary,
    FidelityStatsSummary,
    MetricGapRecord,
    PredictionGapStatsSummary,
    StatsAggregationResult,
    VerificationStatsRecord,
)
from libs.utils.hashing import stable_hash

if TYPE_CHECKING:
    from libs.schema.experiment import ExperimentResult, ExperimentSuiteResult
    from libs.schema.system_binding import SystemAcceptanceResult

CORE_GAP_METRICS = ("dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _normalized_failure_mode(raw_mode: str) -> str:
    mapping = {
        "none": "none",
        "design_failure": "design_failure",
        "operating_region_failure": "design_failure",
        "stability_failure": "stability_failure",
        "drive_bandwidth_failure": "bandwidth_failure",
        "power_failure": "power_failure",
        "noise_power_area_tradeoff_failure": "power_failure",
        "measurement_failure": "measurement_failure",
        "analysis_failure": "analysis_failure",
        "simulation_invalid": "simulation_invalid",
        "netlist_failure": "netlist_failure",
        "simulator_failure": "simulation_invalid",
    }
    return mapping.get(raw_mode, "design_failure")


def build_verification_stats_record(simulation_bundle, simulation_request, verification_result) -> VerificationStatsRecord:
    """Build one formal statistics record from a real verification execution."""

    result = verification_result
    measured_metrics = {metric.metric: float(metric.value) for metric in result.measurement_report.measured_metrics}
    statuses = {
        measurement.metric: f"{measurement.status.status}:{measurement.failure_reason.code}"
        for measurement in result.measurement_report.measurement_results
    }
    runtime_sec = round(sum(record.runtime_ms for record in result.measurement_report.executed_analyses) / 1000.0, 6)
    truth_metrics = {metric.metric: float(metric.value) for metric in result.calibration_payload.truth_record.metrics}
    predicted_metrics: dict[str, float | None] = {}
    for metric_name in CORE_GAP_METRICS:
        if metric_name in result.calibration_payload.residual_metrics and metric_name in truth_metrics:
            predicted_metrics[metric_name] = truth_metrics[metric_name] - float(result.calibration_payload.residual_metrics[metric_name])
        elif metric_name in truth_metrics:
            predicted_metrics[metric_name] = None
    gap_records: list[MetricGapRecord] = []
    for metric_name in CORE_GAP_METRICS:
        truth_value = truth_metrics.get(metric_name)
        predicted_value = predicted_metrics.get(metric_name)
        absolute_error = None
        relative_error = None
        if truth_value is not None and predicted_value is not None:
            absolute_error = round(abs(predicted_value - truth_value), 6)
            relative_error = round(absolute_error / max(abs(truth_value), 1e-12), 6)
        gap_records.append(
            MetricGapRecord(
                metric=metric_name,
                predicted_value=predicted_value,
                ground_truth_value=truth_value,
                absolute_error=absolute_error,
                relative_error=relative_error,
            )
        )
    return VerificationStatsRecord(
        record_id=f"vstats_{stable_hash(result.result_id)[:12]}",
        candidate_id=result.candidate_id,
        task_id=simulation_bundle.parent_task_id,
        family=simulation_bundle.netlist_instance.template_binding.circuit_family,
        truth_level=result.validation_status.truth_level,
        fidelity_level=result.executed_fidelity,
        analysis_types=[record.analysis_type for record in result.measurement_report.executed_analyses],
        runtime_sec=runtime_sec,
        measured_metrics=measured_metrics,
        measurement_statuses=statuses,
        feasibility_status=result.feasibility_status,
        dominant_failure_mode=_normalized_failure_mode(result.failure_attribution.primary_failure_class),
        prediction_ground_truth_gap=gap_records,
        artifact_refs=list(result.artifact_refs),
        planner_escalation_reason=result.planner_feedback.escalation_reason or simulation_request.escalation_reason,
        validation_status=result.validation_status.validity_state,
    )


def build_prediction_gap_summary(records: list[VerificationStatsRecord]) -> PredictionGapStatsSummary:
    """Aggregate prediction-gap statistics over verification records."""

    abs_by_metric: defaultdict[str, list[float]] = defaultdict(list)
    rel_by_metric: defaultdict[str, list[float]] = defaultdict(list)
    by_task: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_family: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_fidelity: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for gap in record.prediction_ground_truth_gap:
            if gap.absolute_error is None:
                continue
            abs_by_metric[gap.metric].append(gap.absolute_error)
            if gap.relative_error is not None:
                rel_by_metric[gap.metric].append(gap.relative_error)
            by_task[record.task_id][gap.metric].append(gap.absolute_error)
            by_family[record.family][gap.metric].append(gap.absolute_error)
            by_fidelity[record.fidelity_level][gap.metric].append(gap.absolute_error)
    return PredictionGapStatsSummary(
        record_count=len(records),
        metrics_covered=sorted(abs_by_metric.keys()),
        average_absolute_error={metric: _mean(values) for metric, values in sorted(abs_by_metric.items())},
        average_relative_error={metric: _mean(values) for metric, values in sorted(rel_by_metric.items())},
        by_task_id={task_id: {metric: _mean(values) for metric, values in sorted(metrics.items())} for task_id, metrics in sorted(by_task.items())},
        by_family={family: {metric: _mean(values) for metric, values in sorted(metrics.items())} for family, metrics in sorted(by_family.items())},
        by_fidelity={fidelity: {metric: _mean(values) for metric, values in sorted(metrics.items())} for fidelity, metrics in sorted(by_fidelity.items())},
    )


def build_failure_mode_summary(records: list[VerificationStatsRecord]) -> FailureModeStatsSummary:
    """Aggregate dominant failure-mode statistics."""

    frequency = Counter(record.dominant_failure_mode for record in records if record.dominant_failure_mode != "none")
    by_fidelity: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_truth_level: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_family: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if record.dominant_failure_mode == "none":
            continue
        by_fidelity[record.fidelity_level][record.dominant_failure_mode] += 1
        by_truth_level[record.truth_level][record.dominant_failure_mode] += 1
        by_family[record.family][record.dominant_failure_mode] += 1
    total = max(1, sum(frequency.values()))
    return FailureModeStatsSummary(
        total_records=len(records),
        frequency=dict(sorted(frequency.items())),
        ratio={mode: round(count / total, 6) for mode, count in sorted(frequency.items())},
        by_fidelity={key: dict(sorted(counter.items())) for key, counter in sorted(by_fidelity.items())},
        by_truth_level={key: dict(sorted(counter.items())) for key, counter in sorted(by_truth_level.items())},
        by_family={key: dict(sorted(counter.items())) for key, counter in sorted(by_family.items())},
    )


def build_fidelity_summary(records: list[VerificationStatsRecord]) -> FidelityStatsSummary:
    """Aggregate fidelity usage and escalation statistics."""

    quick = sum(1 for record in records if record.fidelity_level == "quick_truth")
    focused = sum(1 for record in records if record.fidelity_level == "focused_truth")
    escalation_counter = Counter(
        record.planner_escalation_reason
        for record in records
        if record.fidelity_level == "focused_truth" and record.planner_escalation_reason
    )
    total = len(records)
    return FidelityStatsSummary(
        total_real_verifications=total,
        quick_truth_calls=quick,
        focused_truth_calls=focused,
        escalation_count=focused,
        escalation_reason_distribution=dict(sorted(escalation_counter.items())),
        focused_truth_ratio=round(focused / total, 6) if total else 0.0,
        fidelity_escalation_rate=round(focused / total, 6) if total else 0.0,
    )


def total_real_simulation_calls(records: list[VerificationStatsRecord]) -> int:
    return len(records)


def measurement_success_rate(records: list[VerificationStatsRecord]) -> float:
    if not records:
        return 0.0
    success_count = 0
    for record in records:
        statuses = list(record.measurement_statuses.values())
        if statuses and all(status.startswith("measured:") for status in statuses):
            success_count += 1
    return round(success_count / len(records), 6)


def feasible_verification_rate(records: list[VerificationStatsRecord]) -> float:
    if not records:
        return 0.0
    feasible = sum(1 for record in records if record.feasibility_status in {"feasible_nominal", "feasible_certified"})
    return round(feasible / len(records), 6)


def predicted_feasible_to_verified_feasible_conversion(records: list[VerificationStatsRecord]) -> float:
    if not records:
        return 0.0
    predicted_feasible = sum(1 for record in records if any(gap.predicted_value is not None for gap in record.prediction_ground_truth_gap))
    if predicted_feasible <= 0:
        return 0.0
    verified = sum(1 for record in records if record.feasibility_status in {"feasible_nominal", "feasible_certified"})
    return round(verified / predicted_feasible, 6)


def average_prediction_gap(records: list[VerificationStatsRecord]) -> PredictionGapStatsSummary:
    return build_prediction_gap_summary(records)


def dominant_failure_mode_distribution(records: list[VerificationStatsRecord]) -> FailureModeStatsSummary:
    return build_failure_mode_summary(records)


def fidelity_escalation_rate(records: list[VerificationStatsRecord]) -> float:
    return build_fidelity_summary(records).fidelity_escalation_rate


def build_experiment_stats_record(result: "ExperimentResult") -> ExperimentStatsRecord:
    """Build a formal experiment stats record from one experiment result."""

    verification_records = list(result.verification_stats)
    return ExperimentStatsRecord(
        run_id=result.run_id,
        task_id=result.task_id,
        mode=result.mode,
        simulation_call_count=result.simulation_call_count,
        candidate_count=result.candidate_count,
        best_feasible_found=result.best_feasible_found,
        convergence_step=result.convergence_step,
        verification_records=verification_records,
        prediction_gap_summary=build_prediction_gap_summary(verification_records),
        fidelity_summary=build_fidelity_summary(verification_records),
        failure_mode_summary=build_failure_mode_summary(verification_records),
    )


def aggregate_stats_from_verification_records(
    records: list[VerificationStatsRecord],
    *,
    scope: Literal["acceptance_run", "experiment_suite", "benchmark_suite"],
    source_ids: list[str],
    baseline_calls: int | None = None,
    system_calls: int | None = None,
) -> StatsAggregationResult:
    """Aggregate verification statistics into a top-level structured summary."""

    fidelity_summary = build_fidelity_summary(records)
    failure_summary = build_failure_mode_summary(records)
    gap_summary = build_prediction_gap_summary(records)
    effective_system_calls = system_calls if system_calls is not None else len(records)
    reduction = simulation_reduction_ratio(baseline_calls, effective_system_calls) if baseline_calls is not None else None
    return StatsAggregationResult(
        aggregation_scope=scope,
        source_ids=source_ids,
        total_real_simulation_calls=total_real_simulation_calls(records),
        measurement_success_rate=measurement_success_rate(records),
        feasible_verification_rate=feasible_verification_rate(records),
        predicted_feasible_to_verified_feasible_conversion=predicted_feasible_to_verified_feasible_conversion(records),
        simulation_reduction_ratio=reduction,
        average_prediction_gap=gap_summary,
        dominant_failure_mode_distribution=failure_summary,
        fidelity_summary=fidelity_summary,
        failure_mode_summary=failure_summary,
        prediction_gap_summary=gap_summary,
    )


def aggregate_stats(target) -> StatsAggregationResult:
    """Aggregate formal statistics from acceptance or experiment scopes."""

    from libs.schema.experiment import ExperimentSuiteResult
    from libs.schema.system_binding import SystemAcceptanceResult

    if isinstance(target, SystemAcceptanceResult):
        return aggregate_stats_from_verification_records(
            list(target.verification_stats),
            scope="acceptance_run",
            source_ids=[target.task_id, target.search_id],
        )
    if isinstance(target, ExperimentSuiteResult):
        records = [record for run in target.runs for record in run.verification_stats]
        baseline_calls = None
        baseline_runs = [run for run in target.runs if run.mode == "full_simulation_baseline"]
        system_runs = [run for run in target.runs if run.mode == "full_system"]
        system_calls = None
        if baseline_runs and system_runs:
            baseline_calls = round(sum(run.simulation_call_count for run in baseline_runs) / len(baseline_runs))
            system_calls = round(sum(run.simulation_call_count for run in system_runs) / len(system_runs))
        return aggregate_stats_from_verification_records(
            records,
            scope="benchmark_suite" if target.task_id.startswith("benchmark-") else "experiment_suite",
            source_ids=[target.task_id, *target.modes],
            baseline_calls=baseline_calls,
            system_calls=system_calls,
        )
    raise TypeError(f"unsupported stats aggregation target: {type(target)!r}")


def export_stats_summary(target) -> StatsAggregationResult:
    """Return the structured statistics summary for a supported target."""

    return aggregate_stats(target)


def export_stats_json(target, output_path: str | Path) -> Path:
    """Export a structured statistics summary as JSON."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = export_stats_summary(target).model_dump(mode="json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def export_stats_csv(target, output_path: str | Path) -> Path:
    """Export a flattened statistics summary as CSV."""

    summary = export_stats_summary(target)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("aggregation_scope", summary.aggregation_scope),
        ("total_real_simulation_calls", summary.total_real_simulation_calls),
        ("measurement_success_rate", summary.measurement_success_rate),
        ("feasible_verification_rate", summary.feasible_verification_rate),
        ("predicted_feasible_to_verified_feasible_conversion", summary.predicted_feasible_to_verified_feasible_conversion),
        ("simulation_reduction_ratio", summary.simulation_reduction_ratio if summary.simulation_reduction_ratio is not None else ""),
        ("fidelity_escalation_rate", summary.fidelity_summary.fidelity_escalation_rate),
        ("focused_truth_ratio", summary.fidelity_summary.focused_truth_ratio),
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
    return output
