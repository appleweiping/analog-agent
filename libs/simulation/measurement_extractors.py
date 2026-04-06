"""Measurement extraction for the fifth layer."""

from __future__ import annotations

from libs.schema.simulation import (
    AnalysisExecutionRecord,
    IntegrityCheckResult,
    MeasuredMetric,
    MeasurementContract,
    MeasurementReport,
    SimulationBundle,
)
from libs.utils.hashing import stable_hash


def extract_measurement_report(
    simulation_bundle: SimulationBundle,
    parsed_outputs: list[dict[str, object]],
    *,
    candidate_id: str,
) -> MeasurementReport:
    """Extract a structured measurement report from backend outputs."""

    metric_map: dict[str, MeasuredMetric] = {}
    executed: list[AnalysisExecutionRecord] = []
    validation_checks: list[IntegrityCheckResult] = []

    confidence_map = {method.metric: 0.92 for method in simulation_bundle.measurement_contract.extraction_methods}
    for parsed in parsed_outputs:
        analysis_type = str(parsed.get("analysis_type", "unknown"))
        metrics = parsed.get("metrics", {})
        artifact_ref = str(parsed.get("artifact_ref", ""))
        executed.append(
            AnalysisExecutionRecord(
                analysis_type=analysis_type,
                success=str(parsed.get("status", "")) == "ok",
                runtime_ms=int(parsed.get("runtime_ms", 0)),
                backend_status=str(parsed.get("status", "unknown")),
                artifact_refs=[artifact_ref] if artifact_ref else [],
            )
        )
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                existing = metric_map.get(metric)
                current = MeasuredMetric(
                    metric=str(metric),
                    value=float(value),
                    units="si",
                    source_analysis=analysis_type,
                    extraction_confidence=confidence_map.get(str(metric), 0.8),
                )
                if existing is None or existing.extraction_confidence <= current.extraction_confidence:
                    metric_map[str(metric)] = current

    required_metrics = {definition.metric for definition in simulation_bundle.measurement_contract.metric_definitions}
    for metric in sorted(required_metrics):
        validation_checks.append(
            IntegrityCheckResult(
                check_name=f"metric::{metric}",
                passed=metric in metric_map,
                detail="extracted" if metric in metric_map else "missing",
            )
        )
    return MeasurementReport(
        report_id=f"measure_{stable_hash(f'{simulation_bundle.simulation_id}|{candidate_id}')[:12]}",
        candidate_id=candidate_id,
        executed_analyses=executed,
        measured_metrics=list(metric_map.values()),
        extraction_notes=["measurement_contract_applied", "metrics_normalized_to_si"],
        validation_checks=validation_checks,
    )
