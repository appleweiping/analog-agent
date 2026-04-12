"""Measurement extraction for the fifth layer."""

from __future__ import annotations

from libs.schema.simulation import (
    AnalysisExecutionRecord,
    IntegrityCheckResult,
    MeasuredMetric,
    MeasurementFailureReason,
    MeasurementReport,
    MeasurementResult,
    MeasurementStatus,
    SimulationBundle,
)
from libs.utils.hashing import stable_hash


def _analysis_success(parsed: dict[str, object]) -> bool:
    return str(parsed.get("status", "unknown")) == "ok"


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _result(
    metric: str,
    units: str,
    source_analysis: str,
    *,
    status: str,
    value: float | None = None,
    raw_value: float | None = None,
    postprocessed_value: float | None = None,
    confidence: float = 0.0,
    reason: str = "none",
    detail: str | None = None,
    provenance: list[str] | None = None,
) -> MeasurementResult:
    return MeasurementResult(
        metric=metric,
        units=units,
        source_analysis=source_analysis,
        status=MeasurementStatus(status=status, detail=detail),
        failure_reason=MeasurementFailureReason(code=reason, detail=detail),
        value=value,
        raw_value=raw_value,
        postprocessed_value=postprocessed_value,
        confidence=confidence,
        provenance=provenance or [],
    )


def _curve_rows(parsed: dict[str, object]) -> list[dict[str, float | None]]:
    rows = parsed.get("ac_curve")
    if isinstance(rows, list):
        normalized: list[dict[str, float | None]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            frequency = _numeric(row.get("frequency_hz"))
            gain_db = _numeric(row.get("gain_db"))
            phase_deg = _numeric(row.get("phase_deg"))
            if frequency is None or gain_db is None:
                continue
            normalized.append(
                {
                    "frequency_hz": frequency,
                    "gain_db": gain_db,
                    "phase_deg": phase_deg,
                }
            )
        return normalized
    return []


def _crossings(rows: list[dict[str, float | None]]) -> list[tuple[float, float | None]]:
    hits: list[tuple[float, float | None]] = []
    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]
        prev_gain = previous["gain_db"]
        curr_gain = current["gain_db"]
        if prev_gain == 0.0:
            hits.append((previous["frequency_hz"], previous.get("phase_deg")))
            continue
        if prev_gain >= 0.0 >= curr_gain:
            ratio = 0.0 if prev_gain == curr_gain else (0.0 - prev_gain) / (curr_gain - prev_gain)
            frequency = previous["frequency_hz"] + (current["frequency_hz"] - previous["frequency_hz"]) * ratio
            prev_phase = previous.get("phase_deg")
            curr_phase = current.get("phase_deg")
            phase = None
            if prev_phase is not None and curr_phase is not None:
                phase = prev_phase + (curr_phase - prev_phase) * ratio
            hits.append((frequency, phase))
    return hits


def _metric_from_direct_source(metric: str, source_analysis: str, parsed: dict[str, object], *, confidence: float) -> MeasurementResult | None:
    metrics = parsed.get("metrics", {})
    if isinstance(metrics, dict):
        value = _numeric(metrics.get(metric))
        if value is not None:
            return _result(
                metric,
                "si",
                source_analysis,
                status="measured",
                value=value,
                raw_value=value,
                postprocessed_value=value,
                confidence=confidence,
                provenance=["direct_metric_field"],
            )
    return None


def _extract_dc_gain(ac_output: dict[str, object]) -> MeasurementResult:
    if not _analysis_success(ac_output):
        return _result("dc_gain_db", "si", "ac", status="analysis_failed", reason="analysis_failure", detail="ac_analysis_failed")
    rows = _curve_rows(ac_output)
    if rows:
        if len(rows) < 3:
            return _result(
                "dc_gain_db",
                "si",
                "ac",
                status="extraction_failed",
                reason="insufficient_curve_quality",
                detail="too_few_ac_samples",
                provenance=["ac_curve"],
            )
        gain = rows[0]["gain_db"]
        confidence = 0.95 if len(rows) >= 8 else 0.72
        return _result(
            "dc_gain_db",
            "si",
            "ac",
            status="measured",
            value=gain,
            raw_value=gain,
            postprocessed_value=gain,
            confidence=confidence,
            provenance=["ac_curve:first_point"],
        )
    direct = _metric_from_direct_source("dc_gain_db", "ac", ac_output, confidence=0.9)
    if direct is not None:
        return direct
    return _result("dc_gain_db", "si", "ac", status="extraction_failed", reason="no_metric_source", detail="missing_ac_gain_source")


def _extract_gbw(ac_output: dict[str, object]) -> MeasurementResult:
    if not _analysis_success(ac_output):
        return _result("gbw_hz", "si", "ac", status="analysis_failed", reason="analysis_failure", detail="ac_analysis_failed")
    rows = _curve_rows(ac_output)
    if rows:
        if len(rows) < 3:
            return _result(
                "gbw_hz",
                "si",
                "ac",
                status="extraction_failed",
                reason="insufficient_curve_quality",
                detail="too_few_ac_samples",
                provenance=["ac_curve"],
            )
        crossings = _crossings(rows)
        if not crossings:
            return _result(
                "gbw_hz",
                "si",
                "ac",
                status="indeterminate",
                reason="curve_exists_but_no_crossing",
                detail="unity_gain_crossing_not_found",
                confidence=0.2,
                provenance=["ac_curve"],
            )
        selected = crossings[0]
        reason = "multiple_crossings" if len(crossings) > 1 else "none"
        detail = "selected_lowest_frequency_crossing" if len(crossings) > 1 else None
        confidence = 0.58 if len(crossings) > 1 else 0.92
        return _result(
            "gbw_hz",
            "si",
            "ac",
            status="measured",
            value=selected[0],
            raw_value=selected[0],
            postprocessed_value=selected[0],
            confidence=confidence,
            reason=reason,
            detail=detail,
            provenance=["ac_curve:crossing_detection"],
        )
    direct = _metric_from_direct_source("gbw_hz", "ac", ac_output, confidence=0.82)
    if direct is not None:
        if (direct.value or 0.0) <= 0.0:
            return _result(
                "gbw_hz",
                "si",
                "ac",
                status="indeterminate",
                reason="curve_exists_but_no_crossing",
                detail="non_positive_direct_gbw",
                confidence=0.2,
                provenance=["direct_metric_field"],
            )
        return direct
    return _result("gbw_hz", "si", "ac", status="extraction_failed", reason="no_metric_source", detail="missing_gbw_source")


def _extract_phase_margin(ac_output: dict[str, object], gbw_result: MeasurementResult) -> MeasurementResult:
    if not _analysis_success(ac_output):
        return _result("phase_margin_deg", "si", "ac", status="analysis_failed", reason="analysis_failure", detail="ac_analysis_failed")
    rows = _curve_rows(ac_output)
    if rows:
        if len(rows) < 3:
            return _result(
                "phase_margin_deg",
                "si",
                "ac",
                status="extraction_failed",
                reason="insufficient_curve_quality",
                detail="too_few_ac_samples",
                provenance=["ac_curve"],
            )
        crossings = _crossings(rows)
        if not crossings:
            direct = _metric_from_direct_source("phase_margin_deg", "ac", ac_output, confidence=0.7)
            if direct is not None and direct.value is not None and direct.value > 0.0:
                return direct.model_copy(
                    update={
                        "failure_reason": MeasurementFailureReason(
                            code="curve_exists_but_no_crossing",
                            detail="used_direct_phase_proxy_without_curve_crossing",
                        ),
                        "provenance": [*direct.provenance, "direct_phase_fallback"],
                    }
                )
            return _result(
                "phase_margin_deg",
                "si",
                "ac",
                status="indeterminate",
                reason="curve_exists_but_no_crossing",
                detail="phase_margin_requires_unity_gain_crossing",
                confidence=0.2,
                provenance=["ac_curve"],
            )
        _, phase_deg = crossings[0]
        if phase_deg is None:
            direct = _metric_from_direct_source("phase_margin_deg", "ac", ac_output, confidence=0.72)
            if direct is not None and direct.value is not None and direct.value > 0.0:
                return direct.model_copy(
                    update={
                        "failure_reason": MeasurementFailureReason(
                            code="invalid_phase_readout",
                            detail="used_direct_phase_proxy_after_missing_phase",
                        ),
                        "provenance": [*direct.provenance, "direct_phase_fallback"],
                    }
                )
            return _result(
                "phase_margin_deg",
                "si",
                "ac",
                status="indeterminate",
                reason="invalid_phase_readout",
                detail="phase_missing_at_selected_crossing",
                confidence=0.2,
                provenance=["ac_curve"],
            )
        phase_margin = 180.0 + phase_deg
        if phase_margin < 0.0 or phase_margin > 180.0:
            direct = _metric_from_direct_source("phase_margin_deg", "ac", ac_output, confidence=0.72)
            if direct is not None and direct.value is not None and direct.value > 0.0:
                return direct.model_copy(
                    update={
                        "failure_reason": MeasurementFailureReason(
                            code="invalid_phase_readout",
                            detail="used_direct_phase_proxy_after_out_of_range_curve_phase",
                        ),
                        "provenance": [*direct.provenance, "direct_phase_fallback"],
                    }
                )
            return _result(
                "phase_margin_deg",
                "si",
                "ac",
                status="indeterminate",
                reason="invalid_phase_readout",
                detail=f"phase_margin_out_of_range:{phase_margin:.4f}",
                confidence=0.2,
                provenance=["ac_curve"],
            )
        reason = gbw_result.failure_reason.code if gbw_result.failure_reason.code == "multiple_crossings" else "none"
        detail = "selected_crossing_matches_gbw_rule" if reason == "multiple_crossings" else None
        confidence = 0.58 if reason == "multiple_crossings" else 0.9
        return _result(
            "phase_margin_deg",
            "si",
            "ac",
            status="measured",
            value=phase_margin,
            raw_value=phase_margin,
            postprocessed_value=phase_margin,
            confidence=confidence,
            reason=reason,
            detail=detail,
            provenance=["ac_curve:phase_at_crossing"],
        )
    direct = _metric_from_direct_source("phase_margin_deg", "ac", ac_output, confidence=0.8)
    if direct is not None:
        if direct.value is None or direct.value <= 0.0:
            return _result(
                "phase_margin_deg",
                "si",
                "ac",
                status="indeterminate",
                reason="invalid_phase_readout",
                detail="non_positive_phase_margin",
                confidence=0.2,
                provenance=["direct_metric_field"],
            )
        return direct
    return _result(
        "phase_margin_deg",
        "si",
        "ac",
        status="indeterminate",
        reason=gbw_result.failure_reason.code if gbw_result.failure_reason.code != "none" else "no_metric_source",
        detail="phase_margin_not_available",
        confidence=0.2,
        provenance=["derived_from_ac"],
    )


def _extract_power(op_output: dict[str, object]) -> MeasurementResult:
    if not _analysis_success(op_output):
        return _result("power_w", "si", "op", status="analysis_failed", reason="analysis_failure", detail="op_analysis_failed")
    diagnostics = op_output.get("op_diagnostics", {})
    if isinstance(diagnostics, dict):
        supply_currents = diagnostics.get("supply_currents")
        supply_voltage = _numeric(diagnostics.get("supply_voltage_v"))
        if isinstance(supply_currents, list):
            numeric_currents = [_numeric(item) for item in supply_currents]
            numeric_currents = [item for item in numeric_currents if item is not None]
            if not numeric_currents:
                return _result(
                    "power_w",
                    "si",
                    "op",
                    status="indeterminate",
                    reason="power_unavailable",
                    detail="supply_currents_present_but_not_numeric",
                    confidence=0.2,
                    provenance=["op_diagnostics"],
                )
            if supply_voltage is None:
                return _result(
                    "power_w",
                    "si",
                    "op",
                    status="indeterminate",
                    reason="power_supply_missing",
                    detail="missing_supply_voltage_for_power_aggregation",
                    confidence=0.2,
                    provenance=["op_diagnostics"],
                )
            has_positive = any(value > 0 for value in numeric_currents)
            has_negative = any(value < 0 for value in numeric_currents)
            if has_positive and has_negative:
                return _result(
                    "power_w",
                    "si",
                    "op",
                    status="indeterminate",
                    reason="current_direction_ambiguous",
                    detail="mixed_supply_current_directions",
                    confidence=0.2,
                    provenance=["op_diagnostics"],
                )
            aggregated_current = sum(abs(value) for value in numeric_currents)
            power = aggregated_current * supply_voltage
            return _result(
                "power_w",
                "si",
                "op",
                status="measured",
                value=power,
                raw_value=power,
                postprocessed_value=power,
                confidence=0.95 if len(numeric_currents) == 1 else 0.88,
                provenance=["op_diagnostics:supply_current_aggregation"],
            )
        if "supply_current_a" in diagnostics:
            current = _numeric(diagnostics.get("supply_current_a"))
            if current is None:
                return _result(
                    "power_w",
                    "si",
                    "op",
                    status="indeterminate",
                    reason="power_unavailable",
                    detail="supply_current_not_numeric",
                    confidence=0.2,
                    provenance=["op_diagnostics"],
                )
            if supply_voltage is None:
                return _result(
                    "power_w",
                    "si",
                    "op",
                    status="indeterminate",
                    reason="power_supply_missing",
                    detail="missing_supply_voltage_for_single_supply_power",
                    confidence=0.2,
                    provenance=["op_diagnostics"],
                )
            power = abs(current) * supply_voltage
            return _result(
                "power_w",
                "si",
                "op",
                status="measured",
                value=power,
                raw_value=power,
                postprocessed_value=power,
                confidence=0.9,
                provenance=["op_diagnostics:supply_current"],
            )
    direct = _metric_from_direct_source("power_w", "op", op_output, confidence=0.76)
    if direct is not None:
        if direct.value == 0.0:
            return _result(
                "power_w",
                "si",
                "op",
                status="indeterminate",
                reason="power_unavailable",
                detail="zero_power_without_supply_context",
                confidence=0.2,
                provenance=["direct_metric_field"],
            )
        return direct
    return _result("power_w", "si", "op", status="indeterminate", reason="power_unavailable", detail="missing_power_source")


def _extract_output_swing(op_output: dict[str, object]) -> MeasurementResult:
    if not _analysis_success(op_output):
        return _result("output_swing_v", "si", "op", status="analysis_failed", reason="analysis_failure", detail="op_analysis_failed")
    diagnostics = op_output.get("op_diagnostics", {})
    if isinstance(diagnostics, dict):
        output_dc_v = _numeric(diagnostics.get("output_dc_v"))
        if output_dc_v is not None:
            return _result(
                "output_swing_v",
                "si",
                "op",
                status="measured",
                value=output_dc_v,
                raw_value=output_dc_v,
                postprocessed_value=output_dc_v,
                confidence=0.88,
                provenance=["op_diagnostics:output_dc_v"],
            )
    direct = _metric_from_direct_source("output_swing_v", "op", op_output, confidence=0.74)
    if direct is not None:
        return direct
    return _result(
        "output_swing_v",
        "si",
        "op",
        status="indeterminate",
        reason="measurement_failure",
        detail="missing_output_voltage_source",
        confidence=0.2,
    )


def _extract_slew_rate(tran_output: dict[str, object]) -> MeasurementResult:
    if not _analysis_success(tran_output):
        return _result("slew_rate_v_per_us", "si", "tran", status="analysis_failed", reason="analysis_failure", detail="tran_analysis_failed")
    direct = _metric_from_direct_source("slew_rate_v_per_us", "tran", tran_output, confidence=0.82)
    if direct is not None:
        return direct
    return _result(
        "slew_rate_v_per_us",
        "si",
        "tran",
        status="indeterminate",
        reason="no_metric_source",
        detail="missing_tran_slew_source",
        confidence=0.2,
    )


def extract_measurement_report(
    simulation_bundle: SimulationBundle,
    parsed_outputs: list[dict[str, object]],
    *,
    candidate_id: str,
) -> MeasurementReport:
    """Extract a structured measurement report from backend outputs."""

    outputs_by_analysis = {
        str(parsed.get("analysis_type", "unknown")): parsed
        for parsed in parsed_outputs
    }
    executed: list[AnalysisExecutionRecord] = []
    results: list[MeasurementResult] = []
    measured_metrics: list[MeasuredMetric] = []
    validation_checks: list[IntegrityCheckResult] = []

    for parsed in parsed_outputs:
        analysis_type = str(parsed.get("analysis_type", "unknown"))
        artifact_ref = str(parsed.get("artifact_ref", ""))
        executed.append(
            AnalysisExecutionRecord(
                analysis_type=analysis_type,
                success=_analysis_success(parsed),
                runtime_ms=int(parsed.get("runtime_ms", 0)),
                backend_status=str(parsed.get("status", "unknown")),
                artifact_refs=[artifact_ref] if artifact_ref else [],
            )
        )

    op_output = outputs_by_analysis.get("op", {})
    ac_output = outputs_by_analysis.get("ac", {})
    tran_output = outputs_by_analysis.get("tran", {})
    extractor_map = {
        "dc_gain_db": lambda: _extract_dc_gain(ac_output),
        "gbw_hz": lambda: _extract_gbw(ac_output),
        "phase_margin_deg": None,
        "power_w": lambda: _extract_power(op_output),
        "output_swing_v": lambda: _extract_output_swing(op_output),
        "slew_rate_v_per_us": lambda: _extract_slew_rate(tran_output),
    }

    gbw_result: MeasurementResult | None = None
    required_definitions = simulation_bundle.measurement_contract.measurement_definitions
    for definition in required_definitions:
        if definition.metric == "phase_margin_deg":
            if gbw_result is None:
                gbw_result = _extract_gbw(ac_output)
                if not any(item.metric == gbw_result.metric for item in results):
                    results.append(gbw_result)
            result = _extract_phase_margin(ac_output, gbw_result)
        else:
            extractor = extractor_map.get(definition.metric)
            result = extractor() if extractor is not None else _result(
                definition.metric,
                definition.units,
                definition.required_analysis_types[0] if definition.required_analysis_types else "unknown",
                status="missing",
                reason="no_metric_source",
                detail="no_formal_extractor_registered",
            )
            if definition.metric == "gbw_hz":
                gbw_result = result

        if not any(item.metric == result.metric for item in results):
            results.append(result)

    for result in results:
        if result.status.status == "measured" and result.value is not None:
            measured_metrics.append(
                MeasuredMetric(
                    metric=result.metric,
                    value=result.value,
                    units=result.units,
                    source_analysis=result.source_analysis,
                    extraction_confidence=result.confidence,
                )
            )
        validation_checks.append(
            IntegrityCheckResult(
                check_name=f"measurement::{result.metric}",
                passed=result.status.status == "measured",
                detail=f"{result.status.status}:{result.failure_reason.code}",
            )
        )

    return MeasurementReport(
        report_id=f"measure_{stable_hash(f'{simulation_bundle.simulation_id}|{candidate_id}')[:12]}",
        candidate_id=candidate_id,
        executed_analyses=executed,
        measurement_results=results,
        measured_metrics=measured_metrics,
        extraction_notes=[
            "measurement_contract_applied",
            "measurement_results_emitted_before_adjudication",
            "metrics_normalized_to_si_when_available",
        ],
        validation_checks=validation_checks,
    )
