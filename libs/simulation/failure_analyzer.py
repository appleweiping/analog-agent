"""Failure attribution for the fifth layer."""

from __future__ import annotations

from libs.schema.simulation import ConstraintAssessmentRecord, FailureAttribution, MeasurementReport, RobustnessCertificate


def attribute_failures(
    assessments: list[ConstraintAssessmentRecord],
    measurement_report: MeasurementReport,
    robustness: RobustnessCertificate,
    *,
    render_ready: bool,
    completion_status: str,
) -> FailureAttribution:
    """Produce a structured failure attribution."""

    if not render_ready:
        return FailureAttribution(
            primary_failure_class="netlist_failure",
            contributing_factors=["netlist_integrity_check_failed"],
            evidence=["render_status=invalid"],
            recommended_focus=["fix_parameter_binding", "fix_topology_template"],
        )
    if completion_status == "simulator_failure":
        return FailureAttribution(
            primary_failure_class="simulator_failure",
            contributing_factors=["backend_execution_failed"],
            evidence=["backend_status!=ok"],
            recommended_focus=["retry_backend", "switch_backend"],
        )
    if not measurement_report.measured_metrics:
        return FailureAttribution(
            primary_failure_class="measurement_failure",
            contributing_factors=["measurement_contract_unfulfilled"],
            evidence=["no_metrics_extracted"],
            recommended_focus=["add_extractor", "inspect_raw_artifacts"],
        )

    failing = [assessment for assessment in assessments if not assessment.is_satisfied]
    if robustness.certification_status == "robustness_failed":
        return FailureAttribution(
            primary_failure_class="robustness_failure",
            contributing_factors=["corner_or_sweep_violation"],
            evidence=[f"weakest_condition={robustness.weakest_condition or 'unknown'}"],
            recommended_focus=["tighten_nominal_margin", "inspect_corner_sensitivity"],
        )
    for assessment in failing:
        metric = assessment.metric
        if metric in {"phase_margin_deg", "dc_gain_db"}:
            return FailureAttribution(
                primary_failure_class="stability_failure",
                contributing_factors=[assessment.constraint_name],
                evidence=[f"{metric}_margin={assessment.margin}"],
                recommended_focus=["compensation_network", "bias_stability_margin"],
            )
        if metric in {"gbw_hz", "slew_rate_v_per_us", "delay_ns"}:
            return FailureAttribution(
                primary_failure_class="drive_bandwidth_failure",
                contributing_factors=[assessment.constraint_name],
                evidence=[f"{metric}_margin={assessment.margin}"],
                recommended_focus=["increase_drive", "rebalance_compensation"],
            )
        if metric in {"noise_nv_per_sqrt_hz", "input_referred_noise_nv_per_sqrt_hz", "power_w", "area_um2"}:
            return FailureAttribution(
                primary_failure_class="noise_power_area_tradeoff_failure",
                contributing_factors=[assessment.constraint_name],
                evidence=[f"{metric}_margin={assessment.margin}"],
                recommended_focus=["resize_input_devices", "rebalance_power_noise_tradeoff"],
            )
    return FailureAttribution(
        primary_failure_class="none",
        contributing_factors=[],
        evidence=["all_constraints_satisfied"],
        recommended_focus=["candidate_ready_for_next_phase"],
    )
