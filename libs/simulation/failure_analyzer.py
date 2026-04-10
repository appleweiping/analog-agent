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
    if completion_status in {"simulator_failure", "timeout"}:
        return FailureAttribution(
            primary_failure_class="simulation_invalid",
            contributing_factors=["backend_execution_failed"],
            evidence=["backend_status!=ok"],
            recommended_focus=["retry_backend", "switch_backend"],
        )
    analysis_failures = [result for result in measurement_report.measurement_results if result.status.status == "analysis_failed"]
    extraction_failures = [
        result
        for result in measurement_report.measurement_results
        if result.status.status in {"extraction_failed", "indeterminate", "missing"}
    ]
    if analysis_failures:
        return FailureAttribution(
            primary_failure_class="analysis_failure",
            contributing_factors=[item.metric for item in analysis_failures],
            evidence=[f"{item.metric}:{item.failure_reason.code}" for item in analysis_failures],
            recommended_focus=["inspect_analysis_plan", "retry_truth_verification"],
        )
    if extraction_failures and not measurement_report.measured_metrics:
        return FailureAttribution(
            primary_failure_class="measurement_failure",
            contributing_factors=[item.metric for item in extraction_failures],
            evidence=[f"{item.metric}:{item.failure_reason.code}" for item in extraction_failures],
            recommended_focus=["inspect_measurement_contract", "inspect_raw_artifacts"],
        )
    measurement_side_failures = [assessment for assessment in assessments if assessment.assessment_basis == "measurement_unavailable"]
    if measurement_side_failures:
        return FailureAttribution(
            primary_failure_class="measurement_failure",
            contributing_factors=[assessment.metric for assessment in measurement_side_failures],
            evidence=[f"{assessment.metric}:{assessment.measurement_failure_reason}" for assessment in measurement_side_failures],
            recommended_focus=["retry_measurement_extraction", "escalate_truth_fidelity"],
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
    if failing:
        return FailureAttribution(
            primary_failure_class="design_failure",
            contributing_factors=[assessment.constraint_name for assessment in failing],
            evidence=[f"{assessment.metric}_margin={assessment.margin}" for assessment in failing],
            recommended_focus=["explore_candidate_neighborhood", "update_planner_priority"],
        )
    return FailureAttribution(
        primary_failure_class="none",
        contributing_factors=[],
        evidence=["all_constraints_satisfied"],
        recommended_focus=["candidate_ready_for_next_phase"],
    )
