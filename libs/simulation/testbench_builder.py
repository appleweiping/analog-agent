"""Analysis-plan and measurement-contract builders for the fifth layer."""

from __future__ import annotations

from libs.schema.design_task import AnalysisSpec, DesignTask
from libs.schema.simulation import (
    ANALYSIS_TYPES,
    AnalysisPlan,
    AnalysisStatement,
    ExtractionMethod,
    FallbackStrategy,
    MeasurementContract,
    MeasurementDefinition,
    PostprocessingRule,
    SimulationRequest,
    ValidationCheck,
)

def _analysis_statement(spec: AnalysisSpec) -> AnalysisStatement:
    return AnalysisStatement(
        analysis_type=spec.analysis_type,
        order=spec.order,
        parameters=dict(spec.config.parameters),
        required_metrics=list(spec.required_metrics),
    )


def build_analysis_plan(task: DesignTask, request: SimulationRequest) -> AnalysisPlan:
    """Build a structured analysis plan from DesignTask.evaluation_plan."""

    base = [_analysis_statement(spec) for spec in sorted(task.evaluation_plan.analyses, key=lambda item: item.order)]
    if request.analysis_scope:
        base = [analysis for analysis in base if analysis.analysis_type in request.analysis_scope]
    if not base:
        base = [_analysis_statement(spec) for spec in sorted(task.evaluation_plan.analyses, key=lambda item: item.order)]

    existing = {analysis.analysis_type for analysis in base}
    extras: list[AnalysisStatement] = []
    real_ota_native = request.backend_preference == "ngspice" and task.circuit_family == "two_stage_ota" and task.topology.topology_mode == "fixed"
    if real_ota_native and request.fidelity_level in {"quick_truth", "focused_validation"}:
        ordered = []
        for analysis in base:
            if analysis.analysis_type not in {"op", "ac"}:
                continue
            required = list(analysis.required_metrics)
            if analysis.analysis_type == "ac" and "dc_gain_db" not in required:
                required = ["dc_gain_db", *required]
            ordered.append(analysis.model_copy(update={"required_metrics": required}))
        execution_policy = "serial"
        termination_rules = ["stop_on_netlist_failure", "stop_on_simulation_error", "stop_after_core_truth_violation"]
    elif request.fidelity_level == "quick_truth":
        ordered = [analysis for analysis in base if analysis.analysis_type in {"op", "ac", "tran"}][:2]
        execution_policy = "serial"
        termination_rules = ["stop_on_critical_integrity_failure", "stop_after_first_core_constraint_violation"]
    elif request.fidelity_level == "focused_validation":
        ordered = [analysis for analysis in base if analysis.analysis_type in {"op", "ac", "tran", "noise"}]
        if "tran" not in existing:
            extras.append(AnalysisStatement(analysis_type="tran", order=len(ordered), parameters={"purpose": "focused_validation"}, required_metrics=["slew_rate_v_per_us"]))
        execution_policy = "phase_gated"
        termination_rules = ["stop_on_core_constraint_failure"]
    elif request.fidelity_level == "targeted_failure_analysis":
        ordered = [analysis for analysis in base if analysis.analysis_type in {"op", "ac", "tran", "noise"}]
        if "noise" not in existing:
            extras.append(AnalysisStatement(analysis_type="noise", order=len(ordered), parameters={"purpose": "failure_analysis"}, required_metrics=["noise_nv_per_sqrt_hz"]))
        execution_policy = "serial"
        termination_rules = ["continue_to_collect_diagnostics"]
    else:
        ordered = list(base)
        if "pvt_sweep" not in existing:
            extras.append(AnalysisStatement(analysis_type="pvt_sweep", order=98, parameters={"corner_count": len(task.evaluation_plan.corners_policy.values)}, required_metrics=["gbw_hz", "phase_margin_deg", "power_w"]))
        if "temperature_sweep" not in existing:
            extras.append(AnalysisStatement(analysis_type="temperature_sweep", order=99, parameters={"temperature_count": len(task.evaluation_plan.temperature_policy.values)}, required_metrics=["offset_mv", "temperature_coefficient_ppm_per_c"]))
        if "load_sweep" not in existing:
            extras.append(AnalysisStatement(analysis_type="load_sweep", order=100, parameters={"load_count": len(task.evaluation_plan.load_policy.values)}, required_metrics=["load_regulation_mv_per_ma", "phase_margin_deg"]))
        if "monte_carlo" not in existing:
            extras.append(AnalysisStatement(analysis_type="monte_carlo", order=101, parameters={"samples": 16}, required_metrics=["offset_mv", "gbw_hz"]))
        execution_policy = "phase_gated"
        termination_rules = ["stop_on_netlist_failure_only"]
    ordered = sorted([*ordered, *extras], key=lambda item: (item.order, ANALYSIS_TYPES.index(item.analysis_type)))

    dependencies: dict[str, list[str]] = {}
    for analysis in ordered:
        deps: list[str] = []
        if analysis.analysis_type in {"ac", "tran", "noise"}:
            deps.append("op")
        if analysis.analysis_type in {"pvt_sweep", "load_sweep", "temperature_sweep", "monte_carlo"}:
            deps.extend(["op", "ac"])
        dependencies[analysis.analysis_type] = deps

    return AnalysisPlan(
        ordered_analyses=ordered,
        analysis_dependencies=dependencies,
        fidelity_level=request.fidelity_level,
        execution_policy=execution_policy,
        early_termination_rules=termination_rules,
    )


def build_measurement_contract(task: DesignTask, analysis_plan: AnalysisPlan) -> MeasurementContract:
    """Build a formal measurement extraction contract."""

    definitions: dict[str, MeasurementDefinition] = {}
    methods: list[ExtractionMethod] = []
    active_analyses = {analysis.analysis_type for analysis in analysis_plan.ordered_analyses}
    for extractor in task.evaluation_plan.metric_extractors:
        if extractor.from_analysis not in active_analyses:
            continue
        role = "power" if extractor.metric == "power_w" else "stability" if extractor.metric in {"dc_gain_db", "gbw_hz", "phase_margin_deg"} else "performance"
        definitions.setdefault(
            extractor.metric,
            MeasurementDefinition(
                metric=extractor.metric,
                units="si",
                required_analysis_types=[extractor.from_analysis],
                semantic_role=role,
                expected_range=[],
            ),
        )
        methods.append(
            ExtractionMethod(
                metric=extractor.metric,
                method=extractor.method,
                from_analysis=extractor.from_analysis,
                preferred_source_field="metrics",
                failure_conditions=["analysis_failure", "measurement_failure"],
            )
        )
    for analysis in analysis_plan.ordered_analyses:
        for metric in analysis.required_metrics:
            role = "power" if metric == "power_w" else "stability" if metric in {"dc_gain_db", "gbw_hz", "phase_margin_deg"} else "performance"
            definitions.setdefault(
                metric,
                MeasurementDefinition(
                    metric=metric,
                    units="si",
                    required_analysis_types=[analysis.analysis_type],
                    semantic_role=role,
                    expected_range=[],
                ),
            )
            if not any(method.metric == metric and method.from_analysis == analysis.analysis_type for method in methods):
                methods.append(
                    ExtractionMethod(
                        metric=metric,
                        method="direct",
                        from_analysis=analysis.analysis_type,
                        preferred_source_field="metrics",
                        failure_conditions=["analysis_failure", "no_metric_source"],
                    )
                )
    return MeasurementContract(
        measurement_definitions=list(definitions.values()),
        extraction_methods=methods,
        postprocessing_rules=[
            PostprocessingRule(rule_name="normalize_to_si_units", applies_to_metrics=list(definitions.keys())),
            PostprocessingRule(rule_name="prefer_highest_confidence_observation", applies_to_metrics=list(definitions.keys())),
        ],
        fallback_strategies=[
            FallbackStrategy(
                strategy_name="flag_missing_metric",
                applies_to_metrics=list(definitions.keys()),
                trigger_condition="metric_not_extractable",
                action="emit_structured_measurement_failure",
            )
        ],
        validation_checks=[
            ValidationCheck(check_name="all_required_metrics_accounted_for", applies_to_metrics=list(definitions.keys()), failure_severity="critical"),
            ValidationCheck(check_name="analysis_sources_resolved", applies_to_metrics=list(definitions.keys()), failure_severity="high"),
            ValidationCheck(check_name="no_metric_conflict", applies_to_metrics=list(definitions.keys()), failure_severity="medium"),
        ],
    )
