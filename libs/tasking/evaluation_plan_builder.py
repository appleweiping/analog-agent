"""Evaluation-plan construction for formalized design tasks."""

from __future__ import annotations

from dataclasses import dataclass

from libs.schema.design_spec import DesignSpec
from libs.schema.design_task import (
    AnalysisConfig,
    AnalysisSpec,
    ConditionPolicy,
    EvaluationPlan,
    MetricExtractor,
    ObjectiveSpec,
    StopCondition,
)
from libs.tasking.constraint_resolver import stage_for_metric

ANALYSIS_ORDER = ("op", "ac", "tran", "noise", "pvt_sweep", "monte_carlo")
ANALYSIS_COST = {
    "op": "cheap",
    "ac": "cheap",
    "tran": "moderate",
    "noise": "moderate",
    "pvt_sweep": "expensive",
    "monte_carlo": "expensive",
}

METRIC_EXTRACTION_METHODS = {
    "dc_gain_db": "ac_gain_at_low_frequency",
    "gbw_hz": "unity_gain_frequency",
    "phase_margin_deg": "phase_margin_at_unity_gain",
    "slew_rate_v_per_us": "max_output_derivative",
    "power_w": "supply_current_times_voltage",
    "area_um2": "device_geometry_accumulation",
    "noise_nv_per_sqrt_hz": "spot_noise_at_frequency",
    "input_referred_noise_nv_per_sqrt_hz": "input_referred_noise_projection",
    "output_swing_v": "steady_state_output_swing",
    "input_common_mode_v": "valid_input_common_mode_window",
}


@dataclass(frozen=True)
class EvaluationPlanCompilation:
    """Evaluation plan plus provenance and safe repairs."""

    evaluation_plan: EvaluationPlan
    derived_fields: list[str]
    repair_history: list[str]


def _ordered(values: set[str]) -> list[str]:
    order = {item: index for index, item in enumerate(ANALYSIS_ORDER)}
    return sorted(values, key=lambda item: (order.get(item, len(order)), item))


def _analysis_config(analysis_type: str, spec: DesignSpec) -> AnalysisConfig:
    parameters: dict[str, float | int | str | bool] = {}
    if spec.supply_voltage_v is not None:
        parameters["supply_voltage_v"] = spec.supply_voltage_v
    if spec.environment.load_cap_f is not None:
        parameters["load_cap_f"] = spec.environment.load_cap_f
    if spec.environment.output_load_ohm is not None:
        parameters["output_load_ohm"] = spec.environment.output_load_ohm
    if analysis_type == "ac":
        parameters.update({"f_start_hz": 1.0, "f_stop_hz": 1e10, "points_per_dec": 20})
    if analysis_type == "tran":
        parameters.update({"t_stop_s": 10e-6, "time_step_s": 50e-9})
    if analysis_type == "noise":
        parameters.update({"f_start_hz": 10.0, "f_stop_hz": 1e7})
    if analysis_type == "pvt_sweep":
        parameters.update(
            {
                "corner_count": len(spec.environment.corners),
                "temperature_count": len(spec.environment.temperature_c),
            }
        )
    return AnalysisConfig(parameters=parameters)


def _condition_policy(values: list[float | str]) -> ConditionPolicy:
    if not values:
        return ConditionPolicy(mode="inherit", values=[])
    if len(values) == 1:
        return ConditionPolicy(mode="fixed", values=values)
    return ConditionPolicy(mode="sweep", values=values)


def _simulation_budget(analyses: list[str], corners_count: int, temperature_count: int) -> str:
    factor = len(analyses) * max(corners_count, 1) * max(temperature_count, 1)
    if factor <= 3:
        return "cheap"
    if factor <= 9:
        return "moderate"
    return "expensive"


def build_evaluation_plan(
    spec: DesignSpec,
    objective: ObjectiveSpec,
    constraint_metrics: list[str],
) -> EvaluationPlanCompilation:
    """Build a simulator-facing evaluation plan from a DesignSpec and compiled objectives."""

    requested_analyses = set(spec.testbench_plan)
    required_metrics = set(objective.reporting_metrics) | set(constraint_metrics)
    repair_history: list[str] = []

    for metric in required_metrics:
        requested_analyses.add(stage_for_metric(metric))

    if len(spec.environment.corners) > 1 or len(spec.environment.temperature_c) > 1:
        requested_analyses.add("pvt_sweep")

    ordered_analyses = _ordered(requested_analyses)
    original_plan = set(spec.testbench_plan)
    missing_coverage = [analysis for analysis in ordered_analyses if analysis not in original_plan]
    if missing_coverage:
        repair_history.append(f"augmented evaluation coverage with analyses {missing_coverage}")

    metrics_by_analysis = {analysis: [] for analysis in ordered_analyses}
    for metric in sorted(required_metrics):
        analysis_type = stage_for_metric(metric)
        if analysis_type in metrics_by_analysis:
            metrics_by_analysis[analysis_type].append(metric)

    analyses = [
        AnalysisSpec(
            analysis_type=analysis_type,
            order=index,
            config=_analysis_config(analysis_type, spec),
            required_metrics=metrics_by_analysis.get(analysis_type, []),
            estimated_cost=ANALYSIS_COST[analysis_type],
        )
        for index, analysis_type in enumerate(ordered_analyses)
    ]
    metric_extractors = [
        MetricExtractor(
            metric=metric,
            from_analysis=stage_for_metric(metric),
            method=METRIC_EXTRACTION_METHODS.get(metric, "direct_metric_lookup"),
        )
        for metric in sorted(required_metrics)
    ]

    corners_policy = _condition_policy(list(spec.environment.corners))
    temperature_policy = _condition_policy(list(spec.environment.temperature_c))
    load_values: list[float | str] = []
    if spec.environment.load_cap_f is not None:
        load_values.append(spec.environment.load_cap_f)
    if spec.environment.output_load_ohm is not None:
        load_values.append(spec.environment.output_load_ohm)
    load_policy = _condition_policy(load_values)

    budget_class = _simulation_budget(
        ordered_analyses,
        len(spec.environment.corners),
        len(spec.environment.temperature_c),
    )
    fidelity_policy = "staged_fidelity" if budget_class == "expensive" or "pvt_sweep" in ordered_analyses else "single_fidelity"
    stop_conditions = [StopCondition(trigger="op_failure", action="stop")]
    if "tran" in ordered_analyses:
        stop_conditions.append(StopCondition(trigger="phase_margin_violation_before_tran", action="fallback"))

    return EvaluationPlanCompilation(
        evaluation_plan=EvaluationPlan(
            analyses=analyses,
            metric_extractors=metric_extractors,
            corners_policy=corners_policy,
            temperature_policy=temperature_policy,
            load_policy=load_policy,
            simulation_budget_class=budget_class,
            fidelity_policy=fidelity_policy,
            stop_conditions=stop_conditions,
        ),
        derived_fields=["evaluation_plan"],
        repair_history=repair_history,
    )
