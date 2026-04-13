"""Lightweight deterministic CMA-ES-style baseline helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from libs.eval.random_search import deterministic_unit
from libs.schema.design_task import DesignTask
from libs.schema.world_model import (
    FeasibilityPrediction,
    MetricEstimate,
    MetricsPrediction,
    SimulationValueEstimate,
    TrustAssessment,
)
from libs.world_model.feature_projection import evaluate_constraints


@dataclass(frozen=True)
class CmaesObservation:
    """One verified observation used by the lightweight CMA-ES-style baseline."""

    candidate_id: str
    parameter_values: dict[str, str | float | int | bool]
    measured_metrics: dict[str, float]
    feasible: bool
    utility: float


def _relevant_metrics(task: DesignTask) -> list[str]:
    metrics = {term.metric for term in task.objective.terms}
    metrics.update(constraint.metric for constraint in task.constraints.hard_constraints)
    metrics.update(constraint.metric for constraint in task.constraints.soft_constraints)
    metrics.update(task.objective.reporting_metrics)
    return sorted(metrics or {"dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w"})


def _metric_prior(metric: str, seed: str) -> float:
    ratio = deterministic_unit(f"{metric}|{seed}")
    ranges = {
        "dc_gain_db": (40.0, 110.0),
        "gbw_hz": (3.0e7, 1.8e8),
        "phase_margin_deg": (38.0, 82.0),
        "power_w": (7.0e-5, 2.2e-3),
        "output_swing_v": (0.45, 1.1),
        "psrr_db": (28.0, 85.0),
        "temperature_coefficient_ppm_per_c": (5.0, 120.0),
        "line_regulation_mv_per_v": (0.5, 40.0),
    }
    low, high = ranges.get(metric, (0.1, 1.0))
    return round(low + (high - low) * ratio, 6)


def _normalized_value(task: DesignTask, variable_name: str, value: str | float | int | bool) -> float:
    variable = next(item for item in task.design_space.variables if item.name == variable_name)
    if variable.kind in {"continuous", "integer"}:
        lower = float(variable.domain.lower)  # type: ignore[arg-type]
        upper = float(variable.domain.upper)  # type: ignore[arg-type]
        span = max(upper - lower, 1e-12)
        return max(0.0, min(1.0, (float(value) - lower) / span))
    choices = list(variable.domain.choices)
    if not choices:
        return 0.0
    try:
        index = choices.index(value)
    except ValueError:
        index = 0
    return index / max(1, len(choices) - 1)


def _denormalized_value(task: DesignTask, variable_name: str, normalized: float) -> str | float | int | bool:
    variable = next(item for item in task.design_space.variables if item.name == variable_name)
    normalized = max(0.0, min(1.0, normalized))
    if variable.kind in {"continuous", "integer"}:
        lower = float(variable.domain.lower)  # type: ignore[arg-type]
        upper = float(variable.domain.upper)  # type: ignore[arg-type]
        if variable.scale == "log" and lower > 0.0 and upper > 0.0:
            value = math.exp(math.log(lower) + (math.log(upper) - math.log(lower)) * normalized)
        else:
            value = lower + (upper - lower) * normalized
        if variable.kind == "integer" or variable.dtype == "int":
            return int(round(value))
        return round(float(value), 12)
    choices = list(variable.domain.choices)
    index = min(int(normalized * len(choices)), len(choices) - 1) if choices else 0
    choice = choices[index] if choices else variable.default
    if variable.dtype == "bool":
        return bool(choice)
    if variable.dtype == "int":
        return int(choice)
    if variable.dtype == "float":
        return float(choice)
    return choice


def _standard_normal(seed: str) -> float:
    u1 = max(deterministic_unit(f"{seed}|u1"), 1e-9)
    u2 = deterministic_unit(f"{seed}|u2")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _observation_utility(task: DesignTask, measured_metrics: dict[str, float], feasible: bool) -> float:
    score = 0.0
    for term in task.objective.terms:
        value = measured_metrics.get(term.metric)
        if value is None:
            continue
        normalized = math.log10(abs(value) + 1.0) / 10.0
        if term.direction == "minimize":
            normalized *= -1.0
        score += term.weight * normalized
    return round(score + (2.25 if feasible else -0.25), 6)


def build_observation(task: DesignTask, execution) -> CmaesObservation:
    """Build one CMA-ES observation from a real verification execution."""

    measured_metrics = {
        metric.metric: float(metric.value)
        for metric in execution.verification_result.measurement_report.measured_metrics
    }
    feasible = execution.verification_result.feasibility_status in {"feasible_nominal", "feasible_certified"}
    return CmaesObservation(
        candidate_id=execution.verification_result.candidate_id,
        parameter_values={parameter.variable_name: parameter.value_si for parameter in execution.simulation_bundle.netlist_instance.parameter_binding},
        measured_metrics=measured_metrics,
        feasible=feasible,
        utility=_observation_utility(task, measured_metrics, feasible),
    )


def _elite_observations(observations: list[CmaesObservation]) -> list[CmaesObservation]:
    ordered = sorted(observations, key=lambda item: (item.feasible, item.utility), reverse=True)
    elite_count = max(1, min(len(ordered), max(2, len(ordered) // 2)))
    return ordered[:elite_count]


def propose_parameter_batch(
    task: DesignTask,
    *,
    observations: list[CmaesObservation],
    run_label: str,
    step_index: int,
    population_size: int,
) -> list[dict[str, str | float | int | bool]]:
    """Generate a deterministic CMA-ES-style proposal population."""

    if not observations:
        from libs.eval.random_search import sample_parameter_values

        return [
            sample_parameter_values(task, run_label=run_label, step_index=step_index, sample_index=sample_index)
            for sample_index in range(max(4, population_size * 2))
        ]

    elite = _elite_observations(observations)
    means: dict[str, float] = {}
    sigmas: dict[str, float] = {}
    for variable in task.design_space.variables:
        normalized_values = [
            _normalized_value(task, variable.name, observation.parameter_values.get(variable.name, variable.default))
            for observation in elite
        ]
        mean = sum(normalized_values) / max(1, len(normalized_values))
        variance = sum((value - mean) ** 2 for value in normalized_values) / max(1, len(normalized_values))
        means[variable.name] = mean
        sigmas[variable.name] = max(0.06, min(0.28, math.sqrt(variance) + 0.05))

    proposals: list[dict[str, str | float | int | bool]] = []
    pool_size = max(6, population_size * 4)
    for sample_index in range(pool_size):
        values = dict(task.initial_state.template_defaults)
        for variable in task.design_space.variables:
            normalized = means[variable.name] + sigmas[variable.name] * _standard_normal(
                f"{run_label}|{step_index}|{sample_index}|{variable.name}"
            )
            values[variable.name] = _denormalized_value(task, variable.name, normalized)
        proposals.append(values)
    return proposals


def build_surrogate_predictions(
    task: DesignTask,
    state_id: str,
    parameter_values: dict[str, str | float | int | bool],
    *,
    observations: list[CmaesObservation],
    seed: str,
) -> tuple[MetricsPrediction, FeasibilityPrediction, SimulationValueEstimate, float]:
    """Build a deterministic CMA-ES-style surrogate prediction and ranking score."""

    relevant_metrics = _relevant_metrics(task)
    if not observations:
        trust = TrustAssessment(
            trust_level="low",
            service_tier="ranking_ready",
            confidence=0.18,
            uncertainty_score=0.9,
            ood_score=0.42,
            must_escalate=False,
            hard_block=False,
            reasons=["cmaes_baseline_cold_start"],
        )
        metric_map = {metric: _metric_prior(metric, seed) for metric in relevant_metrics}
        feasibility_constraints = [
            item.model_copy(update={"source": "prediction"})
            for item in evaluate_constraints(task, metric_map)
        ]
        metrics_prediction = MetricsPrediction(
            state_id=state_id,
            task_id=task.task_id,
            metrics=[
                MetricEstimate(
                    metric=metric,
                    value=value,
                    lower_bound=value * 0.94,
                    upper_bound=value * 1.06,
                    uncertainty=trust.uncertainty_score,
                    trust_level=trust.trust_level,
                    source="prediction",
                )
                for metric, value in sorted(metric_map.items())
            ],
            auxiliary_features={"surrogate_kind_code": 2.0, "observation_count": 0.0},
            trust_assessment=trust,
        )
        feasible_probability = 0.33
        feasibility_prediction = FeasibilityPrediction(
            state_id=state_id,
            task_id=task.task_id,
            overall_feasibility=feasible_probability,
            per_group_constraints=feasibility_constraints,
            most_likely_failure_reasons=[item.constraint_name for item in feasibility_constraints if item.margin < 0.0],
            confidence=trust.confidence,
            trust_assessment=trust,
        )
        sim_value = SimulationValueEstimate(
            state_id=state_id,
            estimated_value=0.78,
            decision="simulate",
            reasons=["cmaes_baseline_cold_start"],
            trust_assessment=trust,
        )
        return metrics_prediction, feasibility_prediction, sim_value, 0.78

    elite = _elite_observations(observations)
    metric_map: dict[str, float] = {}
    feasible_score = sum(1.0 if item.feasible else 0.0 for item in elite) / max(1, len(elite))
    utility_mean = sum(item.utility for item in elite) / max(1, len(elite))
    dispersion = 0.0

    for metric in relevant_metrics:
        values = [item.measured_metrics[metric] for item in elite if metric in item.measured_metrics]
        if not values:
            metric_map[metric] = _metric_prior(metric, seed)
            continue
        metric_map[metric] = round(sum(values) / len(values), 6)
        mean = metric_map[metric]
        dispersion += sum(abs(value - mean) for value in values) / max(1, len(values))

    uncertainty = round(max(0.08, min(0.85, dispersion / max(1.0, len(relevant_metrics) * 100.0))), 6)
    confidence = round(max(0.15, min(0.9, 1.0 - uncertainty * 0.9)), 6)
    trust = TrustAssessment(
        trust_level="medium" if confidence >= 0.45 else "low",
        service_tier="ranking_ready",
        confidence=confidence,
        uncertainty_score=uncertainty,
        ood_score=round(min(1.0, uncertainty * 0.7 + 0.1), 6),
        must_escalate=False,
        hard_block=False,
        reasons=["cmaes_baseline_surrogate"],
    )
    constraints = [
        item.model_copy(update={"source": "prediction"})
        for item in evaluate_constraints(task, metric_map)
    ]
    metrics_prediction = MetricsPrediction(
        state_id=state_id,
        task_id=task.task_id,
        metrics=[
            MetricEstimate(
                metric=metric,
                value=value,
                lower_bound=value * 0.96,
                upper_bound=value * 1.04,
                uncertainty=trust.uncertainty_score,
                trust_level=trust.trust_level,
                source="prediction",
            )
            for metric, value in sorted(metric_map.items())
        ],
        auxiliary_features={"surrogate_kind_code": 2.0, "observation_count": float(len(observations))},
        trust_assessment=trust,
    )
    feasibility_prediction = FeasibilityPrediction(
        state_id=state_id,
        task_id=task.task_id,
        overall_feasibility=round(max(0.05, min(0.95, feasible_score * confidence + 0.1)), 6),
        per_group_constraints=constraints,
        most_likely_failure_reasons=[item.constraint_name for item in constraints if item.margin < 0.0],
        confidence=confidence,
        trust_assessment=trust,
    )
    estimated_value = round(max(0.2, min(0.95, 0.45 + utility_mean * 0.1 + feasible_score * 0.25)), 6)
    simulation_value = SimulationValueEstimate(
        state_id=state_id,
        estimated_value=estimated_value,
        decision="simulate" if estimated_value >= 0.45 else "defer",
        reasons=["cmaes_baseline"],
        trust_assessment=trust,
    )
    return metrics_prediction, feasibility_prediction, simulation_value, estimated_value
