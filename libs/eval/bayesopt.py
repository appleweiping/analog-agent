"""Lightweight deterministic BayesOpt-style baseline helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from libs.eval.random_search import deterministic_unit, sample_parameter_values
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
class BayesOptObservation:
    """One verified observation used by the lightweight baseline surrogate."""

    candidate_id: str
    parameter_values: dict[str, str | float | int | bool]
    measured_metrics: dict[str, float]
    feasible: bool
    utility: float


def _metric_prior(metric: str, seed: str) -> float:
    ratio = deterministic_unit(f"{metric}|{seed}")
    ranges = {
        "dc_gain_db": (45.0, 120.0),
        "gbw_hz": (4.0e7, 2.0e8),
        "phase_margin_deg": (40.0, 85.0),
        "power_w": (8.0e-5, 2.0e-3),
        "output_swing_v": (0.5, 1.1),
        "psrr_db": (30.0, 90.0),
        "temperature_coefficient_ppm_per_c": (5.0, 120.0),
        "line_regulation_mv_per_v": (0.5, 40.0),
    }
    low, high = ranges.get(metric, (0.1, 1.0))
    return round(low + (high - low) * ratio, 6)


def _relevant_metrics(task: DesignTask) -> list[str]:
    metrics = {term.metric for term in task.objective.terms}
    metrics.update(constraint.metric for constraint in task.constraints.hard_constraints)
    metrics.update(constraint.metric for constraint in task.constraints.soft_constraints)
    metrics.update(task.objective.reporting_metrics)
    return sorted(metrics or {"dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w"})


def _normalized_vector(task: DesignTask, values: dict[str, str | float | int | bool]) -> list[float]:
    vector: list[float] = []
    for variable in task.design_space.variables:
        value = values.get(variable.name, variable.default)
        if variable.kind in {"continuous", "integer"}:
            lower = float(variable.domain.lower)  # type: ignore[arg-type]
            upper = float(variable.domain.upper)  # type: ignore[arg-type]
            current = float(value) if value is not None else lower
            span = max(upper - lower, 1e-12)
            vector.append((current - lower) / span)
        else:
            choices = list(variable.domain.choices)
            if not choices:
                vector.append(0.0)
                continue
            try:
                index = choices.index(value)
            except ValueError:
                index = 0
            vector.append(index / max(1, len(choices) - 1))
    return vector


def _distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 1.0
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)) / len(a))


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
    return round(score + (2.0 if feasible else -0.5), 6)


def build_observation(task: DesignTask, execution) -> BayesOptObservation:
    """Build one lightweight surrogate observation from a real verification execution."""

    measured_metrics = {
        metric.metric: float(metric.value)
        for metric in execution.verification_result.measurement_report.measured_metrics
    }
    feasible = execution.verification_result.feasibility_status in {"feasible_nominal", "feasible_certified"}
    return BayesOptObservation(
        candidate_id=execution.verification_result.candidate_id,
        parameter_values={parameter.variable_name: parameter.value_si for parameter in execution.simulation_bundle.netlist_instance.parameter_binding},
        measured_metrics=measured_metrics,
        feasible=feasible,
        utility=_observation_utility(task, measured_metrics, feasible),
    )


def propose_parameter_batch(
    task: DesignTask,
    *,
    run_label: str,
    step_index: int,
    batch_size: int,
    pool_multiplier: int = 4,
) -> list[dict[str, str | float | int | bool]]:
    """Generate a deterministic proposal pool for one BayesOpt baseline step."""

    pool_size = max(batch_size * pool_multiplier, 8)
    return [
        sample_parameter_values(task, run_label=run_label, step_index=step_index, sample_index=sample_index)
        for sample_index in range(pool_size)
    ]


def build_surrogate_predictions(
    task: DesignTask,
    state_id: str,
    parameter_values: dict[str, str | float | int | bool],
    *,
    observations: list[BayesOptObservation],
    seed: str,
) -> tuple[MetricsPrediction, FeasibilityPrediction, SimulationValueEstimate, float]:
    """Build a deterministic BayesOpt-style surrogate prediction and acquisition score."""

    candidate_vector = _normalized_vector(task, parameter_values)
    relevant_metrics = _relevant_metrics(task)
    if not observations:
        trust = TrustAssessment(
            trust_level="low",
            service_tier="ranking_ready",
            confidence=0.15,
            uncertainty_score=0.95,
            ood_score=0.45,
            must_escalate=False,
            hard_block=False,
            reasons=["bayesopt_baseline_cold_start"],
        )
        metric_map = {metric: _metric_prior(metric, seed) for metric in relevant_metrics}
        feasibility_constraints = [
            item.model_copy(update={"source": "prediction"})
            for item in evaluate_constraints(task, metric_map)
        ]
        overall = 0.35
        metrics_prediction = MetricsPrediction(
            state_id=state_id,
            task_id=task.task_id,
            metrics=[
                MetricEstimate(
                    metric=metric,
                    value=value,
                    lower_bound=value * 0.95,
                    upper_bound=value * 1.05,
                    uncertainty=trust.uncertainty_score,
                    trust_level=trust.trust_level,
                    source="prediction",
                )
                for metric, value in sorted(metric_map.items())
            ],
            auxiliary_features={"surrogate_kind_code": 0.0, "observation_count": 0.0},
            trust_assessment=trust,
        )
        feasibility_prediction = FeasibilityPrediction(
            state_id=state_id,
            task_id=task.task_id,
            overall_feasibility=overall,
            per_group_constraints=feasibility_constraints,
            most_likely_failure_reasons=[item.constraint_name for item in feasibility_constraints if item.margin < 0.0],
            confidence=trust.confidence,
            trust_assessment=trust,
        )
        simulation_value = SimulationValueEstimate(
            state_id=state_id,
            estimated_value=0.85,
            decision="simulate",
            reasons=["bayesopt_baseline_cold_start"],
            trust_assessment=trust,
        )
        return metrics_prediction, feasibility_prediction, simulation_value, 0.85

    weighted_metrics: dict[str, list[tuple[float, float]]] = {metric: [] for metric in relevant_metrics}
    utilities: list[tuple[float, float]] = []
    feasible_weights: list[tuple[float, float]] = []
    max_kernel = 0.0
    for observation in observations:
        observation_vector = _normalized_vector(task, observation.parameter_values)
        distance = _distance(candidate_vector, observation_vector)
        kernel = math.exp(-(distance**2) / (2 * (0.35**2)))
        max_kernel = max(max_kernel, kernel)
        utilities.append((kernel, observation.utility))
        feasible_weights.append((kernel, 1.0 if observation.feasible else 0.0))
        for metric in relevant_metrics:
            if metric in observation.measured_metrics:
                weighted_metrics[metric].append((kernel, observation.measured_metrics[metric]))

    uncertainty = round(max(0.05, min(1.0, 1.0 - max_kernel)), 6)
    confidence = round(max(0.1, min(0.95, 1.0 - uncertainty * 0.8)), 6)
    trust = TrustAssessment(
        trust_level="medium" if confidence >= 0.45 else "low",
        service_tier="ranking_ready",
        confidence=confidence,
        uncertainty_score=uncertainty,
        ood_score=round(uncertainty * 0.75, 6),
        must_escalate=False,
        hard_block=False,
        reasons=["bayesopt_baseline_surrogate"],
    )

    metric_map: dict[str, float] = {}
    for metric in relevant_metrics:
        pairs = weighted_metrics[metric]
        if not pairs:
            metric_map[metric] = _metric_prior(metric, seed)
            continue
        total_weight = sum(weight for weight, _ in pairs)
        metric_map[metric] = round(sum(weight * value for weight, value in pairs) / max(total_weight, 1e-12), 6)

    feasibility_constraints = [
        item.model_copy(update={"source": "prediction"})
        for item in evaluate_constraints(task, metric_map)
    ]
    feasible_weight = sum(weight for weight, _ in feasible_weights)
    feasible_probability = (
        sum(weight * value for weight, value in feasible_weights) / max(feasible_weight, 1e-12)
        if feasible_weights
        else 0.0
    )
    overall = round(max(0.0, min(1.0, 0.6 * feasible_probability + 0.4 * (1.0 - uncertainty))), 6)

    total_utility_weight = sum(weight for weight, _ in utilities)
    utility_mean = sum(weight * value for weight, value in utilities) / max(total_utility_weight, 1e-12)
    normalized_utility = 1.0 / (1.0 + math.exp(-utility_mean))
    acquisition = round(max(0.0, min(1.0, 0.7 * normalized_utility + 0.3 * uncertainty)), 6)

    metrics_prediction = MetricsPrediction(
        state_id=state_id,
        task_id=task.task_id,
        metrics=[
            MetricEstimate(
                metric=metric,
                value=value,
                lower_bound=value * (1.0 - min(0.2, uncertainty)),
                upper_bound=value * (1.0 + min(0.2, uncertainty)),
                uncertainty=uncertainty,
                trust_level=trust.trust_level,
                source="prediction",
            )
            for metric, value in sorted(metric_map.items())
        ],
        auxiliary_features={"surrogate_kind_code": 1.0, "observation_count": float(len(observations))},
        trust_assessment=trust,
    )
    feasibility_prediction = FeasibilityPrediction(
        state_id=state_id,
        task_id=task.task_id,
        overall_feasibility=overall,
        per_group_constraints=feasibility_constraints,
        most_likely_failure_reasons=[item.constraint_name for item in feasibility_constraints if item.margin < 0.0],
        confidence=confidence,
        trust_assessment=trust,
    )
    simulation_value = SimulationValueEstimate(
        state_id=state_id,
        estimated_value=acquisition,
        decision="simulate" if acquisition >= 0.45 else "defer",
        reasons=["bayesopt_acquisition"],
        trust_assessment=trust,
    )
    return metrics_prediction, feasibility_prediction, simulation_value, acquisition
