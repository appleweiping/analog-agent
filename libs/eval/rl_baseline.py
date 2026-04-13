"""Lightweight deterministic RL-style baseline helpers."""

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
class RlObservation:
    """One verified observation used by the lightweight RL-style baseline."""

    candidate_id: str
    parameter_values: dict[str, str | float | int | bool]
    measured_metrics: dict[str, float]
    feasible: bool
    reward: float


def _relevant_metrics(task: DesignTask) -> list[str]:
    metrics = {term.metric for term in task.objective.terms}
    metrics.update(constraint.metric for constraint in task.constraints.hard_constraints)
    metrics.update(constraint.metric for constraint in task.constraints.soft_constraints)
    metrics.update(task.objective.reporting_metrics)
    return sorted(metrics or {"dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w"})


def _metric_prior(metric: str, seed: str) -> float:
    ratio = deterministic_unit(f"{metric}|{seed}")
    ranges = {
        "dc_gain_db": (42.0, 115.0),
        "gbw_hz": (3.5e7, 1.9e8),
        "phase_margin_deg": (40.0, 84.0),
        "power_w": (7.5e-5, 2.0e-3),
        "output_swing_v": (0.45, 1.1),
        "psrr_db": (30.0, 88.0),
        "temperature_coefficient_ppm_per_c": (5.0, 120.0),
        "line_regulation_mv_per_v": (0.5, 40.0),
    }
    low, high = ranges.get(metric, (0.1, 1.0))
    return round(low + (high - low) * ratio, 6)


def _normalize(task: DesignTask, variable_name: str, value: str | float | int | bool) -> float:
    variable = next(item for item in task.design_space.variables if item.name == variable_name)
    if variable.kind in {"continuous", "integer"}:
        lower = float(variable.domain.lower)  # type: ignore[arg-type]
        upper = float(variable.domain.upper)  # type: ignore[arg-type]
        return max(0.0, min(1.0, (float(value) - lower) / max(upper - lower, 1e-12)))
    choices = list(variable.domain.choices)
    if not choices:
        return 0.0
    try:
        index = choices.index(value)
    except ValueError:
        index = 0
    return index / max(1, len(choices) - 1)


def _denormalize(task: DesignTask, variable_name: str, normalized: float) -> str | float | int | bool:
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


def _reward(task: DesignTask, measured_metrics: dict[str, float], feasible: bool) -> float:
    reward = 0.0
    for term in task.objective.terms:
        value = measured_metrics.get(term.metric)
        if value is None:
            continue
        normalized = math.log10(abs(value) + 1.0) / 10.0
        if term.direction == "minimize":
            normalized *= -1.0
        reward += term.weight * normalized
    return round(reward + (2.0 if feasible else -0.35), 6)


def build_observation(task: DesignTask, execution) -> RlObservation:
    """Build one RL observation from a real verification execution."""

    measured_metrics = {
        metric.metric: float(metric.value)
        for metric in execution.verification_result.measurement_report.measured_metrics
    }
    feasible = execution.verification_result.feasibility_status in {"feasible_nominal", "feasible_certified"}
    return RlObservation(
        candidate_id=execution.verification_result.candidate_id,
        parameter_values={parameter.variable_name: parameter.value_si for parameter in execution.simulation_bundle.netlist_instance.parameter_binding},
        measured_metrics=measured_metrics,
        feasible=feasible,
        reward=_reward(task, measured_metrics, feasible),
    )


def _elite(observations: list[RlObservation]) -> list[RlObservation]:
    ordered = sorted(observations, key=lambda item: (item.feasible, item.reward), reverse=True)
    return ordered[: max(1, min(len(ordered), max(2, len(ordered) // 2)))]


def propose_parameter_batch(
    task: DesignTask,
    *,
    observations: list[RlObservation],
    run_label: str,
    step_index: int,
    population_size: int,
) -> list[dict[str, str | float | int | bool]]:
    """Generate a deterministic policy-driven proposal population."""

    if not observations:
        from libs.eval.random_search import sample_parameter_values

        return [
            sample_parameter_values(task, run_label=run_label, step_index=step_index, sample_index=sample_index)
            for sample_index in range(max(4, population_size * 2))
        ]

    elite = _elite(observations)
    reward_sum = sum(max(0.1, observation.reward + 3.0) for observation in elite)
    means: dict[str, float] = {}
    spreads: dict[str, float] = {}
    for variable in task.design_space.variables:
        values = [_normalize(task, variable.name, obs.parameter_values.get(variable.name, variable.default)) for obs in elite]
        weights = [max(0.1, obs.reward + 3.0) / max(reward_sum, 1e-12) for obs in elite]
        mean = sum(weight * value for weight, value in zip(weights, values))
        variance = sum(weight * ((value - mean) ** 2) for weight, value in zip(weights, values))
        means[variable.name] = mean
        spreads[variable.name] = max(0.05, min(0.22, math.sqrt(max(variance, 0.0)) + 0.04))

    pool_size = max(6, population_size * 4)
    proposals: list[dict[str, str | float | int | bool]] = []
    for sample_index in range(pool_size):
        values = dict(task.initial_state.template_defaults)
        for variable in task.design_space.variables:
            normalized = means[variable.name] + spreads[variable.name] * _standard_normal(
                f"{run_label}|{step_index}|{sample_index}|{variable.name}"
            )
            values[variable.name] = _denormalize(task, variable.name, normalized)
        proposals.append(values)
    return proposals


def build_surrogate_predictions(
    task: DesignTask,
    state_id: str,
    parameter_values: dict[str, str | float | int | bool],
    *,
    observations: list[RlObservation],
    seed: str,
) -> tuple[MetricsPrediction, FeasibilityPrediction, SimulationValueEstimate, float]:
    """Build a deterministic RL-style surrogate prediction and policy score."""

    relevant_metrics = _relevant_metrics(task)
    if not observations:
        trust = TrustAssessment(
            trust_level="low",
            service_tier="ranking_ready",
            confidence=0.2,
            uncertainty_score=0.88,
            ood_score=0.4,
            must_escalate=False,
            hard_block=False,
            reasons=["rl_baseline_cold_start"],
        )
        metric_map = {metric: _metric_prior(metric, seed) for metric in relevant_metrics}
        constraints = [item.model_copy(update={"source": "prediction"}) for item in evaluate_constraints(task, metric_map)]
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
            auxiliary_features={"surrogate_kind_code": 3.0, "observation_count": 0.0},
            trust_assessment=trust,
        )
        feasibility_prediction = FeasibilityPrediction(
            state_id=state_id,
            task_id=task.task_id,
            overall_feasibility=0.34,
            per_group_constraints=constraints,
            most_likely_failure_reasons=[item.constraint_name for item in constraints if item.margin < 0.0],
            confidence=trust.confidence,
            trust_assessment=trust,
        )
        sim_value = SimulationValueEstimate(
            state_id=state_id,
            estimated_value=0.8,
            decision="simulate",
            reasons=["rl_baseline_cold_start"],
            trust_assessment=trust,
        )
        return metrics_prediction, feasibility_prediction, sim_value, 0.8

    elite = _elite(observations)
    reward_sum = sum(max(0.1, obs.reward + 3.0) for obs in elite)
    metric_map: dict[str, float] = {}
    for metric in relevant_metrics:
        weighted_values = [
            (max(0.1, obs.reward + 3.0), obs.measured_metrics[metric])
            for obs in elite
            if metric in obs.measured_metrics
        ]
        if not weighted_values:
            metric_map[metric] = _metric_prior(metric, seed)
            continue
        total = sum(weight for weight, _ in weighted_values)
        metric_map[metric] = round(sum(weight * value for weight, value in weighted_values) / max(total, 1e-12), 6)

    reward_mean = sum(obs.reward for obs in elite) / max(1, len(elite))
    feasible_score = sum(1.0 if obs.feasible else 0.0 for obs in elite) / max(1, len(elite))
    uncertainty = round(max(0.08, min(0.8, 0.55 - min(0.35, reward_mean * 0.05) + (1.0 - feasible_score) * 0.2)), 6)
    confidence = round(max(0.18, min(0.92, 1.0 - uncertainty * 0.85)), 6)
    trust = TrustAssessment(
        trust_level="medium" if confidence >= 0.45 else "low",
        service_tier="ranking_ready",
        confidence=confidence,
        uncertainty_score=uncertainty,
        ood_score=round(min(1.0, uncertainty * 0.7 + 0.08), 6),
        must_escalate=False,
        hard_block=False,
        reasons=["rl_baseline_policy_surrogate"],
    )
    constraints = [item.model_copy(update={"source": "prediction"}) for item in evaluate_constraints(task, metric_map)]
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
        auxiliary_features={"surrogate_kind_code": 3.0, "observation_count": float(len(observations))},
        trust_assessment=trust,
    )
    feasibility_prediction = FeasibilityPrediction(
        state_id=state_id,
        task_id=task.task_id,
        overall_feasibility=round(max(0.05, min(0.95, feasible_score * confidence + 0.12)), 6),
        per_group_constraints=constraints,
        most_likely_failure_reasons=[item.constraint_name for item in constraints if item.margin < 0.0],
        confidence=confidence,
        trust_assessment=trust,
    )
    estimated_value = round(max(0.2, min(0.95, 0.48 + reward_mean * 0.08 + feasible_score * 0.22)), 6)
    sim_value = SimulationValueEstimate(
        state_id=state_id,
        estimated_value=estimated_value,
        decision="simulate" if estimated_value >= 0.45 else "defer",
        reasons=["rl_baseline"],
        trust_assessment=trust,
    )
    return metrics_prediction, feasibility_prediction, sim_value, estimated_value
