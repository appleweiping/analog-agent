"""Evaluation metric helpers."""

from __future__ import annotations

from collections import Counter

from libs.schema.experiment import ExperimentResult


def pass_rate(results: list[bool]) -> float:
    """Compute the fraction of successful runs."""
    return sum(results) / len(results) if results else 0.0


def simulation_reduction_ratio(baseline_calls: int, system_calls: int) -> float:
    """Compute the relative reduction in real simulation calls."""

    if baseline_calls <= 0:
        return 0.0
    return round((baseline_calls - system_calls) / baseline_calls, 6)


def feasible_hit_rate(results: list[ExperimentResult]) -> float:
    """Compute the fraction of runs that found a feasible solution."""

    if not results:
        return 0.0
    return round(sum(1 for result in results if result.best_feasible_found) / len(results), 6)


def efficiency_score(feasible_found: bool, simulation_calls: int) -> float:
    """Compute feasible-found efficiency normalized by real simulation calls."""

    if simulation_calls <= 0:
        return 1.0 if feasible_found else 0.0
    return round((1.0 if feasible_found else 0.0) / simulation_calls, 6)


def aggregate_failure_type_distribution(results: list[ExperimentResult]) -> dict[str, int]:
    """Aggregate failure type counts across experiment runs."""

    counter: Counter[str] = Counter()
    for result in results:
        counter.update(result.failure_type_distribution)
    return dict(sorted(counter.items()))
