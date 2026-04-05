"""Objective-scalarization helpers for formalized design tasks."""

from __future__ import annotations


from libs.schema.design_spec import OBJECTIVE_METRICS


def canonical_metric_order(metrics: list[str]) -> list[str]:
    """Return metrics in a deterministic project-wide canonical order."""

    order = {name: index for index, name in enumerate(OBJECTIVE_METRICS)}
    return sorted(dict.fromkeys(metrics), key=lambda item: (order.get(item, len(order)), item))


def default_scalarization(term_count: int) -> str:
    """Choose a deterministic scalarization policy."""

    if term_count <= 1:
        return "weighted_sum"
    return "weighted_sum"


def default_weight(term_count: int) -> float:
    """Return a stable default weight for one objective term."""

    if term_count <= 0:
        return 1.0
    return round(1.0 / term_count, 6)
