"""Retrieval-policy helpers for the memory layer."""

from __future__ import annotations

from libs.schema.memory import TaskSignature


def signature_overlap(query: TaskSignature, candidate: TaskSignature) -> float:
    """Compute deterministic overlap between two task signatures."""

    score = 0.0
    if query.circuit_family == candidate.circuit_family:
        score += 0.35
    if query.task_type == candidate.task_type:
        score += 0.2
    overlap_sets = (
        (set(query.constraint_vector), set(candidate.constraint_vector), 0.15),
        (set(query.environment_profile), set(candidate.environment_profile), 0.1),
        (set(query.evaluation_profile), set(candidate.evaluation_profile), 0.1),
        (set(query.design_space_shape), set(candidate.design_space_shape), 0.1),
    )
    for left, right, weight in overlap_sets:
        if not left or not right:
            continue
        score += weight * (len(left & right) / max(1, len(left | right)))
    return round(min(score, 1.0), 4)
