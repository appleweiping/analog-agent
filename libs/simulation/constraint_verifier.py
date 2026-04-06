"""Constraint adjudication for the fifth layer."""

from __future__ import annotations

from libs.schema.design_task import ConstraintSpec, DesignTask
from libs.schema.simulation import ConstraintAssessmentRecord, VerificationPolicy


def _severity(constraint: ConstraintSpec, is_satisfied: bool, margin: float) -> str:
    if is_satisfied:
        return "low"
    if constraint.criticality == "core" or margin < -0.1:
        return "critical"
    if constraint.criticality == "important":
        return "high"
    return "medium"


def _margin(constraint: ConstraintSpec, value: float) -> float:
    if constraint.relation == ">=":
        assert constraint.threshold is not None
        return value - float(constraint.threshold)
    if constraint.relation == "<=":
        assert constraint.threshold is not None
        return float(constraint.threshold) - value
    if constraint.relation == "==":
        assert constraint.threshold is not None
        return constraint.tolerance - abs(value - float(constraint.threshold))
    assert constraint.lower_threshold is not None and constraint.upper_threshold is not None
    return min(value - float(constraint.lower_threshold), float(constraint.upper_threshold) - value)


def _is_satisfied(constraint: ConstraintSpec, value: float) -> bool:
    if constraint.relation == ">=":
        assert constraint.threshold is not None
        return value + constraint.tolerance >= float(constraint.threshold)
    if constraint.relation == "<=":
        assert constraint.threshold is not None
        return value - constraint.tolerance <= float(constraint.threshold)
    if constraint.relation == "==":
        assert constraint.threshold is not None
        return abs(value - float(constraint.threshold)) <= constraint.tolerance
    assert constraint.lower_threshold is not None and constraint.upper_threshold is not None
    return float(constraint.lower_threshold) - constraint.tolerance <= value <= float(constraint.upper_threshold) + constraint.tolerance


def verify_constraints(
    task: DesignTask,
    metric_values: dict[str, float],
    verification_policy: VerificationPolicy,
) -> list[ConstraintAssessmentRecord]:
    """Return structured constraint assessments."""

    assessments: list[ConstraintAssessmentRecord] = []
    group_map = {
        member: group.name
        for group in task.constraints.constraint_groups
        for member in group.members
    }
    for constraint in [*task.constraints.hard_constraints, *task.constraints.soft_constraints]:
        value = metric_values.get(constraint.metric)
        if value is None:
            assessments.append(
                ConstraintAssessmentRecord(
                    constraint_name=constraint.name,
                    constraint_group=group_map.get(constraint.name, "ungrouped"),
                    metric=constraint.metric,
                    is_satisfied=False,
                    margin=-1.0,
                    severity="critical" if constraint.criticality == "core" else "high",
                )
            )
            continue
        margin = _margin(constraint, value)
        satisfied = _is_satisfied(constraint, value)
        assessments.append(
            ConstraintAssessmentRecord(
                constraint_name=constraint.name,
                constraint_group=group_map.get(constraint.name, "ungrouped"),
                metric=constraint.metric,
                is_satisfied=satisfied,
                margin=round(float(margin), 6),
                severity=_severity(constraint, satisfied, margin),
            )
        )
    return assessments
