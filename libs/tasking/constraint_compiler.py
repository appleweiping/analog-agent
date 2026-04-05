"""Constraint compilation for the task formalization layer."""

from __future__ import annotations

from dataclasses import dataclass

from libs.schema.design_spec import DesignSpec
from libs.schema.design_task import ConstraintGroup, ConstraintSet, ConstraintSpec
from libs.tasking.constraint_resolver import group_for_metric, normalize_metric_range, stage_for_metric

OPERATING_REGION_RULES = {
    "two_stage_ota": ["input_pair_devices_in_saturation", "second_stage_device_in_saturation"],
    "folded_cascode_ota": ["folded_branch_devices_in_saturation", "output_branch_headroom_valid"],
    "telescopic_ota": ["cascode_stack_headroom_valid", "input_pair_devices_in_saturation"],
    "comparator": ["regenerative_pair_bias_valid"],
    "ldo": ["pass_device_headroom_valid", "error_amplifier_bias_valid"],
    "bandgap": ["core_branch_currents_positive", "startup_path_bias_valid"],
    "unknown": [],
}

FEASIBILITY_RULES = {
    "two_stage_ota": ["dc_operating_point_must_converge"],
    "folded_cascode_ota": ["dc_operating_point_must_converge"],
    "telescopic_ota": ["dc_operating_point_must_converge"],
    "comparator": ["clocked_operation_bias_must_converge"],
    "ldo": ["regulation_loop_must_converge"],
    "bandgap": ["reference_core_must_converge"],
    "unknown": [],
}


@dataclass(frozen=True)
class ConstraintCompilation:
    """Compiled constraint set plus provenance."""

    constraints: ConstraintSet
    derived_fields: list[str]
    assumptions: list[str]


def _criticality(metric: str) -> str:
    if metric in {"phase_margin_deg", "output_swing_v", "input_common_mode_v"}:
        return "important"
    return "core"


def _constraint_name(metric: str, relation: str) -> str:
    relation_suffix = {
        ">=": "lower_bound",
        "<=": "upper_bound",
        "==": "target",
        "in_range": "range",
    }[relation]
    return f"{metric}_{relation_suffix}"


def compile_constraints(spec: DesignSpec) -> ConstraintCompilation:
    """Convert DesignSpec hard constraints into a solver-facing constraint system."""

    hard_constraints: list[ConstraintSpec] = []
    groups: dict[str, list[str]] = {}

    for metric, metric_range in spec.hard_constraints.items():
        relation_payload = normalize_metric_range(metric, metric_range)
        relation = str(relation_payload["relation"])
        constraint = ConstraintSpec(
            name=_constraint_name(metric, relation),
            metric=metric,
            relation=relation,
            threshold=float(relation_payload["threshold"]) if "threshold" in relation_payload else None,
            lower_threshold=float(relation_payload["lower_threshold"]) if "lower_threshold" in relation_payload else None,
            upper_threshold=float(relation_payload["upper_threshold"]) if "upper_threshold" in relation_payload else None,
            tolerance=float(relation_payload.get("tolerance", 0.0)),
            evaluation_stage=stage_for_metric(metric),
            penalty_policy="hard_fail",
            source="user",
            criticality=_criticality(metric),
        )
        hard_constraints.append(constraint)
        groups.setdefault(group_for_metric(metric), []).append(constraint.name)

    constraint_groups = [
        ConstraintGroup(name=f"{group}_group", members=members)
        for group, members in sorted(groups.items())
    ]

    constraints = ConstraintSet(
        hard_constraints=hard_constraints,
        soft_constraints=[],
        feasibility_rules=FEASIBILITY_RULES.get(spec.circuit_family, []),
        operating_region_rules=OPERATING_REGION_RULES.get(spec.circuit_family, []),
        constraint_groups=constraint_groups,
    )
    return ConstraintCompilation(
        constraints=constraints,
        derived_fields=["constraints"],
        assumptions=[],
    )
