"""Objective compilation for the task formalization layer."""

from __future__ import annotations

from dataclasses import dataclass

from libs.schema.design_spec import DesignSpec
from libs.schema.design_task import ObjectiveSpec, ObjectiveTerm
from libs.tasking.objective_scalarizer import canonical_metric_order, default_scalarization, default_weight


@dataclass(frozen=True)
class ObjectiveCompilation:
    """Compiled objective object plus provenance."""

    objective: ObjectiveSpec
    derived_fields: list[str]
    assumptions: list[str]


def compile_objective(spec: DesignSpec) -> ObjectiveCompilation:
    """Compile soft intents into a solver-facing objective definition."""

    objective_metrics = canonical_metric_order([*spec.objectives.maximize, *spec.objectives.minimize])
    terms: list[ObjectiveTerm] = []
    term_count = len(objective_metrics)
    weight = default_weight(term_count)

    for metric in canonical_metric_order(spec.objectives.maximize):
        terms.append(
            ObjectiveTerm(
                metric=metric,
                direction="maximize",
                weight=weight,
                transform="identity",
                normalization="minmax",
                source_constraint_relation="upstream_objective",
            )
        )
    for metric in canonical_metric_order(spec.objectives.minimize):
        terms.append(
            ObjectiveTerm(
                metric=metric,
                direction="minimize",
                weight=weight,
                transform="identity",
                normalization="minmax",
                source_constraint_relation="upstream_objective",
            )
        )

    if not terms:
        objective_mode = "feasibility"
        scalarization = "none"
    elif len(terms) == 1 and spec.hard_constraints:
        objective_mode = "constrained_single"
        scalarization = default_scalarization(len(terms))
    elif len(terms) == 1:
        objective_mode = "single"
        scalarization = default_scalarization(len(terms))
    else:
        objective_mode = "multi_objective"
        scalarization = default_scalarization(len(terms))

    reporting_metrics = canonical_metric_order([*objective_metrics, *list(spec.hard_constraints.keys())])
    reference_point = {
        metric: float(spec.hard_constraints[metric].target)
        for metric in spec.hard_constraints
        if spec.hard_constraints[metric].target is not None
    }
    priority_policy = "feasibility_first" if spec.hard_constraints else "balanced"
    objective = ObjectiveSpec(
        objective_mode=objective_mode,
        terms=terms,
        scalarization=scalarization,
        reference_point=reference_point,
        priority_policy=priority_policy,
        reporting_metrics=reporting_metrics,
    )

    assumptions: list[str] = []
    if objective_mode == "feasibility":
        assumptions.append("task carries no explicit soft objective and is compiled as a feasibility problem")

    return ObjectiveCompilation(
        objective=objective,
        derived_fields=["objective"],
        assumptions=assumptions,
    )
