"""Difficulty profiling for formalized design tasks."""

from __future__ import annotations

from libs.schema.design_task import DesignSpace, DifficultyProfile, EvaluationPlan


def build_difficulty_profile(
    design_space: DesignSpace,
    hard_constraint_count: int,
    evaluation_plan: EvaluationPlan,
    unresolved_dependencies: list[str],
) -> DifficultyProfile:
    """Estimate optimization difficulty from the compiled task structure."""

    variable_dimension = len(design_space.variables)
    discrete_count = sum(1 for variable in design_space.variables if variable.kind in {"categorical", "binary", "integer"})
    discrete_degree = round(discrete_count / variable_dimension, 4) if variable_dimension else 0.0

    if hard_constraint_count >= 4 or unresolved_dependencies:
        constraint_tightness = "high"
    elif hard_constraint_count >= 2:
        constraint_tightness = "medium"
    else:
        constraint_tightness = "low"

    evaluation_cost = evaluation_plan.simulation_budget_class
    if unresolved_dependencies:
        expected_feasibility = "unknown"
    elif hard_constraint_count >= 4:
        expected_feasibility = "low"
    elif hard_constraint_count >= 2:
        expected_feasibility = "medium"
    else:
        expected_feasibility = "high"

    sensitivity_hint: list[str] = []
    if any(variable.name in {"cc", "c_comp"} for variable in design_space.variables):
        sensitivity_hint.append("stability-related compensation variables are likely high-sensitivity")
    if any(variable.name in {"ibias", "w_pass"} for variable in design_space.variables):
        sensitivity_hint.append("bias-related variables strongly affect both feasibility and power")

    risk_flags = list(unresolved_dependencies)
    if evaluation_cost == "expensive":
        risk_flags.append("evaluation_budget_pressure")
    if discrete_degree > 0.2:
        risk_flags.append("mixed_discrete_continuous_search")

    return DifficultyProfile(
        variable_dimension=variable_dimension,
        discrete_degree=discrete_degree,
        constraint_tightness=constraint_tightness,
        evaluation_cost=evaluation_cost,
        expected_feasibility=expected_feasibility,
        sensitivity_hint=sensitivity_hint,
        risk_flags=risk_flags,
    )
