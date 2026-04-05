"""Validation engine for compiled design tasks."""

from __future__ import annotations

from libs.schema.design_spec import DesignSpec
from libs.schema.design_task import DesignTask, TaskValidationIssue, ValidationStatus
from libs.tasking.constraint_resolver import stage_for_metric

CORE_UNRESOLVED_FIELDS = {
    "circuit_family",
    "process_node",
    "supply_voltage_v",
    "load_cap_f",
}


def _issue(code: str, path: str, message: str, severity: str) -> TaskValidationIssue:
    return TaskValidationIssue(code=code, path=path, message=message, severity=severity)


def validate_design_task(task: DesignTask, source_spec: DesignSpec) -> ValidationStatus:
    """Validate a DesignTask for structure, semantics, and downstream readiness."""

    errors: list[TaskValidationIssue] = []
    warnings: list[TaskValidationIssue] = []
    unresolved = list(dict.fromkeys(task.validation_status.unresolved_dependencies))

    if not task.design_space.variables:
        errors.append(_issue("missing_problem_core", "design_space.variables", "design space must expose at least one variable", "error"))
    if not task.objective.reporting_metrics and not task.constraints.hard_constraints:
        errors.append(_issue("missing_problem_core", "objective", "task must expose objectives or hard constraints", "error"))
    if not task.evaluation_plan.analyses or not task.evaluation_plan.metric_extractors:
        errors.append(_issue("missing_problem_core", "evaluation_plan", "evaluation plan must be fully populated", "error"))
    if not task.task_graph.nodes or not task.task_graph.edges:
        errors.append(_issue("missing_problem_core", "task_graph", "task graph must be non-empty", "error"))

    if task.task_type == "sizing" and task.topology.topology_mode != "fixed":
        errors.append(_issue("topology_mode_inconsistency", "topology.topology_mode", "sizing tasks must bind to a fixed topology", "error"))
    if task.task_type == "topology_sizing" and task.topology.topology_mode == "fixed":
        errors.append(_issue("topology_mode_inconsistency", "topology.topology_mode", "topology_sizing tasks cannot use fixed topology mode", "error"))

    has_structural_variable = any(variable.role == "structural_template_choice" for variable in task.design_space.variables)
    if task.topology.topology_mode == "fixed" and has_structural_variable:
        errors.append(_issue("topology_mode_inconsistency", "design_space.variables", "fixed topology tasks cannot expose structural selection variables", "error"))
    if task.topology.topology_mode != "fixed" and not has_structural_variable:
        errors.append(_issue("topology_mode_inconsistency", "design_space.variables", "topology search tasks must expose a structural selection variable", "error"))

    extractor_metrics = {extractor.metric for extractor in task.evaluation_plan.metric_extractors}
    analysis_types = {analysis.analysis_type for analysis in task.evaluation_plan.analyses}
    for metric in task.objective.reporting_metrics:
        if metric not in extractor_metrics:
            errors.append(_issue("evaluation_coverage_error", "evaluation_plan.metric_extractors", f"missing extractor for reporting metric {metric}", "error"))
        expected_stage = stage_for_metric(metric)
        if expected_stage not in analysis_types:
            errors.append(_issue("evaluation_coverage_error", "evaluation_plan.analyses", f"missing analysis {expected_stage} for metric {metric}", "error"))

    for constraint in task.constraints.hard_constraints:
        if constraint.metric not in extractor_metrics:
            errors.append(_issue("evaluation_coverage_error", f"constraints.hard_constraints.{constraint.name}", f"missing extractor for constrained metric {constraint.metric}", "error"))
        if constraint.evaluation_stage not in analysis_types:
            errors.append(_issue("evaluation_coverage_error", f"constraints.hard_constraints.{constraint.name}", f"missing analysis {constraint.evaluation_stage}", "error"))

    objective_directions = {term.metric: term.direction for term in task.objective.terms}
    for constraint in task.constraints.hard_constraints:
        direction = objective_directions.get(constraint.metric)
        if direction == "minimize" and constraint.relation == ">=":
            errors.append(_issue("constraint_direction_error", f"constraints.hard_constraints.{constraint.name}", f"constraint direction conflicts with objective for {constraint.metric}", "error"))
        if direction == "maximize" and constraint.relation == "<=":
            errors.append(_issue("constraint_direction_error", f"constraints.hard_constraints.{constraint.name}", f"constraint direction conflicts with objective for {constraint.metric}", "error"))

    if source_spec.circuit_family == "unknown" and task.task_type == "sizing":
        errors.append(_issue("semantic_consistency_error", "task_type", "unknown circuit families cannot compile to a fixed sizing task", "error"))

    unresolved_core = [item for item in unresolved if item in CORE_UNRESOLVED_FIELDS]
    for dependency in unresolved:
        warnings.append(_issue("unresolved_dependency_error", "validation_status.unresolved_dependencies", f"unresolved dependency: {dependency}", "warning"))

    if not errors and unresolved_core:
        warnings.append(_issue("solver_readiness_error", "validation_status", "task is structurally compiled but not fully solver-ready", "warning"))

    completeness = 1.0
    completeness -= 0.15 * len(unresolved)
    completeness -= 0.25 * len(errors)
    completeness_score = round(max(0.0, min(1.0, completeness)), 4)
    is_valid = not errors and not unresolved_core

    return ValidationStatus(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        unresolved_dependencies=unresolved,
        repair_history=task.validation_status.repair_history,
        completeness_score=completeness_score,
    )
