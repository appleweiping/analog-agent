"""Validation helpers for the planning layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.planning import (
    CandidateRecord,
    PlanningBundle,
    PlanningValidationIssue,
    PlanningValidationStatus,
    SearchState,
)
from libs.schema.world_model import WorldModelBundle


def _issue(code: str, path: str, message: str, severity: str = "error") -> PlanningValidationIssue:
    return PlanningValidationIssue(code=code, path=path, message=message, severity=severity)


def validate_planning_bundle(bundle: PlanningBundle, task: DesignTask, world_model_bundle: WorldModelBundle) -> PlanningValidationStatus:
    """Validate a compiled PlanningBundle."""

    errors: list[PlanningValidationIssue] = []
    warnings: list[PlanningValidationIssue] = []
    unresolved: list[str] = []

    if bundle.parent_task_id != task.task_id:
        errors.append(_issue("schema_failure", "parent_task_id", "planning bundle must bind to the source DesignTask"))
    if bundle.world_model_binding.parent_task_id != task.task_id:
        errors.append(_issue("world_model_binding_failure", "world_model_binding.parent_task_id", "world-model binding must align with the source task"))
    if bundle.world_model_binding.world_model_id != world_model_bundle.world_model_id:
        errors.append(_issue("world_model_binding_failure", "world_model_binding.world_model_id", "planning bundle must bind the provided WorldModelBundle"))
    if task.task_type not in bundle.world_model_binding.supported_task_types:
        errors.append(_issue("unsupported_task_type", "world_model_binding.supported_task_types", "task type not supported by world-model binding"))
    if task.circuit_family not in bundle.world_model_binding.supported_circuit_families:
        errors.append(_issue("unsupported_family", "world_model_binding.supported_circuit_families", "circuit family not supported by world-model binding"))
    if bundle.rollout_config.horizon <= 0 or bundle.rollout_config.beam_width <= 0:
        errors.append(_issue("schema_failure", "rollout_config", "rollout configuration must use positive horizon and beam width"))
    if bundle.budget_controller.max_real_simulations <= 0:
        errors.append(_issue("budget_state_failure", "budget_controller.max_real_simulations", "planner must reserve at least one real simulation"))
    if bundle.termination_policy.max_iterations <= 0:
        errors.append(_issue("schema_failure", "termination_policy.max_iterations", "termination policy must allow positive iterations"))
    if not bundle.phase_controller.allowed_transitions:
        warnings.append(_issue("phase_state_failure", "phase_controller.allowed_transitions", "no explicit phase transitions defined", "warning"))
    if bundle.selection_policy.ranking_source != "world_model_ranker":
        errors.append(_issue("serving_contract_failure", "selection_policy.ranking_source", "planning layer must rely on the world-model ranker"))
    completeness = 1.0
    if warnings:
        completeness -= 0.05 * len(warnings)
    if unresolved:
        completeness -= 0.1 * len(unresolved)
    if errors:
        completeness = max(0.0, 0.55 - 0.1 * len(errors))
    return PlanningValidationStatus(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        unresolved_dependencies=unresolved,
        repair_history=[],
        completeness_score=max(0.0, round(completeness, 4)),
    )


def validate_candidate_record(bundle: PlanningBundle, candidate: CandidateRecord) -> PlanningValidationStatus:
    """Validate one candidate record."""

    errors: list[PlanningValidationIssue] = []
    warnings: list[PlanningValidationIssue] = []

    if candidate.task_id != bundle.parent_task_id:
        errors.append(_issue("candidate_lifecycle_failure", "task_id", "candidate task_id must align with planning bundle"))
    if candidate.world_state_ref != candidate.world_state_snapshot.state_id:
        errors.append(_issue("candidate_lifecycle_failure", "world_state_ref", "candidate world_state_ref must match the embedded world state"))
    if candidate.predicted_uncertainty and candidate.predicted_feasibility:
        if candidate.predicted_uncertainty.service_tier == "hard_block" and candidate.lifecycle_status not in {"rejected", "screened_out"}:
            warnings.append(_issue("candidate_lifecycle_failure", "lifecycle_status", "hard-block candidates should not remain active", "warning"))
    completeness = 1.0 if not errors else 0.5
    return PlanningValidationStatus(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        unresolved_dependencies=[],
        repair_history=[],
        completeness_score=completeness,
    )


def validate_search_state(bundle: PlanningBundle, state: SearchState) -> PlanningValidationStatus:
    """Validate one SearchState object."""

    errors: list[PlanningValidationIssue] = []
    warnings: list[PlanningValidationIssue] = []
    candidate_ids = {candidate.candidate_id for candidate in state.candidate_pool_state.candidates}

    if state.task_id != bundle.parent_task_id:
        errors.append(_issue("search_state_failure", "task_id", "search state must align with planning bundle task"))
    if state.budget_state.proxy_evaluations_used > state.budget_state.proxy_evaluation_budget:
        errors.append(_issue("budget_state_failure", "budget_state.proxy_evaluations_used", "proxy-evaluation budget exceeded"))
    if state.budget_state.rollouts_used > state.budget_state.rollout_budget:
        errors.append(_issue("budget_state_failure", "budget_state.rollouts_used", "rollout budget exceeded"))
    if state.budget_state.simulations_used > state.budget_state.simulation_budget:
        errors.append(_issue("budget_state_failure", "budget_state.simulations_used", "simulation budget exceeded"))
    if state.phase_state.current_phase not in bundle.search_state_schema.phase_values:
        errors.append(_issue("phase_state_failure", "phase_state.current_phase", "search state carries an unsupported phase"))
    invalid_frontier = [candidate_id for candidate_id in state.frontier_state.frontier_candidate_ids if candidate_id not in candidate_ids]
    if invalid_frontier:
        errors.append(_issue("search_state_failure", "frontier_state.frontier_candidate_ids", f"frontier references unknown candidates: {invalid_frontier}"))
    if state.best_known_feasible and state.best_known_feasible.candidate_id not in candidate_ids:
        errors.append(_issue("search_state_failure", "best_known_feasible", "best feasible candidate must exist in candidate pool"))
    if state.best_known_infeasible and state.best_known_infeasible.candidate_id not in candidate_ids:
        errors.append(_issue("search_state_failure", "best_known_infeasible", "best infeasible candidate must exist in candidate pool"))
    if not state.trace_log:
        warnings.append(_issue("trace_integrity_failure", "trace_log", "search state has no optimization traces yet", "warning"))
    completeness = 1.0
    if warnings:
        completeness -= 0.05 * len(warnings)
    if errors:
        completeness = max(0.0, 0.5 - 0.05 * len(errors))
    return PlanningValidationStatus(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        unresolved_dependencies=[],
        repair_history=[],
        completeness_score=max(0.0, round(completeness, 4)),
    )

