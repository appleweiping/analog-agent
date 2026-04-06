"""Validation helpers for the world model layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.world_model import (
    DesignAction,
    VALIDATION_ERROR_CODES,
    WorldModelBundle,
    WorldModelValidationIssue,
    WorldModelValidationStatus,
    WorldState,
)


def _issue(code: str, path: str, message: str, severity: str) -> WorldModelValidationIssue:
    if code not in VALIDATION_ERROR_CODES:
        raise ValueError(f"unsupported validation code: {code}")
    return WorldModelValidationIssue(code=code, path=path, message=message, severity=severity)


def validate_world_model_bundle(bundle: WorldModelBundle) -> WorldModelValidationStatus:
    """Validate a compiled world-model bundle."""

    errors: list[WorldModelValidationIssue] = []
    warnings: list[WorldModelValidationIssue] = []

    if not bundle.supported_circuit_families:
        errors.append(_issue("unsupported_family", "supported_circuit_families", "bundle must support at least one family", "error"))
    if not bundle.supported_task_types:
        errors.append(_issue("unsupported_task_type", "supported_task_types", "bundle must support at least one task type", "error"))
    if not bundle.prediction_heads.metric_prediction_head.supported_metrics:
        errors.append(_issue("schema_failure", "prediction_heads.metric_prediction_head", "metric head must expose supported metrics", "error"))
    if not bundle.serving_contract.predict_metrics.output_schema:
        errors.append(_issue("serving_contract_error", "serving_contract", "predict_metrics contract must be populated", "error"))
    if not bundle.calibration_state.usable_regions:
        warnings.append(_issue("calibration_state_error", "calibration_state.usable_regions", "bundle has no explicitly usable regions yet", "warning"))
    if bundle.training_state.sample_counts.static_samples <= 0:
        warnings.append(_issue("training_state_error", "training_state.sample_counts", "training sample counts are placeholder-only", "warning"))

    completeness = 1.0 - 0.2 * len(errors) - 0.05 * len(warnings)
    return WorldModelValidationStatus(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        unresolved_dependencies=[],
        repair_history=[],
        completeness_score=max(0.0, min(1.0, round(completeness, 4))),
    )


def validate_world_state(bundle: WorldModelBundle, task: DesignTask, state: WorldState) -> WorldModelValidationStatus:
    """Validate one world-state object against bundle and task context."""

    errors: list[WorldModelValidationIssue] = []
    warnings: list[WorldModelValidationIssue] = []

    if state.task_id != task.task_id or state.task_id != bundle.parent_task_id:
        errors.append(_issue("state_context_error", "task_id", "state task_id must match the bundle parent task", "error"))
    if state.topology_context.topology_mode not in {"fixed", "template_family", "search_space"}:
        errors.append(_issue("schema_failure", "topology_context.topology_mode", "unsupported topology mode", "error"))
    if not state.parameter_state:
        errors.append(_issue("schema_failure", "parameter_state", "world state must expose parameter_state", "error"))
    if not state.performance_observation:
        warnings.append(_issue("state_context_error", "performance_observation", "world state has no performance observations yet", "warning"))
    if not state.constraint_observation:
        warnings.append(_issue("state_context_error", "constraint_observation", "world state has no constraint observations yet", "warning"))

    out_of_domain = [parameter.variable_name for parameter in state.parameter_state if parameter.normalized_value < 0.0 or parameter.normalized_value > 1.0]
    if out_of_domain:
        warnings.append(_issue("state_domain_error", "parameter_state", f"parameters outside nominal domain: {out_of_domain}", "warning"))
    if state.uncertainty_context.ood_score > 0.75:
        warnings.append(_issue("ood_risk_error", "uncertainty_context.ood_score", "state is deep in an OOD region", "warning"))

    completeness = 1.0 - 0.25 * len(errors) - 0.08 * len(warnings)
    return WorldModelValidationStatus(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        unresolved_dependencies=[],
        repair_history=[],
        completeness_score=max(0.0, min(1.0, round(completeness, 4))),
    )


def validate_design_action(bundle: WorldModelBundle, task: DesignTask, state: WorldState, action: DesignAction) -> WorldModelValidationStatus:
    """Validate one world-model action against bundle, task, and state."""

    errors: list[WorldModelValidationIssue] = []
    warnings: list[WorldModelValidationIssue] = []

    if action.task_id != task.task_id or action.task_id != state.task_id:
        errors.append(_issue("action_validity_error", "task_id", "action task_id must match task and state", "error"))
    if task.topology.topology_mode not in action.validity_guard.allowed_topology_modes:
        errors.append(_issue("action_validity_error", "validity_guard.allowed_topology_modes", "action is not valid for the current topology mode", "error"))
    if task.task_type not in action.validity_guard.allowed_task_types:
        errors.append(_issue("action_validity_error", "validity_guard.allowed_task_types", "action is not valid for the current task type", "error"))

    frozen = {parameter.variable_name for parameter in state.parameter_state if parameter.is_frozen}
    if action.validity_guard.blocked_when_frozen and frozen.intersection(action.action_target.variable_names):
        errors.append(_issue("action_validity_error", "action_target", "action touches frozen parameters", "error"))

    if action.action_family == "topology_switch" and task.topology.topology_mode == "fixed":
        errors.append(_issue("action_validity_error", "action_family", "fixed-topology tasks cannot accept topology_switch actions", "error"))
    if action.action_operator in {"set", "shift", "scale"} and not action.action_payload:
        errors.append(_issue("action_validity_error", "action_payload", "numeric update actions require payload values", "error"))

    completeness = 1.0 - 0.25 * len(errors) - 0.05 * len(warnings)
    return WorldModelValidationStatus(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        unresolved_dependencies=[],
        repair_history=[],
        completeness_score=max(0.0, min(1.0, round(completeness, 4))),
    )
