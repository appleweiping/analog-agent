"""Validation helpers for the fifth layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.planning import CandidateRecord
from libs.schema.simulation import (
    SimulationBundle,
    SimulationValidationIssue,
    SimulationValidationStatus,
    VerificationResult,
)


def validate_simulation_bundle(simulation_bundle: SimulationBundle, task: DesignTask, candidate: CandidateRecord) -> SimulationValidationStatus:
    """Validate a compiled SimulationBundle."""

    errors: list[SimulationValidationIssue] = []
    warnings: list[SimulationValidationIssue] = []
    unresolved: list[str] = []

    if simulation_bundle.parent_task_id != task.task_id:
        errors.append(
            SimulationValidationIssue(
                code="candidate_binding_failure",
                path="parent_task_id",
                message="simulation bundle is not bound to the provided DesignTask",
                severity="error",
            )
        )
    if simulation_bundle.candidate_id != candidate.candidate_id:
        errors.append(
            SimulationValidationIssue(
                code="candidate_binding_failure",
                path="candidate_id",
                message="simulation bundle candidate does not match requested candidate",
                severity="error",
            )
        )
    if simulation_bundle.netlist_instance.render_status != "ready":
        errors.append(
            SimulationValidationIssue(
                code="netlist_render_failure",
                path="netlist_instance.render_status",
                message="netlist instance failed integrity checks",
                severity="error",
            )
        )
    if not simulation_bundle.analysis_plan.ordered_analyses:
        errors.append(
            SimulationValidationIssue(
                code="analysis_plan_failure",
                path="analysis_plan.ordered_analyses",
                message="analysis plan is empty",
                severity="error",
            )
        )
    if not simulation_bundle.measurement_contract.measurement_definitions:
        errors.append(
            SimulationValidationIssue(
                code="measurement_contract_failure",
                path="measurement_contract.measurement_definitions",
                message="measurement contract defines no metrics",
                severity="error",
            )
        )
    if simulation_bundle.backend_binding.invocation_mode != "native":
        warnings.append(
            SimulationValidationIssue(
                code="backend_binding_failure",
                path="backend_binding.invocation_mode",
                message="backend is operating in mock_truth mode",
                severity="warning",
            )
        )
    if not simulation_bundle.artifact_registry.run_directory:
        unresolved.append("artifact_registry.run_directory")

    total_checks = 6
    score = 1.0 - (len(errors) * 0.18 + len(unresolved) * 0.08 + len(warnings) * 0.03)
    return SimulationValidationStatus(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        unresolved_dependencies=unresolved,
        repair_history=[],
        completeness_score=max(0.0, min(1.0, round(score, 4))),
    )


def validate_verification_result(result: VerificationResult) -> SimulationValidationStatus:
    """Validate a VerificationResult before emission."""

    errors: list[SimulationValidationIssue] = []
    warnings: list[SimulationValidationIssue] = []
    if not result.measurement_report.measurement_results:
        errors.append(
            SimulationValidationIssue(
                code="measurement_contract_failure",
                path="measurement_report.measurement_results",
                message="verification result has no measurement results",
                severity="error",
            )
        )
    if not result.constraint_assessment:
        errors.append(
            SimulationValidationIssue(
                code="verification_policy_failure",
                path="constraint_assessment",
                message="verification result has no constraint assessments",
                severity="error",
            )
        )
    if result.robustness_summary.certification_status == "robustness_failed":
        warnings.append(
            SimulationValidationIssue(
                code="execution_failure",
                path="robustness_summary",
                message="candidate failed robustness certification",
                severity="warning",
            )
        )
    score = 1.0 - (len(errors) * 0.2 + len(warnings) * 0.04)
    return SimulationValidationStatus(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        unresolved_dependencies=[],
        repair_history=[],
        completeness_score=max(0.0, min(1.0, round(score, 4))),
    )
