"""Acceptance and adapter-style testing helpers for the fifth layer."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_task import DesignTask
from libs.schema.simulation import (
    SimulationAcceptanceFailureRecord,
    SimulationExecutionResponse,
)
from libs.simulation.service import SimulationService
from libs.world_model.compiler import compile_world_model_bundle


class SimulationAcceptanceCase(BaseModel):
    """Serializable fifth-layer acceptance case."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    design_task: DesignTask
    fidelity_level: str = "focused_validation"
    backend: str = "ngspice"


class SimulationAcceptanceResult(BaseModel):
    """Per-case acceptance outcome."""

    model_config = ConfigDict(extra="forbid")

    case: SimulationAcceptanceCase
    schema_valid: bool
    measurement_correct: bool
    constraint_correct: bool
    diagnosis_correct: bool
    robustness_consistent: bool
    feedback_usable: bool
    result: str
    failures: list[SimulationAcceptanceFailureRecord] = Field(default_factory=list)


class SimulationAcceptanceSummary(BaseModel):
    """Aggregated acceptance summary for the fifth layer."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int
    passed_cases: int
    schema_validity_rate: float
    measurement_correctness_rate: float
    constraint_accuracy_rate: float
    diagnosis_quality_rate: float
    robustness_consistency_rate: float
    feedback_utility_rate: float
    failure_type_distribution: dict[str, int] = Field(default_factory=dict)


def _execute_case(case: SimulationAcceptanceCase) -> SimulationExecutionResponse:
    world_model_bundle = compile_world_model_bundle(case.design_task).world_model_bundle
    assert world_model_bundle is not None
    planning_bundle = compile_planning_bundle(case.design_task, world_model_bundle).planning_bundle
    assert planning_bundle is not None
    search_state = PlanningService(planning_bundle, case.design_task, world_model_bundle).initialize_search().search_state
    candidate_id = search_state.candidate_pool_state.candidates[0].candidate_id
    return SimulationService(case.design_task, planning_bundle, search_state).execute(
        candidate_id,
        fidelity_level=case.fidelity_level,
        backend_preference=case.backend,
        escalation_reason=f"acceptance::{case.category}",
    )


def fake_world_model_consume(execution: SimulationExecutionResponse) -> dict[str, object]:
    """Simulate third-layer consumption of calibration feedback."""

    payload = execution.verification_result.calibration_payload
    return {
        "truth_metric_count": len(payload.truth_record.metrics),
        "constraint_count": len(payload.truth_record.constraints),
        "retrain_priority": payload.retrain_priority,
    }


def fake_planner_consume(execution: SimulationExecutionResponse) -> dict[str, object]:
    """Simulate fourth-layer consumption of planner feedback."""

    payload = execution.verification_result.planner_feedback
    return {
        "lifecycle_update": payload.lifecycle_update,
        "trust_alert_count": len(payload.trust_alerts),
        "artifact_ref_count": len(payload.artifact_refs),
    }


def evaluate_case(case: SimulationAcceptanceCase) -> SimulationAcceptanceResult:
    """Execute one formal fifth-layer acceptance case."""

    execution = _execute_case(case)
    failures: list[SimulationAcceptanceFailureRecord] = []
    schema_valid = execution.simulation_bundle.validation_status.is_valid
    measurement_correct = bool(execution.verification_result.measurement_report.measured_metrics)
    constraint_correct = bool(execution.verification_result.constraint_assessment)
    diagnosis_correct = execution.verification_result.failure_attribution.primary_failure_class in {
        "none",
        "operating_region_failure",
        "stability_failure",
        "drive_bandwidth_failure",
        "noise_power_area_tradeoff_failure",
        "robustness_failure",
        "netlist_failure",
        "simulator_failure",
        "measurement_failure",
    }
    robustness_consistent = execution.verification_result.robustness_summary.certification_status in {
        "not_applicable",
        "nominal_only",
        "partial_robust",
        "robust_certified",
        "robustness_failed",
    }
    feedback_world = fake_world_model_consume(execution)
    feedback_plan = fake_planner_consume(execution)
    feedback_usable = feedback_world["truth_metric_count"] > 0 and feedback_plan["artifact_ref_count"] > 0

    if not schema_valid:
        failures.append(SimulationAcceptanceFailureRecord(code="schema_failure", message="simulation bundle/result validation failed"))
    if not measurement_correct:
        failures.append(SimulationAcceptanceFailureRecord(code="measurement_failure", message="no measurements extracted"))
    if not constraint_correct:
        failures.append(SimulationAcceptanceFailureRecord(code="constraint_failure", message="constraint adjudication missing"))
    if not diagnosis_correct:
        failures.append(SimulationAcceptanceFailureRecord(code="diagnosis_failure", message="failure attribution malformed"))
    if not robustness_consistent:
        failures.append(SimulationAcceptanceFailureRecord(code="robustness_failure", message="robustness certificate malformed"))
    if not feedback_usable:
        failures.append(SimulationAcceptanceFailureRecord(code="feedback_failure", message="feedback payload unusable downstream"))
    result = "pass" if not failures else "fail"
    return SimulationAcceptanceResult(
        case=case,
        schema_valid=schema_valid,
        measurement_correct=measurement_correct,
        constraint_correct=constraint_correct,
        diagnosis_correct=diagnosis_correct,
        robustness_consistent=robustness_consistent,
        feedback_usable=feedback_usable,
        result=result,
        failures=failures,
    )


def build_acceptance_summary(results: list[SimulationAcceptanceResult]) -> SimulationAcceptanceSummary:
    """Aggregate acceptance metrics for the fifth layer."""

    total = len(results)
    passed = sum(1 for result in results if result.result == "pass")
    counter = Counter(failure.code for result in results for failure in result.failures)
    return SimulationAcceptanceSummary(
        total_cases=total,
        passed_cases=passed,
        schema_validity_rate=sum(1 for result in results if result.schema_valid) / total if total else 0.0,
        measurement_correctness_rate=sum(1 for result in results if result.measurement_correct) / total if total else 0.0,
        constraint_accuracy_rate=sum(1 for result in results if result.constraint_correct) / total if total else 0.0,
        diagnosis_quality_rate=sum(1 for result in results if result.diagnosis_correct) / total if total else 0.0,
        robustness_consistency_rate=sum(1 for result in results if result.robustness_consistent) / total if total else 0.0,
        feedback_utility_rate=sum(1 for result in results if result.feedback_usable) / total if total else 0.0,
        failure_type_distribution=dict(counter),
    )
