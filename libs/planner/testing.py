"""Testing helpers for the planning layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_task import DesignTask
from libs.schema.planning import PlanningAcceptanceFailureRecord, PlanningAcceptanceSummary, PlanningBestResult, SearchState
from libs.schema.world_model import TruthCalibrationRecord, TruthConstraint, TruthMetric
from libs.world_model.compiler import compile_world_model_bundle


@dataclass(slots=True)
class PlanningAcceptanceCase:
    name: str
    category: str
    design_task: DesignTask
    max_rounds: int = 2
    inject_feedback: bool = False


@dataclass(slots=True)
class PlanningAcceptanceResult:
    name: str
    schema_valid: bool
    decision_quality_ok: bool
    budget_efficient: bool
    feasibility_progressed: bool
    world_model_safe: bool
    trace_complete: bool
    proxy_evaluations_used: int
    simulations_used: int
    feasible_found: bool
    failures: list[PlanningAcceptanceFailureRecord] = field(default_factory=list)


def fake_simulator_feedback(search_state: SearchState, candidate_id: str) -> TruthCalibrationRecord:
    """Build deterministic fake simulator feedback for one candidate."""

    candidate = next(candidate for candidate in search_state.candidate_pool_state.candidates if candidate.candidate_id == candidate_id)
    metrics = []
    if candidate.predicted_metrics is not None:
        for item in candidate.predicted_metrics.metrics[:3]:
            metrics.append(TruthMetric(metric=item.metric, value=item.value * 0.97))
    constraints = []
    if candidate.predicted_feasibility is not None:
        for constraint in candidate.predicted_feasibility.per_group_constraints:
            constraints.append(
                TruthConstraint(
                    constraint_name=constraint.constraint_name,
                    constraint_group=constraint.constraint_group,
                    is_satisfied=constraint.margin >= 0.0,
                    margin=constraint.margin,
                )
            )
    return TruthCalibrationRecord(
        simulator_signature="fake-simulator",
        analysis_fidelity="full_ground_truth",
        metrics=metrics,
        constraints=constraints,
        artifact_refs=[f"artifact://simulation/{candidate_id}"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def fake_orchestrator_run(case: PlanningAcceptanceCase) -> tuple[PlanningService, SearchState, PlanningBestResult]:
    """Run a lightweight orchestrator loop over the planning service."""

    world_model_bundle = compile_world_model_bundle(case.design_task).world_model_bundle
    assert world_model_bundle is not None
    planning_bundle = compile_planning_bundle(case.design_task, world_model_bundle).planning_bundle
    assert planning_bundle is not None
    service = PlanningService(planning_bundle, case.design_task, world_model_bundle)
    initialized = service.initialize_search()
    state = initialized.search_state

    for _ in range(case.max_rounds):
        state = service.propose_candidates(state).search_state
        state = service.evaluate_candidates(state).search_state
        selection = service.select_for_simulation(state)
        state = selection.search_state
        if case.inject_feedback and selection.selected_candidates:
            feedback = fake_simulator_feedback(state, selection.selected_candidates[0].candidate_id)
            state = service.ingest_simulation_feedback(state, selection.selected_candidates[0].candidate_id, feedback).search_state
        state = service.advance_phase(state).search_state
        if service.should_terminate(state).should_terminate:
            break

    return service, state, service.get_best_result(state)


def evaluate_case(case: PlanningAcceptanceCase) -> PlanningAcceptanceResult:
    """Evaluate one planning acceptance case."""

    service, state, best = fake_orchestrator_run(case)
    validation = service.validate_search_state(state)
    decision_quality_ok = any(candidate.priority_score >= 0.0 for candidate in state.candidate_pool_state.candidates)
    budget_efficient = state.budget_state.simulations_used <= state.budget_state.simulation_budget
    feasibility_progressed = state.best_known_feasible is not None or state.best_known_infeasible is not None
    world_model_safe = not state.risk_context.calibration_required or state.phase_state.current_phase == "calibration_recovery"
    trace_complete = len(state.trace_log) > 0 and all(trace.outcome_tag for trace in state.trace_log)
    failures: list[PlanningAcceptanceFailureRecord] = []
    if not validation.is_valid:
        failures.append(PlanningAcceptanceFailureRecord(code="schema_failure", message="search state validation failed"))
    if not decision_quality_ok:
        failures.append(PlanningAcceptanceFailureRecord(code="decision_quality_failure", message="candidate priorities were not established"))
    if not budget_efficient:
        failures.append(PlanningAcceptanceFailureRecord(code="budget_misallocation", message="simulation budget was exceeded"))
    if not feasibility_progressed:
        failures.append(PlanningAcceptanceFailureRecord(code="feasibility_progression_failure", message="planner did not establish feasible or boundary candidates"))
    if not world_model_safe:
        failures.append(PlanningAcceptanceFailureRecord(code="world_model_safety_failure", message="planner ignored world-model trust degradation"))
    if not trace_complete:
        failures.append(PlanningAcceptanceFailureRecord(code="traceability_failure", message="trace log is incomplete"))

    return PlanningAcceptanceResult(
        name=case.name,
        schema_valid=validation.is_valid,
        decision_quality_ok=decision_quality_ok,
        budget_efficient=budget_efficient,
        feasibility_progressed=feasibility_progressed,
        world_model_safe=world_model_safe,
        trace_complete=trace_complete,
        proxy_evaluations_used=state.budget_state.proxy_evaluations_used,
        simulations_used=state.budget_state.simulations_used,
        feasible_found=best.candidate is not None and best.summary.get("feasible_found", False) is True,
        failures=failures,
    )


def build_acceptance_summary(results: list[PlanningAcceptanceResult]) -> PlanningAcceptanceSummary:
    """Aggregate planning acceptance results."""

    total = max(1, len(results))
    all_failures = [failure for result in results for failure in result.failures]
    return PlanningAcceptanceSummary(
        total_cases=len(results),
        schema_validity_rate=sum(1 for result in results if result.schema_valid) / total,
        decision_quality_rate=sum(1 for result in results if result.decision_quality_ok) / total,
        budget_efficiency_rate=sum(1 for result in results if result.budget_efficient) / total,
        feasibility_progression_rate=sum(1 for result in results if result.feasibility_progressed) / total,
        world_model_safety_rate=sum(1 for result in results if result.world_model_safe) / total,
        traceability_rate=sum(1 for result in results if result.trace_complete) / total,
        average_proxy_evaluations=sum(result.proxy_evaluations_used for result in results) / total,
        average_simulations=sum(result.simulations_used for result in results) / total,
        feasible_hit_rate=sum(1 for result in results if result.feasible_found) / total,
        failures=all_failures,
    )

