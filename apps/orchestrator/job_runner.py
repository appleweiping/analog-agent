"""Job execution record helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from libs.eval.stats import aggregate_stats
from libs.memory.compiler import compile_memory_bundle
from libs.memory.service import MemoryService
from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_task import DesignTask
from libs.schema.system_binding import (
    AcceptanceSummary,
    AcceptanceTaskConfig,
    ArtifactTrace,
    BaselineComparisonSummary,
    CrossLayerTrace,
    PlanningStepSummary,
    PlanningTruthLoopResponse,
    StepTrace,
    SystemAcceptanceResult,
    WorldModelTruthBindingResponse,
)
from libs.simulation.service import SimulationService
from libs.simulation.compiler import normalize_fidelity_level
from libs.world_model.action_builder import build_design_action
from libs.world_model.compiler import compile_world_model_bundle
from libs.world_model.service import WorldModelService
from libs.world_model.state_builder import build_world_state_from_design_task
from libs.schema.world_model import DeltaFeatures, TransitionRecord
from libs.utils.hashing import stable_hash


def build_run_record(job_id: str, state: str) -> dict[str, str]:
    """Build a small, serializable job status record."""
    return {
        "job_id": job_id,
        "state": state,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def run_world_model_truth_binding(
    design_task: DesignTask,
    *,
    candidate_id: str | None = None,
    fidelity_level: str = "focused_truth",
    backend_preference: str = "ngspice",
    escalation_reason: str = "world_model_truth_binding",
) -> WorldModelTruthBindingResponse:
    """Run the formal L3 -> L5 -> L3 minimal closure path."""

    compiled_world_model = compile_world_model_bundle(design_task)
    if compiled_world_model.world_model_bundle is None:
        raise ValueError("world model bundle failed to compile")
    world_model_bundle = compiled_world_model.world_model_bundle
    world_model_service = WorldModelService(world_model_bundle, design_task)

    planning_bundle = compile_planning_bundle(design_task, world_model_bundle).planning_bundle
    if planning_bundle is None:
        raise ValueError("planning bundle failed to compile")
    planning_service = PlanningService(planning_bundle, design_task, world_model_bundle)
    search_state = planning_service.initialize_search().search_state
    selected_candidate_id = candidate_id or search_state.candidate_pool_state.candidates[0].candidate_id
    candidate = next(item for item in search_state.candidate_pool_state.candidates if item.candidate_id == selected_candidate_id)
    world_state = candidate.world_state_snapshot

    metrics_prediction = world_model_service.predict_metrics(world_state)
    feasibility_prediction = world_model_service.predict_feasibility(world_state)
    simulation_value_estimate = world_model_service.estimate_simulation_value(world_state)

    simulation_execution = SimulationService(design_task, planning_bundle, search_state).verify_candidate(
        selected_candidate_id,
        fidelity_level=fidelity_level,
        backend_preference=backend_preference,
        escalation_reason=escalation_reason,
    )

    truth_action = build_design_action(
        design_task,
        action_family="fidelity_upgrade",
        target_kind="evaluation_strategy",
        operator="promote_fidelity",
        payload={"target_fidelity": simulation_execution.verification_result.executed_fidelity},
        expected_scope=["evaluation_cost", "feasibility"],
        source="planner",
    )
    truth_state = world_model_service.build_truth_state_from_verification(world_state, simulation_execution.verification_result)
    truth_metrics = {metric.metric: metric.value for metric in world_state.performance_observation}
    simulated_metrics = {metric.metric: metric.value for metric in truth_state.performance_observation}
    delta_features = DeltaFeatures(
        metric_deltas={
            metric: round(simulated_metrics[metric] - truth_metrics.get(metric, 0.0), 6)
            for metric in simulated_metrics
        },
        margin_deltas={
            item.constraint_group: round(
                item.margin
                - next((previous.margin for previous in world_state.constraint_observation if previous.constraint_group == item.constraint_group), 0.0),
                6,
            )
            for item in truth_state.constraint_observation
        },
        operating_point_deltas={},
    )
    transition_record = TransitionRecord(
        transition_id=f"truth_transition_{stable_hash(f'{world_state.state_id}|{selected_candidate_id}|{simulation_execution.verification_result.result_id}')[:12]}",
        task_id=design_task.task_id,
        state_t=world_state,
        action_t=truth_action,
        state_t_plus_1=truth_state,
        delta_features=delta_features,
        predicted_metrics=metrics_prediction.metrics,
        ground_truth_metrics=truth_state.performance_observation,
        predicted_constraints=feasibility_prediction.per_group_constraints,
        ground_truth_constraints=truth_state.constraint_observation,
        analysis_fidelity="full_ground_truth",
        trust_snapshot=metrics_prediction.trust_assessment,
        raw_artifact_refs=list(simulation_execution.verification_result.artifact_refs),
        label_quality="high_fidelity",
    )
    calibration_update = world_model_service.calibrate_with_truth(
        world_state,
        simulation_execution.verification_result.calibration_payload.truth_record,
    )
    return WorldModelTruthBindingResponse(
        world_model_bundle=calibration_update.updated_bundle,
        world_state=world_state,
        prediction_action=truth_action,
        metrics_prediction=metrics_prediction,
        feasibility_prediction=feasibility_prediction,
        simulation_value_estimate=simulation_value_estimate,
        transition_record=transition_record,
        simulation_execution=simulation_execution,
        calibration_update=calibration_update,
    )


def _truth_metric_value(simulation_execution, metric_name: str = "gbw_hz") -> float:
    for metric in simulation_execution.verification_result.measurement_report.measured_metrics:
        if metric.metric == metric_name:
            return float(metric.value)
    return 0.0


def _selectable_candidates(search_state):
    return [
        candidate
        for candidate in search_state.candidate_pool_state.candidates
        if candidate.lifecycle_status in {"frontier", "best_feasible", "best_infeasible"}
        and candidate.predicted_uncertainty is not None
        and candidate.simulation_value_estimate is not None
    ]


def _baseline_candidates(search_state):
    """Candidates a full-simulation baseline would send to truth verification."""

    candidates = [
        candidate
        for candidate in _selectable_candidates(search_state)
        if candidate.predicted_uncertainty.service_tier != "hard_block"
    ]
    if candidates:
        return sorted(candidates, key=lambda item: item.priority_score, reverse=True)
    return []


def _run_full_simulation_baseline(
    design_task: DesignTask,
    planning_bundle,
    search_state,
    *,
    fidelity_level: str,
    backend_preference: str,
    escalation_reason: str,
):
    """Run a real-backend full-simulation baseline on all selectable candidates."""

    baseline_candidates = _baseline_candidates(search_state)
    simulation_service = SimulationService(design_task, planning_bundle, search_state)
    executions = [
        simulation_service.verify_candidate(
            candidate.candidate_id,
            fidelity_level=fidelity_level,
            backend_preference=backend_preference,
            escalation_reason=f"{escalation_reason}_baseline",
        )
        for candidate in baseline_candidates
    ]
    return baseline_candidates, executions


def run_planning_truth_loop(
    design_task: DesignTask,
    *,
    max_steps: int = 3,
    fidelity_level: str = "quick_truth",
    backend_preference: str = "ngspice",
    escalation_reason: str = "planning_truth_loop",
) -> PlanningTruthLoopResponse:
    """Run the Day-3 selective simulation loop with real ngspice verification."""

    compiled_world_model = compile_world_model_bundle(design_task)
    if compiled_world_model.world_model_bundle is None:
        raise ValueError("world model bundle failed to compile")
    world_model_bundle = compiled_world_model.world_model_bundle

    compiled_planning = compile_planning_bundle(design_task, world_model_bundle)
    if compiled_planning.planning_bundle is None:
        raise ValueError("planning bundle failed to compile")
    planning_bundle = compiled_planning.planning_bundle

    planning_service = PlanningService(planning_bundle, design_task, world_model_bundle)
    search_state = planning_service.initialize_search().search_state
    simulation_executions = []
    step_summaries: list[PlanningStepSummary] = []
    baseline_call_counter = 0

    for step_index in range(max_steps):
        search_state = planning_service.propose_candidates(search_state).search_state
        search_state = planning_service.evaluate_candidates(search_state).search_state
        baseline_call_counter += len(_baseline_candidates(search_state))
        selection = planning_service.select_for_simulation(search_state)
        search_state = selection.search_state

        simulated_candidate_ids: list[str] = []
        for candidate in selection.selected_candidates:
            requested_fidelity = selection.requested_fidelity_map.get(candidate.candidate_id, normalize_fidelity_level(fidelity_level))
            execution = SimulationService(design_task, planning_bundle, search_state).verify_candidate(
                candidate.candidate_id,
                fidelity_level=requested_fidelity,
                backend_preference=backend_preference,
                escalation_reason=f"{escalation_reason}:{requested_fidelity}",
            )
            simulation_executions.append(execution)
            simulated_candidate_ids.append(candidate.candidate_id)
            feedback = planning_service.ingest_simulation_feedback(
                search_state,
                candidate.candidate_id,
                execution.verification_result.calibration_payload.truth_record,
                execution.verification_result.planner_feedback,
            )
            search_state = feedback.search_state
            planning_service.world_model_bundle = feedback.updated_world_model_bundle
            planning_service.world_model_service.bundle = feedback.updated_world_model_bundle

        search_state = planning_service.advance_phase(search_state).search_state
        best_result = planning_service.get_best_result(search_state)
        best_candidate = best_result.candidate
        step_summaries.append(
            PlanningStepSummary(
                step_index=step_index,
                candidate_pool_size=len(search_state.candidate_pool_state.candidates),
                selected_for_simulation=[candidate.candidate_id for candidate in selection.selected_candidates],
                simulated_candidate_ids=simulated_candidate_ids,
                requested_fidelity=dict(selection.requested_fidelity_map),
                best_candidate_id=best_candidate.candidate_id if best_candidate is not None else None,
                best_priority_score=best_candidate.priority_score if best_candidate is not None else 0.0,
                simulation_calls_used=len(simulated_candidate_ids),
            )
        )
        if planning_service.should_terminate(search_state).should_terminate:
            break

    best_result = planning_service.get_best_result(search_state)
    best_candidate = best_result.candidate
    baseline_candidates, baseline_executions = _run_full_simulation_baseline(
        design_task,
        planning_bundle,
        search_state,
        fidelity_level=fidelity_level,
        backend_preference=backend_preference,
        escalation_reason=escalation_reason,
    )
    selective_best_metric = max((_truth_metric_value(execution) for execution in simulation_executions), default=0.0)
    baseline_best_metric = max((_truth_metric_value(execution) for execution in baseline_executions), default=0.0)
    comparison = BaselineComparisonSummary(
        selective_simulation_calls=len(simulation_executions),
        baseline_full_simulation_calls=max(baseline_call_counter, len(baseline_executions)),
        simulations_saved=max(0, max(baseline_call_counter, len(baseline_executions)) - len(simulation_executions)),
        selective_best_candidate_id=best_candidate.candidate_id if best_candidate is not None else None,
        baseline_best_candidate_id=baseline_candidates[0].candidate_id if baseline_candidates else None,
        selective_best_truth_metric=round(selective_best_metric, 6),
        baseline_best_truth_metric=round(baseline_best_metric, 6),
        selective_quality_ratio=round(selective_best_metric / max(baseline_best_metric, 1e-12), 6) if baseline_best_metric else 1.0,
    )
    return PlanningTruthLoopResponse(
        planning_bundle=planning_bundle,
        world_model_bundle=planning_service.world_model_bundle,
        final_search_state=search_state,
        best_result=best_result,
        best_candidate=best_candidate,
        simulation_executions=simulation_executions,
        step_summaries=step_summaries,
        comparison_summary=comparison,
    )


def _build_artifact_traces(
    execution: SimulationExecutionResponse,
) -> list[ArtifactTrace]:
    traces: list[ArtifactTrace] = []
    for record in execution.simulation_bundle.artifact_registry.records:
        provenance = record.simulation_provenance or execution.verification_result.simulation_provenance
        validation = record.validation_status or execution.verification_result.validation_status
        execution_context = record.execution_context
        traces.append(
            ArtifactTrace(
                artifact_id=record.artifact_id,
                artifact_type=record.artifact_type,
                artifact_path=record.path,
                candidate_id=execution.verification_result.candidate_id,
                simulation_id=execution.simulation_bundle.simulation_id,
                verification_result_id=execution.verification_result.result_id,
                model_reference=provenance.model_binding.backend_model_ref,
                truth_level=validation.truth_level,
                validation_status=validation.validity_state,
                invocation_mode=provenance.invocation_mode,
                resolved_simulator_binary=(
                    execution_context.resolved_simulator_binary
                    if execution_context is not None
                    else provenance.resolved_simulator_binary
                ),
                paper_mode=execution_context.paper_mode if execution_context is not None else provenance.paper_mode,
                paper_safe=execution_context.paper_safe if execution_context is not None else provenance.paper_safe,
                replayable=execution_context.replayable if execution_context is not None else False,
                replay_hint=execution_context.replay_hint if execution_context is not None else None,
            )
        )
    return traces


def _acceptance_summary(
    result: SystemAcceptanceResult,
    simulation_executions: list[SimulationExecutionResponse],
) -> AcceptanceSummary:
    cross_layer_ok = all(
        trace.candidate_id and trace.world_state_id and trace.simulation_id and trace.verification_result_id and trace.calibration_id
        for trace in result.cross_layer_traces
    )
    artifact_ok = bool(result.artifact_traces) and all(
        artifact.model_reference and artifact.truth_level and artifact.validation_status and artifact.invocation_mode
        for artifact in result.artifact_traces
    )
    fidelity_ok = all(
        trace.executed_fidelity in {"quick_truth", "focused_truth", "full_robustness_certification", "targeted_failure_analysis"}
        and trace.requested_fidelity
        for trace in result.cross_layer_traces
    )
    validation_ok = all(
        trace.truth_level in {"demonstrator_truth", "configured_truth"}
        and trace.validation_status in {"strong", "weak", "invalid"}
        for trace in result.cross_layer_traces
    )
    measurement_ok = all(
        statuses
        for step in result.step_traces
        for statuses in step.measurement_statuses.values()
    )
    feedback_ok = all(
        step.planner_updates and step.calibration_actions
        for step in result.step_traces
        if step.selected_candidate_ids
    )
    closed_loop = cross_layer_ok and artifact_ok and fidelity_ok and validation_ok and feedback_ok and result.episode_record is not None
    notes: list[str] = []
    if result.cross_layer_traces and all(trace.truth_level == "demonstrator_truth" for trace in result.cross_layer_traces):
        notes.append("current_e2e_runs_operate_at_demonstrator_truth_level")
    return AcceptanceSummary(
        schema_completeness_ok=result.memory_bundle.validation_status.is_valid and cross_layer_ok,
        backend_execution_validity_ok=all(execution.backend_report.is_available for execution in simulation_executions) if simulation_executions else False,
        measurement_correctness_ok=measurement_ok,
        fidelity_correctness_ok=fidelity_ok,
        validation_correctness_ok=validation_ok,
        feedback_consistency_ok=feedback_ok,
        system_closed_loop_established=closed_loop,
        simulation_execution_count=len(result.cross_layer_traces),
        step_count=len(result.step_traces),
        memory_episode_count=len(result.memory_bundle.episode_records),
        notes=notes,
    )


def run_full_system_acceptance(task_config: AcceptanceTaskConfig) -> SystemAcceptanceResult:
    """Run the formal Day-8 L2-L3-L4-L5-L6 end-to-end acceptance loop."""

    design_task = task_config.design_task
    compiled_world_model = compile_world_model_bundle(design_task)
    if compiled_world_model.world_model_bundle is None:
        raise ValueError("world model bundle failed to compile")
    world_model_bundle = compiled_world_model.world_model_bundle

    compiled_planning = compile_planning_bundle(design_task, world_model_bundle)
    if compiled_planning.planning_bundle is None:
        raise ValueError("planning bundle failed to compile")
    planning_bundle = compiled_planning.planning_bundle

    compiled_memory = compile_memory_bundle()
    if compiled_memory.memory_bundle is None:
        raise ValueError("memory bundle failed to compile")
    memory_service = MemoryService(compiled_memory.memory_bundle)

    planning_service = PlanningService(planning_bundle, design_task, world_model_bundle)
    search_state = planning_service.initialize_search().search_state

    step_traces: list[StepTrace] = []
    cross_layer_traces: list[CrossLayerTrace] = []
    artifact_traces: list[ArtifactTrace] = []
    simulation_executions: list[SimulationExecutionResponse] = []
    verification_stats = []
    verification_by_candidate: dict[str, SimulationExecutionResponse] = {}

    for step_index in range(task_config.max_steps):
        phase_before = search_state.phase_state.current_phase
        candidate_pool_size_before = len(search_state.candidate_pool_state.candidates)
        search_state = planning_service.propose_candidates(search_state).search_state
        search_state = planning_service.evaluate_candidates(search_state).search_state
        selection = planning_service.select_for_simulation(search_state)
        search_state = selection.search_state

        selected_candidate_ids = [candidate.candidate_id for candidate in selection.selected_candidates]
        measurement_statuses: dict[str, list[str]] = {}
        validation_states: dict[str, str] = {}
        planner_updates: dict[str, str] = {}
        calibration_actions: dict[str, str] = {}
        verification_result_ids: list[str] = []

        for candidate in selection.selected_candidates:
            requested_fidelity = selection.requested_fidelity_map.get(candidate.candidate_id, normalize_fidelity_level(task_config.default_fidelity))
            if step_index == 0 and not simulation_executions:
                requested_fidelity = normalize_fidelity_level(task_config.default_fidelity)
            execution = SimulationService(design_task, planning_bundle, search_state).verify_candidate(
                candidate.candidate_id,
                fidelity_level=requested_fidelity,
                backend_preference=task_config.backend_preference,
                escalation_reason=f"{task_config.escalation_reason}:{requested_fidelity}",
            )
            simulation_executions.append(execution)
            verification_stats.append(execution.verification_stats)
            verification_by_candidate[candidate.candidate_id] = execution
            verification_result_ids.append(execution.verification_result.result_id)
            measurement_statuses[candidate.candidate_id] = [
                f"{item.metric}:{item.status.status}:{item.failure_reason.code}"
                for item in execution.verification_result.measurement_report.measurement_results
            ]
            validation_states[candidate.candidate_id] = execution.verification_result.validation_status.validity_state
            planner_updates[candidate.candidate_id] = execution.verification_result.planner_feedback.lifecycle_update
            calibration_actions[candidate.candidate_id] = execution.verification_result.calibration_payload.retrain_priority

            feedback = planning_service.ingest_simulation_feedback(
                search_state,
                candidate.candidate_id,
                execution.verification_result.calibration_payload.truth_record,
                execution.verification_result.planner_feedback,
            )
            search_state = feedback.search_state
            planning_service.world_model_bundle = feedback.updated_world_model_bundle
            planning_service.world_model_service.bundle = feedback.updated_world_model_bundle

            cross_layer_traces.append(
                CrossLayerTrace(
                    task_id=design_task.task_id,
                    search_id=search_state.search_id,
                    planning_id=planning_bundle.planning_id,
                    world_model_id=planning_service.world_model_bundle.world_model_id,
                    candidate_id=candidate.candidate_id,
                    parent_candidate_id=candidate.parent_candidate_id,
                    world_state_id=candidate.world_state_ref,
                    simulation_request_id=execution.simulation_request.request_id,
                    simulation_id=execution.simulation_bundle.simulation_id,
                    verification_result_id=execution.verification_result.result_id,
                    calibration_id=execution.verification_result.calibration_payload.calibration_id,
                    requested_fidelity=requested_fidelity,
                    executed_fidelity=execution.verification_result.executed_fidelity,
                    truth_level=execution.verification_result.validation_status.truth_level,
                    validation_status=execution.verification_result.validation_status.validity_state,
                    planner_lifecycle_update=execution.verification_result.planner_feedback.lifecycle_update,
                    memory_recorded=False,
                )
            )
            artifact_traces.extend(_build_artifact_traces(execution))

        search_state = planning_service.advance_phase(search_state).search_state
        step_traces.append(
            StepTrace(
                step_index=step_index,
                phase_before=phase_before,
                phase_after=search_state.phase_state.current_phase,
                candidate_pool_size_before=candidate_pool_size_before,
                candidate_pool_size_after=len(search_state.candidate_pool_state.candidates),
                selected_candidate_ids=selected_candidate_ids,
                requested_fidelity=dict(selection.requested_fidelity_map),
                verification_result_ids=verification_result_ids,
                measurement_statuses=measurement_statuses,
                validation_states=validation_states,
                planner_updates=planner_updates,
                calibration_actions=calibration_actions,
                notes=[f"phase={search_state.phase_state.current_phase}"],
            )
        )
        if planning_service.should_terminate(search_state).should_terminate:
            break

    best_result = planning_service.get_best_result(search_state)
    best_candidate = best_result.candidate
    final_execution = None
    if best_candidate is not None:
        final_execution = verification_by_candidate.get(best_candidate.candidate_id)
    if final_execution is None and simulation_executions:
        final_execution = simulation_executions[-1]

    ingestion = memory_service.ingest_episode(
        design_task,
        search_state,
        final_execution.verification_result if final_execution is not None else None,
    )

    for index, trace in enumerate(cross_layer_traces):
        cross_layer_traces[index] = trace.model_copy(
            update={
                "episode_memory_id": ingestion.episode_record.episode_memory_id,
                "memory_recorded": True,
            }
        )

    provisional_result = SystemAcceptanceResult(
        task_id=design_task.task_id,
        world_model_id=planning_service.world_model_bundle.world_model_id,
        planning_id=planning_bundle.planning_id,
        memory_id=ingestion.memory_bundle.memory_id,
        search_id=search_state.search_id,
        best_candidate_id=best_candidate.candidate_id if best_candidate is not None else None,
        best_feasible_found=best_result.summary.get("feasible_found", False) is True,
        final_verification_result_id=final_execution.verification_result.result_id if final_execution is not None else None,
        episode_memory_id=ingestion.episode_record.episode_memory_id,
        memory_bundle=ingestion.memory_bundle,
        episode_record=ingestion.episode_record,
        step_traces=step_traces,
        cross_layer_traces=cross_layer_traces,
        artifact_traces=artifact_traces,
        verification_stats=verification_stats,
        stats_summary=None,
        acceptance_summary=AcceptanceSummary(
            schema_completeness_ok=False,
            backend_execution_validity_ok=False,
            measurement_correctness_ok=False,
            fidelity_correctness_ok=False,
            validation_correctness_ok=False,
            feedback_consistency_ok=False,
            system_closed_loop_established=False,
            simulation_execution_count=0,
            step_count=0,
            memory_episode_count=0,
            notes=[],
        ),
    )
    return provisional_result.model_copy(
        update={
            "stats_summary": aggregate_stats(provisional_result),
            "acceptance_summary": _acceptance_summary(provisional_result, simulation_executions),
        }
    )
