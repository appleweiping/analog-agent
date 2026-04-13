"""Unified experiment runner for baseline and methodology comparisons."""

from __future__ import annotations

from collections import Counter, defaultdict

from libs.eval.metrics import (
    aggregate_failure_type_distribution,
    efficiency_score,
    feasible_hit_rate,
)
from libs.eval.bayesopt import build_observation as build_bayesopt_observation
from libs.eval.bayesopt import build_surrogate_predictions, propose_parameter_batch
from libs.eval.cmaes import build_observation as build_cmaes_observation
from libs.eval.cmaes import build_surrogate_predictions as build_cmaes_surrogate_predictions
from libs.eval.cmaes import propose_parameter_batch as propose_cmaes_parameter_batch
from libs.eval.random_search import deterministic_unit, sample_parameter_values
from libs.eval.rl_baseline import build_observation as build_rl_observation
from libs.eval.rl_baseline import build_surrogate_predictions as build_rl_surrogate_predictions
from libs.eval.rl_baseline import propose_parameter_batch as propose_rl_parameter_batch
from libs.eval.stats import aggregate_stats, build_experiment_stats_record
from libs.planner.budget_controller import initialize_budget_state, remaining_simulations
from libs.planner.candidate_manager import append_decision_event, append_evaluation_event
from libs.planner.candidate_manager import frontier_candidates, summarize_candidate, upsert_candidate
from libs.planner.compiler import compile_planning_bundle
from libs.planner.phase_controller import initialize_phase_state
from libs.planner.rollout_planner import build_candidate_actions
from libs.planner.selection_engine import apply_priority_scores
from libs.planner.service import PlanningService, _timestamp
from libs.schema.experiment import (
    ExperimentAggregateSummary,
    ExperimentBudget,
    ExperimentLogRecord,
    ExperimentMode,
    ExperimentResult,
    ExperimentSuiteResult,
    MethodComparisonResult,
    MethodComponentConfig,
    MethodConclusionSummary,
    MethodDeltaSummary,
    MethodModeSummary,
)
from libs.schema.planning import CandidatePoolState, FrontierState, SearchProvenance, SearchState, SimulationDecision, SimulationSelectionResponse, StrategyContext
from libs.schema.world_model import FeasibilityPrediction, MetricEstimate, MetricsPrediction, SimulationValueEstimate, TrustAssessment
from libs.simulation.service import SimulationService
from libs.utils.hashing import stable_hash
from libs.world_model.compiler import compile_world_model_bundle
from libs.world_model.feature_projection import evaluate_constraints
from libs.world_model.state_builder import build_world_state


def _pseudo_unit(seed: str) -> float:
    return deterministic_unit(seed)


def _parameter_map(candidate) -> dict[str, float | int | str | bool]:
    return {parameter.variable_name: parameter.value for parameter in candidate.world_state_snapshot.parameter_state}


def _apply_action(values: dict[str, float | int | str | bool], action) -> dict[str, float | int | str | bool]:
    updated = dict(values)
    targets = action.action_target.variable_names
    if action.action_operator in {"freeze", "unfreeze", "promote_fidelity", "drop_candidate"}:
        return updated
    for target in targets:
        current = updated[target]
        if action.action_operator == "set":
            updated[target] = action.action_payload.get("value", current)
        elif action.action_operator == "shift":
            updated[target] = float(current) + float(action.action_payload.get("delta", 0.0))
        elif action.action_operator == "scale":
            updated[target] = float(current) * float(action.action_payload.get("factor", 1.0))
        elif action.action_operator == "swap":
            updated[target] = action.action_payload.get("value", current)
    return updated


def _relevant_metrics(task) -> list[str]:
    metrics = {term.metric for term in task.objective.terms}
    metrics.update(constraint.metric for constraint in task.constraints.hard_constraints)
    metrics.update(constraint.metric for constraint in task.constraints.soft_constraints)
    metrics.update(task.objective.reporting_metrics)
    metrics.update({"dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w"})
    return sorted(metrics)


def _baseline_metric_value(metric: str, seed: str) -> float:
    ratio = _pseudo_unit(f"{metric}|{seed}")
    ranges = {
        "dc_gain_db": (45.0, 120.0),
        "gbw_hz": (4.0e7, 2.0e8),
        "phase_margin_deg": (40.0, 85.0),
        "power_w": (8.0e-5, 2.0e-3),
        "output_swing_v": (0.5, 1.1),
        "psrr_db": (30.0, 90.0),
    }
    low, high = ranges.get(metric, (0.1, 1.0))
    return round(low + (high - low) * ratio, 6)


def _baseline_trust(mode: str, seed: str) -> TrustAssessment:
    if mode == "full_simulation_baseline":
        return TrustAssessment(
            trust_level="low",
            service_tier="must_escalate",
            confidence=0.15,
            uncertainty_score=0.95,
            ood_score=0.5,
            must_escalate=True,
            hard_block=False,
            reasons=["full_simulation_baseline"],
        )
    if mode == "random_search_baseline":
        ratio = _pseudo_unit(seed)
        return TrustAssessment(
            trust_level="low",
            service_tier="screening_only",
            confidence=round(0.1 + 0.05 * ratio, 4),
            uncertainty_score=round(0.75 + 0.15 * (1.0 - ratio), 4),
            ood_score=round(0.45 + 0.1 * ratio, 4),
            must_escalate=False,
            hard_block=False,
            reasons=["random_search_baseline"],
        )
    ratio = _pseudo_unit(seed)
    return TrustAssessment(
        trust_level="low",
        service_tier="ranking_ready",
        confidence=round(0.25 + 0.15 * ratio, 4),
        uncertainty_score=round(0.45 + 0.1 * (1.0 - ratio), 4),
        ood_score=round(0.35 + 0.1 * ratio, 4),
        must_escalate=False,
        hard_block=False,
        reasons=["no_world_model_baseline"],
    )


def _baseline_predictions(task, candidate, mode: str, run_index: int):
    seed = f"{candidate.candidate_id}|{mode}|{run_index}"
    trust = _baseline_trust(mode, seed)
    metrics = [
        MetricEstimate(
            metric=metric,
            value=_baseline_metric_value(metric, seed),
            lower_bound=_baseline_metric_value(metric, seed) * 0.95,
            upper_bound=_baseline_metric_value(metric, seed) * 1.05,
            uncertainty=trust.uncertainty_score,
            trust_level=trust.trust_level,
            source="prediction",
        )
        for metric in _relevant_metrics(task)
    ]
    metric_map = {metric.metric: metric.value for metric in metrics}
    constraints = [
        item.model_copy(update={"source": "prediction"})
        for item in evaluate_constraints(task, metric_map)
    ]
    overall = 1.0
    failure_reasons: list[str] = []
    for item in constraints:
        overall *= max(0.05, item.satisfied_probability)
        if item.margin < 0.0:
            failure_reasons.append(item.constraint_name)
    metrics_prediction = MetricsPrediction(
        state_id=candidate.world_state_ref,
        task_id=task.task_id,
        metrics=metrics,
        auxiliary_features={"baseline_randomness": _pseudo_unit(seed)},
        trust_assessment=trust,
    )
    feasibility_prediction = FeasibilityPrediction(
        state_id=candidate.world_state_ref,
        task_id=task.task_id,
        overall_feasibility=overall,
        per_group_constraints=constraints,
        most_likely_failure_reasons=failure_reasons,
        confidence=trust.confidence,
        trust_assessment=trust,
    )
    if mode == "full_simulation_baseline":
        simulation_value = SimulationValueEstimate(
            state_id=candidate.world_state_ref,
            estimated_value=1.0,
            decision="prioritize",
            reasons=["full_simulation_baseline"],
            trust_assessment=trust,
        )
    elif mode == "random_search_baseline":
        simulation_value = SimulationValueEstimate(
            state_id=candidate.world_state_ref,
            estimated_value=1.0,
            decision="simulate",
            reasons=["random_search_baseline"],
            trust_assessment=trust,
        )
    else:
        value = round(0.35 + 0.4 * _pseudo_unit(f"sim-value|{seed}"), 6)
        decision = "simulate" if value >= 0.55 else "defer"
        simulation_value = SimulationValueEstimate(
            state_id=candidate.world_state_ref,
            estimated_value=value,
            decision=decision,
            reasons=["no_world_model_baseline"],
            trust_assessment=trust,
        )
    return metrics_prediction, feasibility_prediction, simulation_value


def _mode_config(mode: ExperimentMode) -> MethodComponentConfig:
    if mode == "full_system":
        return MethodComponentConfig(
            mode=mode,
            use_world_model=True,
            use_calibration=True,
            use_fidelity_escalation=True,
            use_full_simulation_baseline=False,
        )
    if mode in {"no_world_model", "no_world_model_baseline"}:
        return MethodComponentConfig(
            mode=mode,
            use_world_model=False,
            use_calibration=False,
            use_fidelity_escalation=True,
            use_full_simulation_baseline=False,
        )
    if mode == "random_search_baseline":
        return MethodComponentConfig(
            mode=mode,
            use_world_model=False,
            use_calibration=False,
            use_fidelity_escalation=False,
            use_full_simulation_baseline=False,
            use_random_search_baseline=True,
        )
    if mode == "bayesopt_baseline":
        return MethodComponentConfig(
            mode=mode,
            use_world_model=False,
            use_calibration=False,
            use_fidelity_escalation=False,
            use_full_simulation_baseline=False,
            use_random_search_baseline=False,
            use_bayesopt_baseline=True,
        )
    if mode == "cmaes_baseline":
        return MethodComponentConfig(
            mode=mode,
            use_world_model=False,
            use_calibration=False,
            use_fidelity_escalation=False,
            use_full_simulation_baseline=False,
            use_random_search_baseline=False,
            use_bayesopt_baseline=False,
            use_cmaes_baseline=True,
        )
    if mode == "rl_baseline":
        return MethodComponentConfig(
            mode=mode,
            use_world_model=False,
            use_calibration=False,
            use_fidelity_escalation=False,
            use_full_simulation_baseline=False,
            use_random_search_baseline=False,
            use_bayesopt_baseline=False,
            use_cmaes_baseline=False,
            use_rl_baseline=True,
        )
    if mode == "no_calibration":
        return MethodComponentConfig(
            mode=mode,
            use_world_model=True,
            use_calibration=False,
            use_fidelity_escalation=True,
            use_full_simulation_baseline=False,
        )
    if mode == "no_fidelity_escalation":
        return MethodComponentConfig(
            mode=mode,
            use_world_model=True,
            use_calibration=True,
            use_fidelity_escalation=False,
            use_full_simulation_baseline=False,
        )
    if mode == "full_simulation_baseline":
        return MethodComponentConfig(
            mode=mode,
            use_world_model=False,
            use_calibration=False,
            use_fidelity_escalation=False,
            use_full_simulation_baseline=True,
        )
    raise ValueError(f"unsupported experiment mode: {mode}")


def _uses_baseline_predictions(config: MethodComponentConfig) -> bool:
    return config.use_full_simulation_baseline or (
        not config.use_world_model
        and not config.use_random_search_baseline
        and not config.use_bayesopt_baseline
        and not config.use_cmaes_baseline
        and not config.use_rl_baseline
    )


def _requested_fidelity(
    config: MethodComponentConfig,
    selection,
    candidate,
    default_fidelity: str,
) -> str:
    if not config.use_fidelity_escalation:
        return "quick_truth"
    return selection.requested_fidelity_map.get(candidate.candidate_id, default_fidelity)


def _initialize_baseline_state(service: PlanningService, mode: str, run_index: int) -> SearchState:
    task = service.task
    seed_states = []
    if task.initial_state.seed_candidates:
        for seed in task.initial_state.seed_candidates:
            parameter_values = {**task.initial_state.template_defaults, **seed.values}
            seed_states.append(build_world_state(task, parameter_values=parameter_values, provenance_type="trajectory_replay", provenance_stage="initial"))
    if not seed_states:
        seed_states.append(build_world_state(task, parameter_values=task.initial_state.template_defaults, provenance_type="trajectory_replay", provenance_stage="initial"))

    budget_state = initialize_budget_state(service.planning_bundle.budget_controller)
    phase_state = initialize_phase_state(service.planning_bundle.phase_controller)
    records = [
        service._candidate_record(state, parent_candidate_id=None, proposal_source="initial_seed", action_chain=[], generation_depth=0)
        for state in seed_states
    ]
    records, budget_state = _evaluate_mode_candidates(service, records, mode, run_index, budget_state)
    pool_state = CandidatePoolState(candidates=[], active_candidate_ids=[], archived_candidate_ids=[], discarded_candidate_ids=[])
    for record in records:
        pool_state = upsert_candidate(pool_state, record)
    frontier_ids = [record.candidate_id for record in records if record.lifecycle_status == "frontier"]
    current = records[0].world_state_snapshot
    return SearchState(
        search_id=f"search_{stable_hash(service.planning_bundle.planning_id + mode)[:12]}",
        task_id=task.task_id,
        episode_id=f"episode_{stable_hash(task.task_id + mode + str(run_index))[:12]}",
        current_world_state=current,
        candidate_pool_state=pool_state,
        frontier_state=FrontierState(
            frontier_candidate_ids=frontier_ids,
            expansion_round=0,
            max_frontier_size=max(4, service.planning_bundle.budget_controller.batch_size * 3),
        ),
        evaluated_state_refs=[record.world_state_ref for record in records],
        pending_simulation_refs=[],
        budget_state=budget_state,
        phase_state=phase_state,
        best_known_feasible=summarize_candidate(records[0]) if frontier_ids else None,
        best_known_infeasible=None,
        strategy_context=StrategyContext(
            active_policy=mode,
            exploration_enabled=True,
            rollout_enabled=False,
            last_selected_candidate_id=records[0].candidate_id if records else None,
            notes=[f"mode={mode}"],
        ),
        risk_context=service._risk_context(records),
        provenance=SearchProvenance(source="initialized", created_at=_timestamp(), artifact_refs=[]),
        trace_log=[],
    )


def _initialize_random_search_state(service: PlanningService, mode: str, run_index: int) -> SearchState:
    current = build_world_state(
        service.task,
        parameter_values=service.task.initial_state.template_defaults,
        provenance_type="trajectory_replay",
        provenance_stage="initial",
    )
    return SearchState(
        search_id=f"search_{stable_hash(service.planning_bundle.planning_id + mode)[:12]}",
        task_id=service.task.task_id,
        episode_id=f"episode_{stable_hash(service.task.task_id + mode + str(run_index))[:12]}",
        current_world_state=current,
        candidate_pool_state=CandidatePoolState(candidates=[], active_candidate_ids=[], archived_candidate_ids=[], discarded_candidate_ids=[]),
        frontier_state=FrontierState(
            frontier_candidate_ids=[],
            expansion_round=0,
            max_frontier_size=max(4, service.planning_bundle.budget_controller.batch_size * 3),
        ),
        evaluated_state_refs=[],
        pending_simulation_refs=[],
        budget_state=initialize_budget_state(service.planning_bundle.budget_controller),
        phase_state=initialize_phase_state(service.planning_bundle.phase_controller),
        best_known_feasible=None,
        best_known_infeasible=None,
        strategy_context=StrategyContext(
            active_policy=mode,
            exploration_enabled=True,
            rollout_enabled=False,
            last_selected_candidate_id=None,
            notes=[f"mode={mode}", "random_search_initialized"],
        ),
        risk_context=service._risk_context([]),
        provenance=SearchProvenance(source="initialized", created_at=_timestamp(), artifact_refs=[]),
        trace_log=[],
    )


def _evaluate_mode_candidates(service: PlanningService, candidates, mode: str, run_index: int, budget_state):
    if mode == "full_system":
        return service._evaluate_records(candidates, budget_state)

    evaluated = []
    ranking_scores = {}
    for candidate in candidates:
        metrics_prediction, feasibility_prediction, simulation_value = _baseline_predictions(service.task, candidate, mode, run_index)
        lifecycle = "frontier" if mode == "no_world_model_baseline" else "frontier"
        updated = candidate.model_copy(
            update={
                "predicted_metrics": metrics_prediction,
                "predicted_feasibility": feasibility_prediction,
                "predicted_uncertainty": feasibility_prediction.trust_assessment,
                "simulation_value_estimate": simulation_value,
                "lifecycle_state": lifecycle,
                "lifecycle_status": lifecycle,
            }
        )
        updated = append_evaluation_event(updated, "predicted", [f"mode={mode}"])
        evaluated.append(updated)
        ranking_scores[updated.world_state_ref] = 0.0 if mode in {"full_simulation_baseline", "random_search_baseline"} else round(_pseudo_unit(updated.candidate_id + str(run_index)), 6)
    if mode == "no_world_model_baseline":
        evaluated = apply_priority_scores(evaluated, ranking_scores=ranking_scores, policy=service.planning_bundle.selection_policy)
        from libs.planner.budget_controller import consume_proxy_evaluations

        budget_state = consume_proxy_evaluations(budget_state, len(candidates))
    return evaluated, budget_state


def _propose_baseline_candidates(service: PlanningService, search_state: SearchState, mode: str, run_index: int):
    frontier = frontier_candidates(search_state.candidate_pool_state)
    frontier = sorted(frontier, key=lambda item: item.priority_score, reverse=True)[: service.planning_bundle.budget_controller.batch_size]
    new_records = []
    for anchor in frontier:
        actions = build_candidate_actions(
            service.task,
            anchor.world_state_snapshot,
            search_state.phase_state.current_phase,
            service.planning_bundle.rollout_config.max_branching_factor,
        )
        for action in actions[: service.planning_bundle.rollout_config.max_branching_factor]:
            parameter_values = _apply_action(_parameter_map(anchor), action)
            next_state = build_world_state(
                service.task,
                parameter_values=parameter_values,
                corner=anchor.world_state_snapshot.environment_state.corner,
                temperature_c=anchor.world_state_snapshot.environment_state.temperature_c,
                analysis_fidelity=anchor.world_state_snapshot.evaluation_context.analysis_fidelity,
                analysis_intent=anchor.world_state_snapshot.evaluation_context.analysis_intent,
                output_load_ohm=anchor.world_state_snapshot.environment_state.output_load_ohm,
                provenance_type="trajectory_replay",
                provenance_stage="predicted",
                artifact_refs=anchor.world_state_snapshot.provenance.artifact_refs,
                recent_actions=anchor.world_state_snapshot.history_context.recent_actions,
            )
            new_records.append(
                service._candidate_record(
                    next_state,
                    parent_candidate_id=anchor.candidate_id,
                    proposal_source="planner_mutation",
                    action_chain=[*anchor.proposal_action_chain, action],
                    generation_depth=anchor.generation_depth + 1,
                )
            )
    new_records, budget_state = _evaluate_mode_candidates(service, new_records, mode, run_index, search_state.budget_state)
    pool_state = search_state.candidate_pool_state
    for candidate in new_records:
        pool_state = upsert_candidate(pool_state, candidate)
    trace = service._make_trace(
        search_state,
        outcome_tag="candidate_proposed",
        selected_candidate_id=frontier[0].candidate_id if frontier else None,
        executed_action_chain=new_records[0].proposal_action_chain if new_records else [],
        world_model_queries=[],
        simulation_decision=SimulationDecision(
            decision="keep",
            candidate_ids=[candidate.candidate_id for candidate in new_records],
            reasons=[f"baseline proposal mode={mode}"],
        ),
        decision_rationale=[f"baseline proposal mode={mode}"],
        reward_or_progress_signal=max((candidate.priority_score for candidate in new_records), default=0.0),
        trust_snapshot=new_records[0].predicted_uncertainty if new_records else None,
    )
    updated_state = service._refresh_search_state(
        search_state,
        pool_state=pool_state,
        frontier_ids=[candidate.candidate_id for candidate in frontier_candidates(pool_state)],
        budget_state=budget_state,
        provenance_source="candidate_proposal",
        traces=[trace],
    )
    return updated_state


def _propose_random_search_candidates(service: PlanningService, search_state: SearchState, run_index: int, step_index: int):
    batch_size = min(
        service.planning_bundle.budget_controller.batch_size,
        remaining_simulations(search_state.budget_state),
    )
    if batch_size <= 0:
        return search_state
    new_records = []
    for sample_index in range(batch_size):
        parameter_values = sample_parameter_values(
            service.task,
            run_label="random_search_baseline",
            step_index=step_index,
            sample_index=(run_index * max(1, batch_size)) + sample_index,
        )
        next_state = build_world_state(
            service.task,
            parameter_values=parameter_values,
            provenance_type="trajectory_replay",
            provenance_stage="predicted",
        )
        new_records.append(
            service._candidate_record(
                next_state,
                parent_candidate_id=None,
                proposal_source="restart_policy",
                action_chain=[],
                generation_depth=0,
            )
        )
    new_records, budget_state = _evaluate_mode_candidates(service, new_records, "random_search_baseline", run_index, search_state.budget_state)
    pool_state = search_state.candidate_pool_state
    for candidate in new_records:
        pool_state = upsert_candidate(pool_state, candidate)
    trace = service._make_trace(
        search_state,
        outcome_tag="candidate_proposed",
        selected_candidate_id=new_records[0].candidate_id if new_records else None,
        executed_action_chain=[],
        world_model_queries=[],
        simulation_decision=SimulationDecision(
            decision="keep",
            candidate_ids=[candidate.candidate_id for candidate in new_records],
            reasons=["random_search_baseline"],
        ),
        decision_rationale=["sampled candidate batch uniformly from DesignTask.design_space"],
        reward_or_progress_signal=max((candidate.priority_score for candidate in new_records), default=0.0),
        trust_snapshot=new_records[0].predicted_uncertainty if new_records else None,
    )
    return service._refresh_search_state(
        search_state,
        pool_state=pool_state,
        frontier_ids=[candidate.candidate_id for candidate in new_records],
        budget_state=budget_state,
        provenance_source="candidate_proposal",
        traces=[trace],
    )


def _propose_bayesopt_candidates(service: PlanningService, search_state: SearchState, observations, run_index: int, step_index: int):
    batch_size = min(
        service.planning_bundle.budget_controller.batch_size,
        remaining_simulations(search_state.budget_state),
    )
    if batch_size <= 0:
        return search_state
    proposal_values = propose_parameter_batch(
        service.task,
        run_label=f"bayesopt_baseline|{run_index}",
        step_index=step_index,
        batch_size=batch_size,
    )
    candidate_records = []
    ranking_scores: dict[str, float] = {}
    for sample_index, parameter_values in enumerate(proposal_values):
        next_state = build_world_state(
            service.task,
            parameter_values=parameter_values,
            provenance_type="trajectory_replay",
            provenance_stage="predicted",
        )
        candidate = service._candidate_record(
            next_state,
            parent_candidate_id=None,
            proposal_source="restart_policy",
            action_chain=[],
            generation_depth=0,
        )
        metrics_prediction, feasibility_prediction, simulation_value, acquisition = build_surrogate_predictions(
            service.task,
            next_state.state_id,
            parameter_values,
            observations=observations,
            seed=f"{candidate.candidate_id}|{run_index}|{step_index}|{sample_index}",
        )
        candidate = append_evaluation_event(
            candidate.model_copy(
                update={
                    "predicted_metrics": metrics_prediction,
                    "predicted_feasibility": feasibility_prediction,
                    "predicted_uncertainty": feasibility_prediction.trust_assessment,
                    "simulation_value_estimate": simulation_value,
                    "lifecycle_state": "frontier",
                    "lifecycle_status": "frontier",
                }
            ),
            "predicted",
            [f"mode=bayesopt_baseline", f"observation_count={len(observations)}"],
        )
        candidate_records.append(candidate)
        ranking_scores[candidate.world_state_ref] = acquisition
    candidate_records = apply_priority_scores(candidate_records, ranking_scores=ranking_scores, policy=service.planning_bundle.selection_policy)
    from libs.planner.budget_controller import consume_proxy_evaluations

    budget_state = consume_proxy_evaluations(search_state.budget_state, len(candidate_records))
    pool_state = search_state.candidate_pool_state
    for candidate in candidate_records:
        pool_state = upsert_candidate(pool_state, candidate)
    trace = service._make_trace(
        search_state,
        outcome_tag="candidate_proposed",
        selected_candidate_id=candidate_records[0].candidate_id if candidate_records else None,
        executed_action_chain=[],
        world_model_queries=[],
        simulation_decision=SimulationDecision(
            decision="prioritize",
            candidate_ids=[candidate.candidate_id for candidate in candidate_records],
            reasons=["bayesopt_baseline"],
        ),
        decision_rationale=["ranked sampled candidates with a lightweight acquisition surrogate built from verified history"],
        reward_or_progress_signal=max((candidate.priority_score for candidate in candidate_records), default=0.0),
        trust_snapshot=candidate_records[0].predicted_uncertainty if candidate_records else None,
    )
    return service._refresh_search_state(
        search_state,
        pool_state=pool_state,
        frontier_ids=[candidate.candidate_id for candidate in candidate_records],
        budget_state=budget_state,
        provenance_source="candidate_proposal",
        traces=[trace],
    )


def _propose_cmaes_candidates(service: PlanningService, search_state: SearchState, observations, run_index: int, step_index: int):
    batch_size = min(
        service.planning_bundle.budget_controller.batch_size,
        remaining_simulations(search_state.budget_state),
    )
    if batch_size <= 0:
        return search_state
    proposal_values = propose_cmaes_parameter_batch(
        service.task,
        observations=observations,
        run_label=f"cmaes_baseline|{run_index}",
        step_index=step_index,
        population_size=batch_size,
    )
    candidate_records = []
    ranking_scores: dict[str, float] = {}
    for sample_index, parameter_values in enumerate(proposal_values):
        next_state = build_world_state(
            service.task,
            parameter_values=parameter_values,
            provenance_type="trajectory_replay",
            provenance_stage="predicted",
        )
        candidate = service._candidate_record(
            next_state,
            parent_candidate_id=None,
            proposal_source="restart_policy",
            action_chain=[],
            generation_depth=0,
        )
        metrics_prediction, feasibility_prediction, simulation_value, acquisition = build_cmaes_surrogate_predictions(
            service.task,
            next_state.state_id,
            parameter_values,
            observations=observations,
            seed=f"{candidate.candidate_id}|{run_index}|{step_index}|{sample_index}",
        )
        candidate = append_evaluation_event(
            candidate.model_copy(
                update={
                    "predicted_metrics": metrics_prediction,
                    "predicted_feasibility": feasibility_prediction,
                    "predicted_uncertainty": feasibility_prediction.trust_assessment,
                    "simulation_value_estimate": simulation_value,
                    "lifecycle_state": "frontier",
                    "lifecycle_status": "frontier",
                }
            ),
            "predicted",
            [f"mode=cmaes_baseline", f"observation_count={len(observations)}"],
        )
        candidate_records.append(candidate)
        ranking_scores[candidate.world_state_ref] = acquisition
    candidate_records = apply_priority_scores(candidate_records, ranking_scores=ranking_scores, policy=service.planning_bundle.selection_policy)
    from libs.planner.budget_controller import consume_proxy_evaluations

    budget_state = consume_proxy_evaluations(search_state.budget_state, len(candidate_records))
    pool_state = search_state.candidate_pool_state
    for candidate in candidate_records:
        pool_state = upsert_candidate(pool_state, candidate)
    trace = service._make_trace(
        search_state,
        outcome_tag="candidate_proposed",
        selected_candidate_id=candidate_records[0].candidate_id if candidate_records else None,
        executed_action_chain=[],
        world_model_queries=[],
        simulation_decision=SimulationDecision(
            decision="prioritize",
            candidate_ids=[candidate.candidate_id for candidate in candidate_records],
            reasons=["cmaes_baseline"],
        ),
        decision_rationale=["ranked sampled candidates with a lightweight CMA-ES-style evolution surrogate built from verified history"],
        reward_or_progress_signal=max((candidate.priority_score for candidate in candidate_records), default=0.0),
        trust_snapshot=candidate_records[0].predicted_uncertainty if candidate_records else None,
    )
    return service._refresh_search_state(
        search_state,
        pool_state=pool_state,
        frontier_ids=[candidate.candidate_id for candidate in candidate_records],
        budget_state=budget_state,
        provenance_source="candidate_proposal",
        traces=[trace],
    )


def _propose_rl_candidates(service: PlanningService, search_state: SearchState, observations, run_index: int, step_index: int):
    batch_size = min(
        service.planning_bundle.budget_controller.batch_size,
        remaining_simulations(search_state.budget_state),
    )
    if batch_size <= 0:
        return search_state
    proposal_values = propose_rl_parameter_batch(
        service.task,
        observations=observations,
        run_label=f"rl_baseline|{run_index}",
        step_index=step_index,
        population_size=batch_size,
    )
    candidate_records = []
    ranking_scores: dict[str, float] = {}
    for sample_index, parameter_values in enumerate(proposal_values):
        next_state = build_world_state(
            service.task,
            parameter_values=parameter_values,
            provenance_type="trajectory_replay",
            provenance_stage="predicted",
        )
        candidate = service._candidate_record(
            next_state,
            parent_candidate_id=None,
            proposal_source="restart_policy",
            action_chain=[],
            generation_depth=0,
        )
        metrics_prediction, feasibility_prediction, simulation_value, acquisition = build_rl_surrogate_predictions(
            service.task,
            next_state.state_id,
            parameter_values,
            observations=observations,
            seed=f"{candidate.candidate_id}|{run_index}|{step_index}|{sample_index}",
        )
        candidate = append_evaluation_event(
            candidate.model_copy(
                update={
                    "predicted_metrics": metrics_prediction,
                    "predicted_feasibility": feasibility_prediction,
                    "predicted_uncertainty": feasibility_prediction.trust_assessment,
                    "simulation_value_estimate": simulation_value,
                    "lifecycle_state": "frontier",
                    "lifecycle_status": "frontier",
                }
            ),
            "predicted",
            [f"mode=rl_baseline", f"observation_count={len(observations)}"],
        )
        candidate_records.append(candidate)
        ranking_scores[candidate.world_state_ref] = acquisition
    candidate_records = apply_priority_scores(candidate_records, ranking_scores=ranking_scores, policy=service.planning_bundle.selection_policy)
    from libs.planner.budget_controller import consume_proxy_evaluations

    budget_state = consume_proxy_evaluations(search_state.budget_state, len(candidate_records))
    pool_state = search_state.candidate_pool_state
    for candidate in candidate_records:
        pool_state = upsert_candidate(pool_state, candidate)
    trace = service._make_trace(
        search_state,
        outcome_tag="candidate_proposed",
        selected_candidate_id=candidate_records[0].candidate_id if candidate_records else None,
        executed_action_chain=[],
        world_model_queries=[],
        simulation_decision=SimulationDecision(
            decision="prioritize",
            candidate_ids=[candidate.candidate_id for candidate in candidate_records],
            reasons=["rl_baseline"],
        ),
        decision_rationale=["ranked sampled candidates with a lightweight policy-learning surrogate built from verified history"],
        reward_or_progress_signal=max((candidate.priority_score for candidate in candidate_records), default=0.0),
        trust_snapshot=candidate_records[0].predicted_uncertainty if candidate_records else None,
    )
    return service._refresh_search_state(
        search_state,
        pool_state=pool_state,
        frontier_ids=[candidate.candidate_id for candidate in candidate_records],
        budget_state=budget_state,
        provenance_source="candidate_proposal",
        traces=[trace],
    )


def _select_candidates_for_mode(service: PlanningService, search_state: SearchState, mode: str):
    if mode == "random_search_baseline":
        frontier_ids = set(search_state.frontier_state.frontier_candidate_ids)
        selected = [
            candidate
            for candidate in search_state.candidate_pool_state.candidates
            if candidate.candidate_id in frontier_ids and candidate.lifecycle_status in {"proposed", "frontier"}
        ]
        selected = selected[: remaining_simulations(search_state.budget_state)]
        queued = [
            append_decision_event(
                candidate.model_copy(update={"lifecycle_state": "queued_for_simulation", "lifecycle_status": "queued_for_simulation"}),
                "simulate",
                "random_search_baseline",
            )
            for candidate in selected
        ]
        pool_state = search_state.candidate_pool_state
        for candidate in queued:
            pool_state = upsert_candidate(pool_state, candidate)
        from libs.planner.budget_controller import consume_simulations

        budget_state = consume_simulations(search_state.budget_state, len(queued))
        trace = service._make_trace(
            search_state,
            outcome_tag="simulation_selected",
            selected_candidate_id=queued[0].candidate_id if queued else None,
            executed_action_chain=[],
            world_model_queries=[],
            simulation_decision=SimulationDecision(
                decision="simulate" if queued else "defer",
                candidate_ids=[candidate.candidate_id for candidate in queued],
                reasons=["random_search_baseline"],
                requested_fidelity="quick_truth" if queued else None,
            ),
            decision_rationale=["simulate each sampled random-search candidate without planner ranking"],
            reward_or_progress_signal=max((candidate.priority_score for candidate in queued), default=0.0),
            trust_snapshot=queued[0].predicted_uncertainty if queued else None,
        )
        updated_state = service._refresh_search_state(
            search_state,
            pool_state=pool_state,
            frontier_ids=[candidate.candidate_id for candidate in frontier_candidates(pool_state)],
            budget_state=budget_state,
            provenance_source="candidate_evaluation",
            traces=[trace],
        ).model_copy(update={"pending_simulation_refs": [*search_state.pending_simulation_refs, *[candidate.candidate_id for candidate in queued]]})
        return SimulationSelectionResponse(
            search_state=updated_state,
            selected_candidates=queued,
            requested_fidelity_map={candidate.candidate_id: "quick_truth" for candidate in queued},
            traces=[trace],
        )
    if mode == "rl_baseline":
        frontier_ids = set(search_state.frontier_state.frontier_candidate_ids)
        frontier = [
            candidate
            for candidate in search_state.candidate_pool_state.candidates
            if candidate.candidate_id in frontier_ids and candidate.lifecycle_status in {"proposed", "frontier"}
        ]
        ranked = sorted(frontier, key=lambda item: item.priority_score, reverse=True)
        selected = ranked[: min(service.planning_bundle.budget_controller.batch_size, remaining_simulations(search_state.budget_state))]
        queued = [
            append_decision_event(
                candidate.model_copy(update={"lifecycle_state": "queued_for_simulation", "lifecycle_status": "queued_for_simulation"}),
                "simulate",
                "rl_baseline",
            )
            for candidate in selected
        ]
        pool_state = search_state.candidate_pool_state
        for candidate in queued:
            pool_state = upsert_candidate(pool_state, candidate)
        from libs.planner.budget_controller import consume_simulations

        budget_state = consume_simulations(search_state.budget_state, len(queued))
        trace = service._make_trace(
            search_state,
            outcome_tag="simulation_selected",
            selected_candidate_id=queued[0].candidate_id if queued else None,
            executed_action_chain=[],
            world_model_queries=[],
            simulation_decision=SimulationDecision(
                decision="simulate" if queued else "defer",
                candidate_ids=[candidate.candidate_id for candidate in queued],
                reasons=["rl_baseline"],
                requested_fidelity="quick_truth" if queued else None,
            ),
            decision_rationale=["selected the top policy-ranked RL baseline candidates for real simulation"],
            reward_or_progress_signal=max((candidate.priority_score for candidate in queued), default=0.0),
            trust_snapshot=queued[0].predicted_uncertainty if queued else None,
        )
        updated_state = service._refresh_search_state(
            search_state,
            pool_state=pool_state,
            frontier_ids=[candidate.candidate_id for candidate in frontier_candidates(pool_state)],
            budget_state=budget_state,
            provenance_source="candidate_evaluation",
            traces=[trace],
        ).model_copy(update={"pending_simulation_refs": [*search_state.pending_simulation_refs, *[candidate.candidate_id for candidate in queued]]})
        return SimulationSelectionResponse(
            search_state=updated_state,
            selected_candidates=queued,
            requested_fidelity_map={candidate.candidate_id: "quick_truth" for candidate in queued},
            traces=[trace],
        )
    if mode == "cmaes_baseline":
        frontier_ids = set(search_state.frontier_state.frontier_candidate_ids)
        frontier = [
            candidate
            for candidate in search_state.candidate_pool_state.candidates
            if candidate.candidate_id in frontier_ids and candidate.lifecycle_status in {"proposed", "frontier"}
        ]
        ranked = sorted(frontier, key=lambda item: item.priority_score, reverse=True)
        selected = ranked[: min(service.planning_bundle.budget_controller.batch_size, remaining_simulations(search_state.budget_state))]
        queued = [
            append_decision_event(
                candidate.model_copy(update={"lifecycle_state": "queued_for_simulation", "lifecycle_status": "queued_for_simulation"}),
                "simulate",
                "cmaes_baseline",
            )
            for candidate in selected
        ]
        pool_state = search_state.candidate_pool_state
        for candidate in queued:
            pool_state = upsert_candidate(pool_state, candidate)
        from libs.planner.budget_controller import consume_simulations

        budget_state = consume_simulations(search_state.budget_state, len(queued))
        trace = service._make_trace(
            search_state,
            outcome_tag="simulation_selected",
            selected_candidate_id=queued[0].candidate_id if queued else None,
            executed_action_chain=[],
            world_model_queries=[],
            simulation_decision=SimulationDecision(
                decision="simulate" if queued else "defer",
                candidate_ids=[candidate.candidate_id for candidate in queued],
                reasons=["cmaes_baseline"],
                requested_fidelity="quick_truth" if queued else None,
            ),
            decision_rationale=["selected the top CMA-ES-ranked candidates for real simulation"],
            reward_or_progress_signal=max((candidate.priority_score for candidate in queued), default=0.0),
            trust_snapshot=queued[0].predicted_uncertainty if queued else None,
        )
        updated_state = service._refresh_search_state(
            search_state,
            pool_state=pool_state,
            frontier_ids=[candidate.candidate_id for candidate in frontier_candidates(pool_state)],
            budget_state=budget_state,
            provenance_source="candidate_evaluation",
            traces=[trace],
        ).model_copy(update={"pending_simulation_refs": [*search_state.pending_simulation_refs, *[candidate.candidate_id for candidate in queued]]})
        return SimulationSelectionResponse(
            search_state=updated_state,
            selected_candidates=queued,
            requested_fidelity_map={candidate.candidate_id: "quick_truth" for candidate in queued},
            traces=[trace],
        )
    if mode == "bayesopt_baseline":
        frontier_ids = set(search_state.frontier_state.frontier_candidate_ids)
        frontier = [
            candidate
            for candidate in search_state.candidate_pool_state.candidates
            if candidate.candidate_id in frontier_ids and candidate.lifecycle_status in {"proposed", "frontier"}
        ]
        ranked = sorted(frontier, key=lambda item: item.priority_score, reverse=True)
        selected = ranked[: min(service.planning_bundle.budget_controller.batch_size, remaining_simulations(search_state.budget_state))]
        queued = [
            append_decision_event(
                candidate.model_copy(update={"lifecycle_state": "queued_for_simulation", "lifecycle_status": "queued_for_simulation"}),
                "simulate",
                "bayesopt_baseline",
            )
            for candidate in selected
        ]
        pool_state = search_state.candidate_pool_state
        for candidate in queued:
            pool_state = upsert_candidate(pool_state, candidate)
        from libs.planner.budget_controller import consume_simulations

        budget_state = consume_simulations(search_state.budget_state, len(queued))
        trace = service._make_trace(
            search_state,
            outcome_tag="simulation_selected",
            selected_candidate_id=queued[0].candidate_id if queued else None,
            executed_action_chain=[],
            world_model_queries=[],
            simulation_decision=SimulationDecision(
                decision="simulate" if queued else "defer",
                candidate_ids=[candidate.candidate_id for candidate in queued],
                reasons=["bayesopt_baseline"],
                requested_fidelity="quick_truth" if queued else None,
            ),
            decision_rationale=["selected the top acquisition-ranked BayesOpt candidates for real simulation"],
            reward_or_progress_signal=max((candidate.priority_score for candidate in queued), default=0.0),
            trust_snapshot=queued[0].predicted_uncertainty if queued else None,
        )
        updated_state = service._refresh_search_state(
            search_state,
            pool_state=pool_state,
            frontier_ids=[candidate.candidate_id for candidate in frontier_candidates(pool_state)],
            budget_state=budget_state,
            provenance_source="candidate_evaluation",
            traces=[trace],
        ).model_copy(update={"pending_simulation_refs": [*search_state.pending_simulation_refs, *[candidate.candidate_id for candidate in queued]]})
        return SimulationSelectionResponse(
            search_state=updated_state,
            selected_candidates=queued,
            requested_fidelity_map={candidate.candidate_id: "quick_truth" for candidate in queued},
            traces=[trace],
        )
    if mode != "full_simulation_baseline":
        return service.select_for_simulation(search_state)
    selected = [
        candidate
        for candidate in search_state.candidate_pool_state.candidates
        if candidate.lifecycle_status in {
            "proposed",
            "frontier",
            "best_feasible",
            "best_infeasible",
            "queued_for_rollout",
        }
    ]
    selected = selected[: remaining_simulations(search_state.budget_state)]
    pool_state = search_state.candidate_pool_state
    queued = [
        append_decision_event(candidate, "simulate", "full simulation baseline")
        for candidate in selected
    ]
    from libs.planner.budget_controller import consume_simulations

    budget_state = consume_simulations(search_state.budget_state, len(queued))
    trace = service._make_trace(
        search_state,
        outcome_tag="simulation_selected",
        selected_candidate_id=queued[0].candidate_id if queued else None,
        executed_action_chain=queued[0].proposal_action_chain if queued else [],
        world_model_queries=[],
        simulation_decision=SimulationDecision(
            decision="simulate" if queued else "defer",
            candidate_ids=[candidate.candidate_id for candidate in queued],
            reasons=["full_simulation_baseline"],
        ),
        decision_rationale=["simulate every candidate without ranking"],
        reward_or_progress_signal=max((candidate.priority_score for candidate in queued), default=0.0),
        trust_snapshot=queued[0].predicted_uncertainty if queued else None,
    )
    updated_state = service._refresh_search_state(
        search_state,
        pool_state=pool_state,
        frontier_ids=[candidate.candidate_id for candidate in frontier_candidates(pool_state)],
        budget_state=budget_state,
        provenance_source="candidate_evaluation",
        traces=[trace],
    )
    return SimulationSelectionResponse(search_state=updated_state, selected_candidates=queued, traces=[trace])


def _gap_summary(executions) -> dict[str, float]:
    gap_accumulator: defaultdict[str, list[float]] = defaultdict(list)
    for execution in executions:
        for metric, value in execution.verification_result.calibration_payload.residual_metrics.items():
            gap_accumulator[metric].append(abs(float(value)))
    return {metric: round(sum(values) / len(values), 6) for metric, values in sorted(gap_accumulator.items()) if values}


def _mean_selected_value(candidates, extractor) -> float:
    values = [extractor(candidate) for candidate in candidates]
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return 0.0
    return round(sum(filtered) / len(filtered), 6)


def _best_metrics(executions) -> dict[str, float]:
    best: dict[str, float] = {}
    for execution in executions:
        for metric in execution.verification_result.measurement_report.measured_metrics:
            current = best.get(metric.metric)
            if metric.metric in {"power_w", "input_referred_noise_nv_sqrt_hz", "settling_time_s"}:
                if current is None or metric.value < current:
                    best[metric.metric] = float(metric.value)
            else:
                if current is None or metric.value > current:
                    best[metric.metric] = float(metric.value)
    return dict(sorted(best.items()))


def run_experiment(
    task,
    mode: ExperimentMode,
    budget: ExperimentBudget,
    steps: int,
    *,
    run_index: int = 0,
    fidelity_level: str = "focused_validation",
    backend_preference: str = "ngspice",
    force_full_steps: bool = False,
) -> ExperimentResult:
    """Run one structured experiment under a unified execution mode."""

    component_config = _mode_config(mode)
    world_model_bundle = compile_world_model_bundle(task).world_model_bundle
    assert world_model_bundle is not None
    planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
    assert planning_bundle is not None
    if not component_config.use_fidelity_escalation:
        planning_bundle = planning_bundle.model_copy(
            update={
                "escalation_policy": planning_bundle.escalation_policy.model_copy(
                    update={
                        "default_fidelity": "quick_truth",
                        "promoted_fidelity": "quick_truth",
                        "max_batch_size": 0,
                    }
                )
            }
        )
    planning_bundle = planning_bundle.model_copy(
        update={
            "budget_controller": planning_bundle.budget_controller.model_copy(
                update={
                    "max_real_simulations": budget.max_simulations,
                    "max_proxy_evaluations": max(
                        planning_bundle.budget_controller.max_proxy_evaluations,
                        budget.max_candidates_per_step * steps,
                    ),
                    "batch_size": min(planning_bundle.budget_controller.batch_size, budget.max_candidates_per_step),
                }
            ),
            "escalation_policy": planning_bundle.escalation_policy.model_copy(
                update={"max_batch_size": max(1, min(planning_bundle.escalation_policy.max_batch_size, budget.max_candidates_per_step))}
            ),
            "termination_policy": planning_bundle.termination_policy.model_copy(
                update={"max_iterations": max(steps * 4, planning_bundle.termination_policy.max_iterations)}
            ),
        }
    )
    service = PlanningService(planning_bundle, task, world_model_bundle)
    if (
        component_config.use_random_search_baseline
        or component_config.use_bayesopt_baseline
        or component_config.use_cmaes_baseline
        or component_config.use_rl_baseline
    ):
        search_state = _initialize_random_search_state(service, mode, run_index)
    elif _uses_baseline_predictions(component_config):
        search_state = _initialize_baseline_state(service, mode, run_index)
    else:
        search_state = service.initialize_search().search_state

    simulation_executions = []
    logs: list[ExperimentLogRecord] = []
    convergence_step: int | None = None
    total_selectable_candidates = 0
    total_selected_candidates = 0
    calibration_update_count = 0
    focused_truth_call_count = 0
    prediction_gap_by_step: list[dict[str, float]] = []
    bayesopt_observations = []
    cmaes_observations = []
    rl_observations = []

    for step_index in range(steps):
        if component_config.use_random_search_baseline:
            search_state = _propose_random_search_candidates(service, search_state, run_index, step_index)
        elif component_config.use_bayesopt_baseline:
            search_state = _propose_bayesopt_candidates(service, search_state, bayesopt_observations, run_index, step_index)
        elif component_config.use_cmaes_baseline:
            search_state = _propose_cmaes_candidates(service, search_state, cmaes_observations, run_index, step_index)
        elif component_config.use_rl_baseline:
            search_state = _propose_rl_candidates(service, search_state, rl_observations, run_index, step_index)
        elif _uses_baseline_predictions(component_config):
            search_state = _propose_baseline_candidates(service, search_state, mode, run_index)
        else:
            search_state = service.propose_candidates(search_state).search_state
            search_state = service.evaluate_candidates(search_state).search_state

        selectable_count = max(
            1,
            len(
                [
                    candidate
                    for candidate in search_state.candidate_pool_state.candidates
                    if candidate.lifecycle_status in {"frontier", "best_feasible", "best_infeasible"}
                ]
            ),
        )
        total_selectable_candidates += selectable_count
        selection = _select_candidates_for_mode(service, search_state, mode)
        search_state = selection.search_state
        total_selected_candidates += len(selection.selected_candidates)
        step_executions = []
        step_fidelity_usage: Counter[str] = Counter()
        step_calibration_updates = 0
        for candidate in selection.selected_candidates:
            requested_fidelity = _requested_fidelity(component_config, selection, candidate, fidelity_level)
            execution = SimulationService(task, planning_bundle, search_state).verify_candidate(
                candidate.candidate_id,
                fidelity_level=requested_fidelity,
                backend_preference=backend_preference,
                escalation_reason=f"experiment_{mode}",
            )
            if requested_fidelity == "focused_truth":
                focused_truth_call_count += 1
            step_fidelity_usage[requested_fidelity] += 1
            simulation_executions.append(execution)
            step_executions.append(execution)
            if component_config.use_bayesopt_baseline:
                bayesopt_observations.append(build_bayesopt_observation(service.task, execution))
            if component_config.use_cmaes_baseline:
                cmaes_observations.append(build_cmaes_observation(service.task, execution))
            if component_config.use_rl_baseline:
                rl_observations.append(build_rl_observation(service.task, execution))
            feedback = service.ingest_simulation_feedback(
                search_state,
                candidate.candidate_id,
                execution.verification_result.calibration_payload.truth_record,
                planner_feedback=execution.verification_result.planner_feedback,
                apply_calibration=component_config.use_calibration,
            )
            if component_config.use_calibration:
                step_calibration_updates += 1
                calibration_update_count += 1
            search_state = feedback.search_state
            service.world_model_bundle = feedback.updated_world_model_bundle
            service.world_model_service.bundle = feedback.updated_world_model_bundle

        search_state = service.advance_phase(search_state).search_state
        if convergence_step is None and search_state.best_known_feasible is not None:
            convergence_step = step_index

        step_gap_summary = _gap_summary(step_executions)
        prediction_gap_by_step.append(step_gap_summary)
        failure_counter = Counter(
            execution.verification_result.failure_attribution.primary_failure_class
            for execution in step_executions
            if execution.verification_result.failure_attribution.primary_failure_class != "none"
        )
        step_feasible_hit = any(
            execution.verification_result.feasibility_status in {"feasible_nominal", "feasible_certified"}
            for execution in step_executions
        )
        logs.append(
            ExperimentLogRecord(
                step_index=step_index,
                mode=mode,
                candidate_ids=[candidate.candidate_id for candidate in selection.selected_candidates],
                predicted_truth_gap=step_gap_summary,
                simulation_selection_ratio=round(len(selection.selected_candidates) / selectable_count, 6),
                feasible_hit=step_feasible_hit,
                failure_type_distribution=dict(sorted(failure_counter.items())),
                fidelity_usage=dict(sorted(step_fidelity_usage.items())),
                calibration_updates_applied=step_calibration_updates,
                world_model_enabled=component_config.use_world_model,
                calibration_enabled=component_config.use_calibration,
                fidelity_escalation_enabled=component_config.use_fidelity_escalation,
                selected_mean_uncertainty=_mean_selected_value(
                    selection.selected_candidates,
                    lambda candidate: candidate.predicted_uncertainty.uncertainty_score if candidate.predicted_uncertainty else None,
                ),
                selected_mean_confidence=_mean_selected_value(
                    selection.selected_candidates,
                    lambda candidate: candidate.predicted_uncertainty.confidence if candidate.predicted_uncertainty else None,
                ),
                selected_mean_simulation_value=_mean_selected_value(
                    selection.selected_candidates,
                    lambda candidate: candidate.simulation_value_estimate.estimated_value if candidate.simulation_value_estimate else None,
                ),
                selected_mean_predicted_feasibility=_mean_selected_value(
                    selection.selected_candidates,
                    lambda candidate: candidate.predicted_feasibility.overall_feasibility if candidate.predicted_feasibility else None,
                ),
            )
        )
        if not force_full_steps and mode != "full_simulation_baseline" and step_feasible_hit:
            break
        if (
            not force_full_steps
            and not _uses_baseline_predictions(component_config)
            and service.should_terminate(search_state).should_terminate
        ):
            break
        if (
            not force_full_steps
            and _uses_baseline_predictions(component_config)
            and remaining_simulations(search_state.budget_state) <= 0
        ):
            break

    feasible_truth_executions = [
        execution
        for execution in simulation_executions
        if execution.verification_result.feasibility_status in {"feasible_nominal", "feasible_certified"}
    ]
    best_feasible_found = bool(feasible_truth_executions)
    if convergence_step is None and best_feasible_found:
        for index, log in enumerate(logs):
            if log.feasible_hit:
                convergence_step = index
                break

    result = ExperimentResult(
        run_id=f"{mode}_{task.task_id}_{run_index}",
        mode=mode,
        task_id=task.task_id,
        component_config=component_config,
        simulation_call_count=len(simulation_executions),
        candidate_count=len(search_state.candidate_pool_state.candidates),
        best_feasible_found=best_feasible_found,
        best_metrics=_best_metrics(simulation_executions),
        convergence_step=convergence_step,
        predicted_truth_gap=_gap_summary(simulation_executions),
        simulation_selection_ratio=round(total_selected_candidates / max(1, total_selectable_candidates), 6),
        feasible_hit_rate=1.0 if best_feasible_found else 0.0,
        failure_type_distribution=dict(
            sorted(
                Counter(
                    execution.verification_result.failure_attribution.primary_failure_class
                    for execution in simulation_executions
                    if execution.verification_result.failure_attribution.primary_failure_class != "none"
                ).items()
            )
        ),
        efficiency_score=efficiency_score(best_feasible_found, len(simulation_executions)),
        prediction_gap_by_step=prediction_gap_by_step,
        calibration_update_count=calibration_update_count,
        focused_truth_call_count=focused_truth_call_count,
        structured_log=logs,
        verification_stats=[execution.verification_stats for execution in simulation_executions],
    )
    return result.model_copy(update={"stats_record": build_experiment_stats_record(result)})


def _method_mode_summary(mode: ExperimentMode, runs: list[ExperimentResult], steps: int) -> MethodModeSummary:
    metric_accumulator: defaultdict[str, list[float]] = defaultdict(list)
    gap_accumulator: defaultdict[str, list[float]] = defaultdict(list)
    escalation_count = 0
    for result in runs:
        for metric, value in result.best_metrics.items():
            metric_accumulator[metric].append(value)
        for metric, value in result.predicted_truth_gap.items():
            gap_accumulator[metric].append(value)
        escalation_count += result.focused_truth_call_count
    component_config = runs[0].component_config if runs else _mode_config(mode)
    total_sim_calls = sum(result.simulation_call_count for result in runs)
    return MethodModeSummary(
        mode=mode,
        component_config=component_config,
        run_count=len(runs),
        simulation_call_count=round(total_sim_calls / max(1, len(runs)), 6),
        feasible_hit_rate=feasible_hit_rate(runs),
        average_prediction_gap={metric: round(sum(values) / len(values), 6) for metric, values in sorted(gap_accumulator.items()) if values},
        average_best_metrics={metric: round(sum(values) / len(values), 6) for metric, values in sorted(metric_accumulator.items()) if values},
        average_convergence_step=round(
            sum((result.convergence_step if result.convergence_step is not None else steps) for result in runs) / max(1, len(runs)),
            6,
        ),
        average_calibration_update_count=round(sum(result.calibration_update_count for result in runs) / max(1, len(runs)), 6),
        focused_truth_ratio=round(
            sum(result.focused_truth_call_count for result in runs) / max(1, total_sim_calls),
            6,
        ) if total_sim_calls else 0.0,
        escalation_count=escalation_count,
        failure_type_distribution=aggregate_failure_type_distribution(runs),
    )


def _delta(best_metrics_a: dict[str, float], best_metrics_b: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(best_metrics_a) | set(best_metrics_b))
    return {
        key: round(best_metrics_b.get(key, 0.0) - best_metrics_a.get(key, 0.0), 6)
        for key in keys
    }


def _gap_delta(gap_a: dict[str, float], gap_b: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(gap_a) | set(gap_b))
    return {
        key: round(gap_b.get(key, 0.0) - gap_a.get(key, 0.0), 6)
        for key in keys
    }


def _build_method_comparison(task_id: str, summaries: list[MethodModeSummary]) -> MethodComparisonResult | None:
    summary_map = {summary.mode: summary for summary in summaries}
    if "full_system" not in summary_map:
        return None
    full = summary_map["full_system"]
    deltas: list[MethodDeltaSummary] = []
    comparison_modes = [mode for mode in ("no_world_model", "no_calibration", "no_fidelity_escalation") if mode in summary_map]
    for mode in comparison_modes:
        compared = summary_map[mode]
        deltas.append(
            MethodDeltaSummary(
                baseline_mode=mode,
                compared_mode="full_system",
                simulation_call_delta=round(full.simulation_call_count - compared.simulation_call_count, 6),
                feasible_hit_rate_delta=round(full.feasible_hit_rate - compared.feasible_hit_rate, 6),
                prediction_gap_delta=_gap_delta(compared.average_prediction_gap, full.average_prediction_gap),
                best_metric_delta=_delta(compared.average_best_metrics, full.average_best_metrics),
                focused_truth_ratio_delta=round(full.focused_truth_ratio - compared.focused_truth_ratio, 6),
                calibration_update_delta=round(full.average_calibration_update_count - compared.average_calibration_update_count, 6),
            )
        )
    world_model_effective = "no_world_model" not in summary_map or (
        full.simulation_call_count <= summary_map["no_world_model"].simulation_call_count
        and full.feasible_hit_rate >= summary_map["no_world_model"].feasible_hit_rate
    )
    calibration_effective = "no_calibration" not in summary_map or (
        full.average_calibration_update_count > 0.0
        and (
            full.average_prediction_gap.get("gbw_hz", 0.0)
            <= summary_map["no_calibration"].average_prediction_gap.get("gbw_hz", float("inf"))
        )
    )
    fidelity_effective = "no_fidelity_escalation" not in summary_map or (
        full.focused_truth_ratio > 0.0
        and full.simulation_call_count <= summary_map["no_fidelity_escalation"].simulation_call_count
    )
    notes: list[str] = []
    if "no_world_model" in summary_map:
        notes.append(
            f"world_model_delta_sim_calls={round(summary_map['no_world_model'].simulation_call_count - full.simulation_call_count, 6)}"
        )
    if "no_calibration" in summary_map:
        notes.append(
            f"calibration_delta_gbw_gap={round(summary_map['no_calibration'].average_prediction_gap.get('gbw_hz', 0.0) - full.average_prediction_gap.get('gbw_hz', 0.0), 6)}"
        )
    if "no_fidelity_escalation" in summary_map:
        notes.append(
            f"fidelity_delta_focused_ratio={round(full.focused_truth_ratio - summary_map['no_fidelity_escalation'].focused_truth_ratio, 6)}"
        )
    return MethodComparisonResult(
        task_id=task_id,
        modes=[summary.mode for summary in summaries],
        mode_summaries=summaries,
        deltas=deltas,
        conclusions=MethodConclusionSummary(
            world_model_effective=world_model_effective,
            calibration_effective=calibration_effective,
            fidelity_effective=fidelity_effective,
            conclusion_notes=notes,
        ),
    )


def run_experiment_suite(
    task,
    modes: list[ExperimentMode],
    budget: ExperimentBudget,
    steps: int,
    *,
    repeat_runs: int = 5,
    fidelity_level: str = "focused_validation",
    backend_preference: str = "ngspice",
    force_full_steps: bool = False,
) -> ExperimentSuiteResult:
    """Run repeated experiments for several execution modes."""

    runs = [
        run_experiment(
            task,
            mode,
            budget,
            steps,
            run_index=run_index,
            fidelity_level=fidelity_level,
            backend_preference=backend_preference,
            force_full_steps=force_full_steps,
        )
        for mode in modes
        for run_index in range(repeat_runs)
    ]
    summaries = []
    method_summaries: list[MethodModeSummary] = []
    for mode in modes:
        mode_runs = [result for result in runs if result.mode == mode]
        metric_accumulator: defaultdict[str, list[float]] = defaultdict(list)
        gap_accumulator: defaultdict[str, list[float]] = defaultdict(list)
        for result in mode_runs:
            for metric, value in result.best_metrics.items():
                metric_accumulator[metric].append(value)
            for metric, value in result.predicted_truth_gap.items():
                gap_accumulator[metric].append(value)
        summaries.append(
            ExperimentAggregateSummary(
                mode=mode,
                run_count=len(mode_runs),
                average_simulation_call_count=round(sum(result.simulation_call_count for result in mode_runs) / max(1, len(mode_runs)), 6),
                feasible_hit_rate=feasible_hit_rate(mode_runs),
                average_efficiency_score=round(sum(result.efficiency_score for result in mode_runs) / max(1, len(mode_runs)), 6),
                average_convergence_step=round(
                    sum((result.convergence_step if result.convergence_step is not None else steps) for result in mode_runs) / max(1, len(mode_runs)),
                    6,
                ),
                average_selection_ratio=round(sum(result.simulation_selection_ratio for result in mode_runs) / max(1, len(mode_runs)), 6),
                average_best_metrics={
                    metric: round(sum(values) / len(values), 6)
                    for metric, values in sorted(metric_accumulator.items())
                    if values
                },
                failure_type_distribution=aggregate_failure_type_distribution(mode_runs),
                average_prediction_gap={
                    metric: round(sum(values) / len(values), 6)
                    for metric, values in sorted(gap_accumulator.items())
                    if values
                },
                average_calibration_update_count=round(sum(result.calibration_update_count for result in mode_runs) / max(1, len(mode_runs)), 6),
                average_focused_truth_call_count=round(sum(result.focused_truth_call_count for result in mode_runs) / max(1, len(mode_runs)), 6),
            )
        )
        if mode in {"full_system", "no_world_model", "no_calibration", "no_fidelity_escalation"}:
            method_summaries.append(_method_mode_summary(mode, mode_runs, steps))
    suite = ExperimentSuiteResult(task_id=task.task_id, modes=modes, runs=runs, summaries=summaries)
    suite = suite.model_copy(update={"aggregated_stats": aggregate_stats(suite)})
    comparison = _build_method_comparison(task.task_id, method_summaries)
    return suite.model_copy(update={"comparison": comparison})
