"""Formal planning service for the fourth layer."""

from __future__ import annotations

from datetime import datetime, timezone

from libs.planner.budget_controller import (
    consume_calibrations,
    consume_proxy_evaluations,
    consume_rollouts,
    consume_simulations,
    initialize_budget_state,
    remaining_rollouts,
    remaining_simulations,
)
from libs.planner.candidate_manager import (
    append_decision_event,
    append_evaluation_event,
    find_candidate,
    frontier_candidates,
    summarize_candidate,
    upsert_candidate,
)
from libs.planner.phase_controller import advance_phase_state, increment_phase_iteration, initialize_phase_state
from libs.planner.rollout_planner import build_candidate_actions, build_rollout_action_chain
from libs.planner.selection_engine import apply_priority_scores, choose_best_known
from libs.planner.validation import validate_search_state
from libs.schema.design_task import DesignTask
from libs.schema.planning import (
    ActionPlanResponse,
    CandidateBatchResponse,
    CandidatePoolState,
    CandidateRecord,
    FrontierState,
    OptimizationTrace,
    PlanningBestResult,
    PlanningBundle,
    RiskContext,
    SearchInitializationResponse,
    SearchProvenance,
    SearchState,
    SimulationDecision,
    SimulationFeedbackResponse,
    SimulationSelectionResponse,
    StrategyContext,
    TerminationDecision,
    WorldModelQueryRecord,
)
from libs.schema.world_model import TruthCalibrationRecord, TrustAssessment, WorldModelBundle, WorldState
from libs.utils.hashing import stable_hash
from libs.world_model.service import WorldModelService
from libs.world_model.state_builder import build_world_state_from_design_task


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot(search_state: SearchState) -> str:
    return (
        f"phase={search_state.phase_state.current_phase};"
        f"candidates={len(search_state.candidate_pool_state.candidates)};"
        f"frontier={len(search_state.frontier_state.frontier_candidate_ids)};"
        f"proxy={search_state.budget_state.proxy_evaluations_used}/{search_state.budget_state.proxy_evaluation_budget};"
        f"rollouts={search_state.budget_state.rollouts_used}/{search_state.budget_state.rollout_budget};"
        f"sims={search_state.budget_state.simulations_used}/{search_state.budget_state.simulation_budget}"
    )


class PlanningService:
    """Budget-aware, uncertainty-aware planning service."""

    def __init__(self, planning_bundle: PlanningBundle, task: DesignTask, world_model_bundle: WorldModelBundle) -> None:
        self.planning_bundle = planning_bundle
        self.task = task
        self.world_model_bundle = world_model_bundle
        self.world_model_service = WorldModelService(world_model_bundle, task)

    def _make_trace(
        self,
        search_state: SearchState,
        *,
        outcome_tag: str,
        selected_candidate_id: str | None,
        executed_action_chain,
        world_model_queries: list[WorldModelQueryRecord],
        simulation_decision: SimulationDecision,
        decision_rationale: list[str],
        reward_or_progress_signal: float,
        simulation_result_ref: str | None = None,
        trust_snapshot: TrustAssessment | None = None,
    ) -> OptimizationTrace:
        step_index = len(search_state.trace_log)
        if trust_snapshot is None:
            trust_snapshot = TrustAssessment(
                trust_level="low",
                service_tier="screening_only",
                confidence=0.0,
                uncertainty_score=1.0,
                ood_score=1.0,
                must_escalate=False,
                hard_block=False,
                reasons=["uninitialized_trust"],
            )
        return OptimizationTrace(
            trace_id=f"trace_{stable_hash(f'{search_state.search_id}|{step_index}|{outcome_tag}')[:12]}",
            task_id=self.task.task_id,
            episode_id=search_state.episode_id,
            step_index=step_index,
            search_state_snapshot=_snapshot(search_state),
            selected_candidate_id=selected_candidate_id,
            executed_action_chain=list(executed_action_chain),
            world_model_queries=world_model_queries,
            simulation_decision=simulation_decision,
            simulation_result_ref=simulation_result_ref,
            reward_or_progress_signal=round(float(reward_or_progress_signal), 6),
            decision_rationale=list(decision_rationale),
            trust_snapshot=trust_snapshot,
            budget_snapshot=search_state.budget_state,
            outcome_tag=outcome_tag,
        )

    def _risk_context(self, candidates: list[CandidateRecord], *, calibration_required: bool = False, extra_failures: list[str] | None = None) -> RiskContext:
        trust_records = [candidate.predicted_uncertainty for candidate in candidates if candidate.predicted_uncertainty is not None]
        if not trust_records:
            return RiskContext(
                average_uncertainty=1.0,
                max_ood_score=1.0,
                trust_alert_count=1,
                calibration_required=calibration_required,
                active_failure_modes=list(extra_failures or []),
            )
        average_uncertainty = sum(item.uncertainty_score for item in trust_records) / max(1, len(trust_records))
        max_ood = max(item.ood_score for item in trust_records)
        alerts = sum(1 for item in trust_records if item.must_escalate or item.hard_block or item.service_tier == "hard_block")
        failure_modes = []
        for candidate in candidates:
            if candidate.predicted_feasibility is not None:
                failure_modes.extend(candidate.predicted_feasibility.most_likely_failure_reasons)
        failure_modes.extend(extra_failures or [])
        return RiskContext(
            average_uncertainty=average_uncertainty,
            max_ood_score=max_ood,
            trust_alert_count=alerts,
            calibration_required=calibration_required or alerts > 0,
            active_failure_modes=failure_modes,
        )

    def _current_world_state(self, candidates: list[CandidateRecord], fallback: WorldState) -> WorldState:
        best = choose_best_known(candidates, feasible=True) or choose_best_known(candidates, feasible=False)
        return best.world_state_snapshot if best is not None else fallback

    def _refresh_search_state(
        self,
        search_state: SearchState,
        *,
        pool_state: CandidatePoolState | None = None,
        frontier_ids: list[str] | None = None,
        budget_state=None,
        phase_state=None,
        provenance_source: str,
        traces: list[OptimizationTrace] | None = None,
        calibration_required: bool = False,
        extra_failures: list[str] | None = None,
    ) -> SearchState:
        pool_state = pool_state or search_state.candidate_pool_state
        candidates = pool_state.candidates
        best_feasible = choose_best_known(candidates, feasible=True)
        best_infeasible = choose_best_known(candidates, feasible=False)
        current_world_state = self._current_world_state(candidates, search_state.current_world_state)
        frontier_ids = list(frontier_ids if frontier_ids is not None else search_state.frontier_state.frontier_candidate_ids)
        trace_log = [*search_state.trace_log, *(traces or [])]
        return search_state.model_copy(
            update={
                "current_world_state": current_world_state,
                "candidate_pool_state": pool_state,
                "frontier_state": search_state.frontier_state.model_copy(
                    update={
                        "frontier_candidate_ids": frontier_ids,
                        "expansion_round": search_state.frontier_state.expansion_round + (1 if provenance_source == "candidate_proposal" else 0),
                    }
                ),
                "budget_state": budget_state or search_state.budget_state,
                "phase_state": phase_state or search_state.phase_state,
                "best_known_feasible": summarize_candidate(best_feasible) if best_feasible else None,
                "best_known_infeasible": summarize_candidate(best_infeasible) if best_infeasible else None,
                "strategy_context": search_state.strategy_context.model_copy(
                    update={"last_selected_candidate_id": frontier_ids[0] if frontier_ids else search_state.strategy_context.last_selected_candidate_id}
                ),
                "risk_context": self._risk_context(candidates, calibration_required=calibration_required, extra_failures=extra_failures),
                "provenance": SearchProvenance(source=provenance_source, created_at=_timestamp(), artifact_refs=[]),
                "trace_log": trace_log,
            }
        )

    def _candidate_record(
        self,
        world_state: WorldState,
        *,
        parent_candidate_id: str | None,
        proposal_source: str,
        action_chain: list,
        generation_depth: int,
    ) -> CandidateRecord:
        root = parent_candidate_id or "root"
        return CandidateRecord(
            candidate_id=f"cand_{stable_hash(f'{world_state.state_id}|{root}|{proposal_source}')[:12]}",
            task_id=self.task.task_id,
            world_state_ref=world_state.state_id,
            world_state_snapshot=world_state,
            parent_candidate_id=parent_candidate_id,
            generation_depth=generation_depth,
            proposal_source=proposal_source,
            proposal_action_chain=list(action_chain),
            priority_score=0.0,
            dominance_status="unknown",
            lifecycle_status="proposed",
            evaluation_history=[],
            decision_history=[],
            artifact_refs=list(world_state.provenance.artifact_refs),
        )

    def _evaluate_records(self, candidates: list[CandidateRecord], budget_state):
        if not candidates:
            return [], budget_state
        states = [candidate.world_state_snapshot for candidate in candidates]
        ranking = self.world_model_service.rank_candidates(states)
        ranking_scores = {item.state_id: item.score for item in ranking.ranked_candidates}
        evaluated: list[CandidateRecord] = []
        for candidate in candidates:
            metrics = self.world_model_service.predict_metrics(candidate.world_state_snapshot)
            feasibility = self.world_model_service.predict_feasibility(candidate.world_state_snapshot)
            simulation_value = self.world_model_service.estimate_simulation_value(candidate.world_state_snapshot)
            updated = candidate.model_copy(
                update={
                    "predicted_metrics": metrics,
                    "predicted_feasibility": feasibility,
                    "predicted_uncertainty": feasibility.trust_assessment,
                    "simulation_value_estimate": simulation_value,
                }
            )
            updated = append_evaluation_event(updated, "predicted", [f"service_tier={feasibility.trust_assessment.service_tier}"])
            lifecycle = "frontier"
            if feasibility.trust_assessment.service_tier == "hard_block":
                lifecycle = "screened_out"
                updated = append_decision_event(updated, "drop", "hard-block trust assessment")
            elif feasibility.overall_feasibility < 0.15 and simulation_value.decision == "defer":
                lifecycle = "screened_out"
                updated = append_decision_event(updated, "drop", "low feasibility with low simulation value")
            else:
                updated = append_decision_event(updated, "keep", "candidate remains eligible for search")
            evaluated.append(updated.model_copy(update={"lifecycle_status": lifecycle}))
        scored = apply_priority_scores(evaluated, ranking_scores=ranking_scores, policy=self.planning_bundle.selection_policy)
        budget_state = consume_proxy_evaluations(budget_state, len(candidates))
        return scored, budget_state

    def initialize_search(self) -> SearchInitializationResponse:
        """Initialize a formal search state from DesignTask and WorldModelBundle."""

        seed_states: list[WorldState] = []
        if self.task.initial_state.seed_candidates:
            for seed in self.task.initial_state.seed_candidates:
                seed_values = {**self.task.initial_state.template_defaults, **seed.values}
                seed_states.append(build_world_state_from_design_task(self.task, parameter_values=seed_values, provenance_stage="initial"))
        if not seed_states:
            seed_states.append(build_world_state_from_design_task(self.task, parameter_values=self.task.initial_state.template_defaults, provenance_stage="initial"))

        budget_state = initialize_budget_state(self.planning_bundle.budget_controller)
        phase_state = initialize_phase_state(self.planning_bundle.phase_controller)
        records = [
            self._candidate_record(
                state,
                parent_candidate_id=None,
                proposal_source="initial_seed",
                action_chain=[],
                generation_depth=0,
            )
            for state in seed_states
        ]
        records, budget_state = self._evaluate_records(records, budget_state)
        pool_state = CandidatePoolState(candidates=[], active_candidate_ids=[], archived_candidate_ids=[], discarded_candidate_ids=[])
        for record in records:
            pool_state = upsert_candidate(pool_state, record)

        current = records[0].world_state_snapshot
        search_state = SearchState(
            search_id=f"search_{stable_hash(self.planning_bundle.planning_id)[:12]}",
            task_id=self.task.task_id,
            episode_id=f"episode_{stable_hash(self.task.task_id + self.planning_bundle.planning_id)[:12]}",
            current_world_state=current,
            candidate_pool_state=pool_state,
            frontier_state=FrontierState(
                frontier_candidate_ids=[record.candidate_id for record in records if record.lifecycle_status == "frontier"],
                expansion_round=0,
                max_frontier_size=max(4, self.planning_bundle.budget_controller.batch_size * 3),
            ),
            evaluated_state_refs=[record.world_state_ref for record in records],
            pending_simulation_refs=[],
            budget_state=budget_state,
            phase_state=phase_state,
            best_known_feasible=None,
            best_known_infeasible=None,
            strategy_context=StrategyContext(
                active_policy=self.planning_bundle.search_policy.feasibility_policy,
                exploration_enabled=True,
                rollout_enabled=True,
                last_selected_candidate_id=records[0].candidate_id,
                notes=["search_initialized"],
            ),
            risk_context=self._risk_context(records),
            provenance=SearchProvenance(source="initialized", created_at=_timestamp(), artifact_refs=[]),
            trace_log=[],
        )
        init_trace = self._make_trace(
            search_state,
            outcome_tag="candidate_evaluated",
            selected_candidate_id=records[0].candidate_id if records else None,
            executed_action_chain=[],
            world_model_queries=[WorldModelQueryRecord(method="rank_candidates", state_id=current.state_id, timestamp=_timestamp())],
            simulation_decision=SimulationDecision(decision="keep", candidate_ids=[record.candidate_id for record in records], reasons=["seed candidates initialized"]),
            decision_rationale=["initialized from DesignTask.initial_state and evaluated with world model"],
            reward_or_progress_signal=max((record.priority_score for record in records), default=0.0),
            trust_snapshot=records[0].predicted_uncertainty if records and records[0].predicted_uncertainty else None,
        )
        search_state = search_state.model_copy(update={"trace_log": [init_trace]})
        search_state = self._refresh_search_state(search_state, provenance_source="initialized", traces=[])
        return SearchInitializationResponse(planning_bundle=self.planning_bundle, search_state=search_state)

    def propose_candidates(self, search_state: SearchState) -> CandidateBatchResponse:
        """Propose new candidates from the current frontier."""

        frontier = frontier_candidates(search_state.candidate_pool_state)
        frontier = sorted(frontier, key=lambda item: item.priority_score, reverse=True)[: self.planning_bundle.budget_controller.batch_size]
        new_records: list[CandidateRecord] = []
        queries: list[WorldModelQueryRecord] = []
        budget_state = search_state.budget_state

        for anchor in frontier:
            if remaining_rollouts(budget_state) <= 0:
                break
            actions = build_candidate_actions(
                self.task,
                anchor.world_state_snapshot,
                search_state.phase_state.current_phase,
                self.planning_bundle.rollout_config.max_branching_factor,
            )
            for action in actions:
                if remaining_rollouts(budget_state) <= 0:
                    break
                transition = self.world_model_service.predict_transition(anchor.world_state_snapshot, action)
                queries.append(WorldModelQueryRecord(method="predict_transition", state_id=anchor.world_state_ref, action_id=action.action_id, timestamp=_timestamp()))
                budget_state = consume_rollouts(budget_state, 1)
                candidate = self._candidate_record(
                    transition.next_state,
                    parent_candidate_id=anchor.candidate_id,
                    proposal_source="planner_mutation",
                    action_chain=[*anchor.proposal_action_chain, action],
                    generation_depth=anchor.generation_depth + 1,
                )
                new_records.append(candidate)

        new_records, budget_state = self._evaluate_records(new_records, budget_state)
        pool_state = search_state.candidate_pool_state
        for candidate in new_records:
            pool_state = upsert_candidate(pool_state, candidate)
        frontier_ids = [candidate.candidate_id for candidate in frontier_candidates(pool_state)]
        trace = self._make_trace(
            search_state,
            outcome_tag="candidate_proposed",
            selected_candidate_id=frontier[0].candidate_id if frontier else None,
            executed_action_chain=new_records[0].proposal_action_chain if new_records else [],
            world_model_queries=queries,
            simulation_decision=SimulationDecision(decision="keep", candidate_ids=[candidate.candidate_id for candidate in new_records], reasons=["candidate proposal via world-model transitions"]),
            decision_rationale=["expanded frontier through deterministic planner mutations"],
            reward_or_progress_signal=max((candidate.priority_score for candidate in new_records), default=0.0),
            trust_snapshot=new_records[0].predicted_uncertainty if new_records and new_records[0].predicted_uncertainty else None,
        )
        updated_state = self._refresh_search_state(
            search_state,
            pool_state=pool_state,
            frontier_ids=frontier_ids,
            budget_state=budget_state,
            phase_state=increment_phase_iteration(search_state.phase_state, improved=bool(new_records)),
            provenance_source="candidate_proposal",
            traces=[trace],
        )
        return CandidateBatchResponse(search_state=updated_state, candidates=new_records, traces=[trace])

    def evaluate_candidates(self, search_state: SearchState) -> CandidateBatchResponse:
        """Re-evaluate frontier candidates through the world-model service."""

        frontier_ids = list(search_state.frontier_state.frontier_candidate_ids)
        frontier_records = [
            candidate
            for candidate in search_state.candidate_pool_state.candidates
            if candidate.candidate_id in frontier_ids
        ]
        budget_state = search_state.budget_state
        reevaluated, budget_state = self._evaluate_records(frontier_records, budget_state)
        pool_state = search_state.candidate_pool_state
        for candidate in reevaluated:
            pool_state = upsert_candidate(pool_state, candidate)
        trace = self._make_trace(
            search_state,
            outcome_tag="candidate_evaluated",
            selected_candidate_id=reevaluated[0].candidate_id if reevaluated else None,
            executed_action_chain=[],
            world_model_queries=[WorldModelQueryRecord(method="rank_candidates", state_id=candidate.world_state_ref, timestamp=_timestamp()) for candidate in reevaluated],
            simulation_decision=SimulationDecision(decision="keep", candidate_ids=[candidate.candidate_id for candidate in reevaluated], reasons=["frontier reevaluated"]),
            decision_rationale=["updated candidate scores from world-model ranking and feasibility outputs"],
            reward_or_progress_signal=max((candidate.priority_score for candidate in reevaluated), default=0.0),
            trust_snapshot=reevaluated[0].predicted_uncertainty if reevaluated and reevaluated[0].predicted_uncertainty else None,
        )
        updated_state = self._refresh_search_state(
            search_state,
            pool_state=pool_state,
            frontier_ids=[candidate.candidate_id for candidate in frontier_candidates(pool_state)],
            budget_state=budget_state,
            phase_state=increment_phase_iteration(search_state.phase_state, improved=bool(reevaluated)),
            provenance_source="candidate_evaluation",
            traces=[trace],
        )
        return CandidateBatchResponse(search_state=updated_state, candidates=reevaluated, traces=[trace])

    def plan_next_actions(self, search_state: SearchState, horizon: int | None = None) -> ActionPlanResponse:
        """Plan a deterministic action chain using world-model rollout."""

        anchor = choose_best_known(search_state.candidate_pool_state.candidates, feasible=True) or choose_best_known(search_state.candidate_pool_state.candidates, feasible=False)
        if anchor is None:
            return ActionPlanResponse(search_state=search_state, anchor_candidate_id=None, action_chain=[], rollout_response=None, traces=[])

        config = self.planning_bundle.rollout_config if horizon is None else self.planning_bundle.rollout_config.model_copy(update={"horizon": horizon})
        action_chain = build_rollout_action_chain(self.task, anchor.world_state_snapshot, search_state.phase_state.current_phase, config)
        updated_state = search_state
        rollout_response = None
        trace_list: list[OptimizationTrace] = []
        if action_chain and remaining_rollouts(search_state.budget_state) >= len(action_chain):
            rollout_response = self.world_model_service.rollout(anchor.world_state_snapshot, action_chain, horizon=len(action_chain))
            updated_state = search_state.model_copy(update={"budget_state": consume_rollouts(search_state.budget_state, len(action_chain))})
            trace = self._make_trace(
                updated_state,
                outcome_tag="action_planned",
                selected_candidate_id=anchor.candidate_id,
                executed_action_chain=action_chain,
                world_model_queries=[WorldModelQueryRecord(method="rollout", state_id=anchor.world_state_ref, timestamp=_timestamp())],
                simulation_decision=SimulationDecision(decision="defer", candidate_ids=[anchor.candidate_id], reasons=["lookahead planning only"]),
                decision_rationale=["planned next actions via world-model rollout"],
                reward_or_progress_signal=rollout_response.steps[-1].simulation_value.estimated_value if rollout_response.steps else 0.0,
                trust_snapshot=rollout_response.trust_assessment,
            )
            updated_state = updated_state.model_copy(update={"trace_log": [*updated_state.trace_log, trace]})
            trace_list = [trace]
        return ActionPlanResponse(
            search_state=updated_state,
            anchor_candidate_id=anchor.candidate_id,
            action_chain=action_chain,
            rollout_response=rollout_response,
            traces=trace_list,
        )

    def select_for_simulation(self, search_state: SearchState) -> SimulationSelectionResponse:
        """Select high-value candidates for real simulation."""

        selectable = []
        for candidate in search_state.candidate_pool_state.candidates:
            if candidate.lifecycle_status not in {"frontier", "best_feasible", "best_infeasible"}:
                continue
            if candidate.simulation_value_estimate is None or candidate.predicted_uncertainty is None:
                continue
            if candidate.predicted_uncertainty.service_tier not in self.planning_bundle.escalation_policy.allowed_service_tiers:
                continue
            if candidate.simulation_value_estimate.estimated_value < self.planning_bundle.escalation_policy.min_simulation_value and not candidate.predicted_uncertainty.must_escalate:
                continue
            selectable.append(candidate)
        selectable = sorted(
            selectable,
            key=lambda item: (
                item.simulation_value_estimate.estimated_value if item.simulation_value_estimate else 0.0,
                item.priority_score,
            ),
            reverse=True,
        )
        selected = selectable[: min(self.planning_bundle.escalation_policy.max_batch_size, remaining_simulations(search_state.budget_state))]
        pool_state = search_state.candidate_pool_state
        for candidate in selected:
            updated = append_decision_event(candidate.model_copy(update={"lifecycle_status": "queued_for_simulation"}), "simulate", "selected by escalation policy")
            pool_state = upsert_candidate(pool_state, updated)
        budget_state = consume_simulations(search_state.budget_state, len(selected))
        trace = self._make_trace(
            search_state,
            outcome_tag="simulation_selected",
            selected_candidate_id=selected[0].candidate_id if selected else None,
            executed_action_chain=selected[0].proposal_action_chain if selected else [],
            world_model_queries=[],
            simulation_decision=SimulationDecision(
                decision="simulate" if selected else "defer",
                candidate_ids=[candidate.candidate_id for candidate in selected],
                reasons=["budget-aware escalation"],
            ),
            decision_rationale=["selected candidates based on simulation value and trust tier"],
            reward_or_progress_signal=max((candidate.priority_score for candidate in selected), default=0.0),
            trust_snapshot=selected[0].predicted_uncertainty if selected and selected[0].predicted_uncertainty else None,
        )
        updated_state = self._refresh_search_state(
            search_state,
            pool_state=pool_state,
            frontier_ids=[candidate.candidate_id for candidate in frontier_candidates(pool_state)],
            budget_state=budget_state,
            provenance_source="candidate_evaluation",
            traces=[trace],
        ).model_copy(update={"pending_simulation_refs": [*search_state.pending_simulation_refs, *[candidate.candidate_id for candidate in selected]]})
        return SimulationSelectionResponse(search_state=updated_state, selected_candidates=selected, traces=[trace])

    def ingest_simulation_feedback(
        self,
        search_state: SearchState,
        candidate_id: str,
        truth: TruthCalibrationRecord,
    ) -> SimulationFeedbackResponse:
        """Ingest real simulator feedback for one candidate."""

        candidate = find_candidate(search_state.candidate_pool_state, candidate_id)
        if candidate is None:
            raise ValueError(f"unknown candidate_id: {candidate_id}")
        calibration = self.world_model_service.calibrate_with_truth(candidate.world_state_snapshot, truth)
        self.world_model_bundle = calibration.updated_bundle
        feasible = all(item.is_satisfied for item in truth.constraints) if truth.constraints else (
            candidate.predicted_feasibility.overall_feasibility >= 0.8 if candidate.predicted_feasibility else False
        )
        lifecycle = "verified" if feasible else "rejected"
        updated_candidate = append_evaluation_event(
            candidate.model_copy(
                update={
                    "lifecycle_status": lifecycle,
                    "artifact_refs": [*candidate.artifact_refs, *truth.artifact_refs],
                }
            ),
            "simulated",
            [f"simulator={truth.simulator_signature}"],
        )
        updated_candidate = append_decision_event(
            updated_candidate,
            "keep" if feasible else "drop",
            "verified feasible by simulator" if feasible else "simulator feedback rejected candidate",
        )
        pool_state = upsert_candidate(search_state.candidate_pool_state, updated_candidate)
        budget_state = consume_calibrations(search_state.budget_state, 1)
        mismatch = bool(
            not feasible
            and (
                candidate.predicted_feasibility is None
                or candidate.predicted_feasibility.overall_feasibility >= 0.5
                or (candidate.predicted_uncertainty is not None and candidate.predicted_uncertainty.confidence >= 0.45)
            )
        )
        trace = self._make_trace(
            search_state,
            outcome_tag="simulation_feedback_ingested",
            selected_candidate_id=candidate_id,
            executed_action_chain=updated_candidate.proposal_action_chain,
            world_model_queries=[WorldModelQueryRecord(method="calibrate_with_truth", state_id=updated_candidate.world_state_ref, timestamp=_timestamp())],
            simulation_decision=SimulationDecision(decision="simulate", candidate_ids=[candidate_id], reasons=["feedback ingested"]),
            decision_rationale=["updated candidate lifecycle and calibrated world model with simulator truth"],
            reward_or_progress_signal=1.0 if feasible else -0.25,
            simulation_result_ref=truth.artifact_refs[0] if truth.artifact_refs else None,
            trust_snapshot=updated_candidate.predicted_uncertainty,
        )
        updated_state = self._refresh_search_state(
            search_state,
            pool_state=pool_state,
            frontier_ids=[candidate.candidate_id for candidate in frontier_candidates(pool_state)],
            budget_state=budget_state,
            provenance_source="simulation_feedback",
            traces=[trace],
            calibration_required=mismatch,
            extra_failures=["trust_violation"] if mismatch else [],
        ).model_copy(update={"pending_simulation_refs": [ref for ref in search_state.pending_simulation_refs if ref != candidate_id]})
        return SimulationFeedbackResponse(
            search_state=updated_state,
            updated_world_model_bundle=self.world_model_bundle,
            traces=[trace],
        )

    def advance_phase(self, search_state: SearchState) -> CandidateBatchResponse:
        """Advance the planning phase when explicit conditions are met."""

        current = search_state.phase_state.current_phase
        next_phase = current
        rationale: list[str] = []
        if current == "feasibility_bootstrapping" and search_state.best_known_feasible is not None:
            next_phase = "performance_refinement"
            rationale.append("feasible candidate discovered")
        elif current == "performance_refinement" and search_state.risk_context.calibration_required:
            next_phase = "calibration_recovery"
            rationale.append("world-model mismatch detected")
        elif current == "performance_refinement" and search_state.best_known_feasible is not None and search_state.budget_state.simulations_used > 0:
            next_phase = "robustness_verification"
            rationale.append("verified feasible candidate available")
        elif current == "calibration_recovery" and not search_state.risk_context.calibration_required:
            next_phase = "performance_refinement"
            rationale.append("trust restored after calibration")

        changed = next_phase != current
        phase_state = advance_phase_state(search_state.phase_state, next_phase, len(search_state.trace_log)) if changed else increment_phase_iteration(search_state.phase_state, improved=False)
        trace = self._make_trace(
            search_state,
            outcome_tag="phase_advanced",
            selected_candidate_id=search_state.best_known_feasible.candidate_id if search_state.best_known_feasible else None,
            executed_action_chain=[],
            world_model_queries=[],
            simulation_decision=SimulationDecision(decision="keep", candidate_ids=[], reasons=rationale or ["phase unchanged"]),
            decision_rationale=rationale or ["phase advance conditions not met"],
            reward_or_progress_signal=1.0 if changed else 0.0,
            trust_snapshot=self.world_model_service.predict_feasibility(search_state.current_world_state).trust_assessment,
        )
        updated_state = self._refresh_search_state(
            search_state,
            phase_state=phase_state,
            provenance_source="phase_transition",
            traces=[trace],
            calibration_required=search_state.risk_context.calibration_required,
            extra_failures=search_state.risk_context.active_failure_modes,
        )
        return CandidateBatchResponse(search_state=updated_state, candidates=updated_state.candidate_pool_state.candidates, traces=[trace])

    def should_terminate(self, search_state: SearchState) -> TerminationDecision:
        """Decide whether the current search should terminate."""

        if (
            search_state.best_known_feasible is not None
            and self.planning_bundle.termination_policy.stop_on_verified_feasible
            and search_state.phase_state.current_phase in {"robustness_verification", "terminated"}
        ):
            return TerminationDecision(should_terminate=True, reason="verified_solution_ready", recommended_action="terminate")
        if len(search_state.trace_log) >= self.planning_bundle.termination_policy.max_iterations:
            return TerminationDecision(should_terminate=True, reason="budget_exhausted", recommended_action="terminate")
        if remaining_simulations(search_state.budget_state) == 0 and not search_state.pending_simulation_refs:
            return TerminationDecision(should_terminate=True, reason="budget_exhausted", recommended_action="terminate")
        if search_state.phase_state.stagnation_counter >= self.planning_bundle.termination_policy.stagnation_patience:
            return TerminationDecision(should_terminate=True, reason="stagnation_limit", recommended_action="recalibrate")
        if search_state.risk_context.calibration_required and remaining_simulations(search_state.budget_state) == 0:
            return TerminationDecision(should_terminate=True, reason="world_model_safety_block", recommended_action="terminate")
        return TerminationDecision(should_terminate=False, reason=None, recommended_action="continue")

    def get_best_result(self, search_state: SearchState) -> PlanningBestResult:
        """Return the best known planning result."""

        best_id = search_state.best_known_feasible.candidate_id if search_state.best_known_feasible else (
            search_state.best_known_infeasible.candidate_id if search_state.best_known_infeasible else None
        )
        candidate = find_candidate(search_state.candidate_pool_state, best_id) if best_id else None
        return PlanningBestResult(
            candidate=candidate,
            phase=search_state.phase_state.current_phase,
            summary={
                "candidate_count": len(search_state.candidate_pool_state.candidates),
                "frontier_count": len(search_state.frontier_state.frontier_candidate_ids),
                "proxy_evaluations_used": search_state.budget_state.proxy_evaluations_used,
                "rollouts_used": search_state.budget_state.rollouts_used,
                "simulations_used": search_state.budget_state.simulations_used,
                "feasible_found": search_state.best_known_feasible is not None,
            },
            termination_decision=self.should_terminate(search_state),
        )

    def validate_search_state(self, search_state: SearchState):
        """Validate a SearchState against the planning bundle."""

        return validate_search_state(self.planning_bundle, search_state)
