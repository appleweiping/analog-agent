"""Trajectory ingestion and episode consolidation for the memory layer."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from libs.memory.task_signature import build_task_signature
from libs.schema.design_task import DesignTask
from libs.schema.memory import (
    ActionSequenceRecord,
    CandidateOutcomeSummary,
    ConstraintProfile,
    EpisodeMemoryRecord,
    EvidenceReference,
    FinalOutcomeSummary,
    InitialConditionsSnapshot,
    PhaseTransitionRecord,
    SearchSummary,
    SimulationBudgetProfile,
    TurningPointRecord,
    WorldModelBehaviorSummary,
)
from libs.schema.planning import CandidateRecord, OptimizationTrace, SearchState
from libs.schema.simulation import VerificationResult
from libs.utils.hashing import stable_hash


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_summary(candidate: CandidateRecord | None, verification: VerificationResult | None) -> CandidateOutcomeSummary | None:
    if candidate is None:
        return None
    feasible_probability = candidate.predicted_feasibility.overall_feasibility if candidate.predicted_feasibility else None
    truth_status = verification.feasibility_status if verification and verification.candidate_id == candidate.candidate_id else None
    return CandidateOutcomeSummary(
        candidate_id=candidate.candidate_id,
        lifecycle_status=candidate.lifecycle_status,
        feasible_probability=feasible_probability,
        priority_score=candidate.priority_score,
        world_state_ref=candidate.world_state_ref,
        truth_feasibility_status=truth_status,
    )


def _build_sequence(trace: OptimizationTrace, effect_type: str) -> ActionSequenceRecord:
    action_ids = [action.action_id for action in trace.executed_action_chain]
    action_families = [action.action_family for action in trace.executed_action_chain]
    observed_effects = [trace.outcome_tag, *trace.decision_rationale]
    return ActionSequenceRecord(
        sequence_id=f"seq_{stable_hash(trace.trace_id + effect_type)[:12]}",
        action_ids=action_ids,
        action_families=action_families,
        effect_type=effect_type,
        observed_effects=observed_effects,
        evidence_refs=[trace.trace_id],
    )


def _dominant_failure_modes(search_state: SearchState, verification: VerificationResult | None) -> list[str]:
    failures = list(search_state.risk_context.active_failure_modes)
    if verification is not None:
        failures.extend(assessment.constraint_name for assessment in verification.constraint_assessment if not assessment.is_satisfied)
        if verification.failure_attribution.primary_failure_class != "none":
            failures.append(verification.failure_attribution.primary_failure_class)
    return list(Counter(failures).keys())[:5]


def _turning_points(search_state: SearchState, verification: VerificationResult | None) -> list[TurningPointRecord]:
    turning_points: list[TurningPointRecord] = []
    for trace in search_state.trace_log:
        if trace.outcome_tag == "phase_advanced":
            turning_points.append(
                TurningPointRecord(
                    turning_point_id=f"tp_{stable_hash(trace.trace_id)[:12]}",
                    turning_point_type="phase_shift",
                    candidate_id=trace.selected_candidate_id,
                    description_tags=[trace.outcome_tag, *trace.decision_rationale],
                    supporting_evidence_refs=[trace.trace_id],
                )
            )
        if trace.trust_snapshot.must_escalate or trace.trust_snapshot.hard_block:
            turning_points.append(
                TurningPointRecord(
                    turning_point_id=f"tp_{stable_hash(trace.trace_id + 'trust')[:12]}",
                    turning_point_type="trust_violation",
                    candidate_id=trace.selected_candidate_id,
                    description_tags=list(trace.trust_snapshot.reasons),
                    supporting_evidence_refs=[trace.trace_id],
                )
            )
    if verification is not None:
        turning_points.append(
            TurningPointRecord(
                turning_point_id=f"tp_{stable_hash(verification.result_id)[:12]}",
                turning_point_type="verification_success" if verification.feasibility_status in {"feasible_nominal", "feasible_certified"} else "verification_failure",
                candidate_id=verification.candidate_id,
                description_tags=[verification.feasibility_status, verification.failure_attribution.primary_failure_class],
                supporting_evidence_refs=[verification.result_id],
            )
        )
    return turning_points[:8]


def consolidate_episode(
    task: DesignTask,
    search_state: SearchState,
    verification: VerificationResult | None = None,
) -> EpisodeMemoryRecord:
    """Consolidate one full planning/simulation episode into an episode memory record."""

    candidates = search_state.candidate_pool_state.candidates
    verified_candidates = sum(1 for candidate in candidates if candidate.lifecycle_status == "verified")
    rejected_candidates = sum(1 for candidate in candidates if candidate.lifecycle_status == "rejected")
    effective_sequences: list[ActionSequenceRecord] = []
    ineffective_sequences: list[ActionSequenceRecord] = []
    for trace in search_state.trace_log:
        if not trace.executed_action_chain:
            continue
        if trace.reward_or_progress_signal >= 0.0 and trace.outcome_tag in {"candidate_evaluated", "phase_advanced", "simulation_feedback_ingested"}:
            effective_sequences.append(_build_sequence(trace, "effective"))
        else:
            ineffective_sequences.append(_build_sequence(trace, "ineffective"))

    phase_transitions = [
        PhaseTransitionRecord(
            from_phase=trace.search_state_snapshot.split(";")[0].replace("phase=", ""),
            to_phase=search_state.phase_state.current_phase if trace.outcome_tag == "phase_advanced" else search_state.phase_state.current_phase,
            trigger=",".join(trace.decision_rationale) or trace.outcome_tag,
            step_index=trace.step_index,
            supporting_trace_refs=[trace.trace_id],
        )
        for trace in search_state.trace_log
        if trace.outcome_tag == "phase_advanced"
    ]
    evidence_refs = [
        EvidenceReference(
            evidence_id=f"ev_{stable_hash(search_state.search_id + trace.trace_id)[:12]}",
            source_layer="layer4",
            source_object_type="OptimizationTrace",
            source_object_id=trace.trace_id,
            evidence_kind=trace.outcome_tag,
            artifact_refs=[trace.simulation_result_ref] if trace.simulation_result_ref else [],
        )
        for trace in search_state.trace_log
    ]
    if verification is not None:
        evidence_refs.append(
            EvidenceReference(
                evidence_id=f"ev_{stable_hash(verification.result_id)[:12]}",
                source_layer="layer5",
                source_object_type="VerificationResult",
                source_object_id=verification.result_id,
                evidence_kind="ground_truth_verification",
                artifact_refs=verification.artifact_refs,
            )
        )
        evidence_refs.append(
            EvidenceReference(
                evidence_id=f"ev_{stable_hash(verification.calibration_payload.calibration_id)[:12]}",
                source_layer="layer5",
                source_object_type="CalibrationFeedback",
                source_object_id=verification.calibration_payload.calibration_id,
                evidence_kind="calibration_feedback",
                artifact_refs=verification.calibration_payload.truth_record.artifact_refs,
            )
        )

    outcome_status = "budget_exhausted" if search_state.budget_state.budget_pressure >= 0.98 else "failure"
    if verification is not None and verification.feasibility_status in {"feasible_nominal", "feasible_certified"}:
        outcome_status = "verified_success"
    elif verification is not None and verification.feasibility_status == "near_feasible":
        outcome_status = "partial_success"
    elif search_state.risk_context.trust_alert_count and search_state.phase_state.current_phase == "terminated":
        outcome_status = "trust_blocked"

    return EpisodeMemoryRecord(
        episode_memory_id=f"episode_{stable_hash(search_state.episode_id + task.task_id)[:12]}",
        task_id=task.task_id,
        task_signature=build_task_signature(task),
        circuit_family=task.circuit_family,
        task_type=task.task_type,
        initial_conditions=InitialConditionsSnapshot(
            init_strategy=task.initial_state.init_strategy,
            seed_count=len(task.initial_state.seed_candidates),
            randomization_strategy=task.initial_state.randomization_policy.strategy,
            warm_start_source=task.initial_state.warm_start_source,
        ),
        constraint_profile=ConstraintProfile(
            hard_constraint_names=[constraint.name for constraint in task.constraints.hard_constraints],
            tightness=task.difficulty_profile.constraint_tightness,
            feasibility_rules=task.constraints.feasibility_rules,
            operating_region_rules=task.constraints.operating_region_rules,
        ),
        search_summary=SearchSummary(
            episode_id=search_state.episode_id,
            search_id=search_state.search_id,
            total_candidates=len(candidates),
            verified_candidates=verified_candidates,
            rejected_candidates=rejected_candidates,
            trace_length=len(search_state.trace_log),
            final_phase=search_state.phase_state.current_phase,
            termination_reason=search_state.trace_log[-1].outcome_tag if search_state.trace_log else None,
            simulation_budget_used=search_state.budget_state.simulations_used,
            proxy_budget_used=search_state.budget_state.proxy_evaluations_used,
            rollout_budget_used=search_state.budget_state.rollouts_used,
            calibration_budget_used=search_state.budget_state.calibrations_used,
        ),
        best_feasible_result=_candidate_summary(next((candidate for candidate in candidates if candidate.candidate_id == (search_state.best_known_feasible.candidate_id if search_state.best_known_feasible else None)), None), verification),
        best_infeasible_result=_candidate_summary(next((candidate for candidate in candidates if candidate.candidate_id == (search_state.best_known_infeasible.candidate_id if search_state.best_known_infeasible else None)), None), verification),
        dominant_failure_modes=_dominant_failure_modes(search_state, verification),
        effective_action_sequences=effective_sequences[: task.validation_status.completeness_score and 4 or 4],
        ineffective_action_sequences=ineffective_sequences[:4],
        world_model_behavior_summary=WorldModelBehaviorSummary(
            trust_alert_count=search_state.risk_context.trust_alert_count,
            must_escalate_count=sum(1 for trace in search_state.trace_log if trace.trust_snapshot.must_escalate),
            disagreement_flags=verification.calibration_payload.trust_violation_flags if verification is not None else [],
            calibration_priority=verification.calibration_payload.retrain_priority if verification is not None else None,
            observed_bias_metrics=sorted((verification.calibration_payload.residual_metrics or {}).keys()) if verification is not None else [],
        ),
        simulation_budget_profile=SimulationBudgetProfile(
            proxy_evaluations_used=search_state.budget_state.proxy_evaluations_used,
            rollouts_used=search_state.budget_state.rollouts_used,
            simulations_used=search_state.budget_state.simulations_used,
            calibrations_used=search_state.budget_state.calibrations_used,
            budget_pressure=search_state.budget_state.budget_pressure,
        ),
        phase_transition_trace=phase_transitions,
        turning_points=_turning_points(search_state, verification),
        final_outcome=FinalOutcomeSummary(
            outcome_status=outcome_status,
            best_candidate_id=verification.candidate_id if verification is not None else (search_state.best_known_feasible.candidate_id if search_state.best_known_feasible else None),
            best_feasibility_status=verification.feasibility_status if verification is not None else None,
            robustness_status=verification.robustness_summary.certification_status if verification is not None else None,
            failure_classes=_dominant_failure_modes(search_state, verification),
        ),
        evidence_refs=evidence_refs,
        confidence_score=0.85 if verification is not None else 0.65,
        timestamp=_timestamp(),
    )
