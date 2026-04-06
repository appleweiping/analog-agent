"""Candidate management helpers for the planning layer."""

from __future__ import annotations

from datetime import datetime, timezone

from libs.schema.planning import (
    CandidateDecisionEvent,
    CandidateEvaluationEvent,
    CandidatePoolState,
    CandidateRecord,
    CandidateSummary,
)


def find_candidate(pool: CandidatePoolState, candidate_id: str) -> CandidateRecord | None:
    """Find one candidate by id."""

    return next((candidate for candidate in pool.candidates if candidate.candidate_id == candidate_id), None)


def upsert_candidate(pool: CandidatePoolState, candidate: CandidateRecord) -> CandidatePoolState:
    """Insert or replace one candidate record in the pool."""

    updated = [item for item in pool.candidates if item.candidate_id != candidate.candidate_id]
    updated.append(candidate)
    active_ids = [item.candidate_id for item in updated if item.lifecycle_status in {"proposed", "frontier", "queued_for_rollout", "queued_for_simulation", "best_feasible", "best_infeasible"}]
    archived_ids = [item.candidate_id for item in updated if item.lifecycle_status in {"verified", "archived"}]
    discarded_ids = [item.candidate_id for item in updated if item.lifecycle_status in {"screened_out", "rejected"}]
    return pool.model_copy(
        update={
            "candidates": updated,
            "active_candidate_ids": active_ids,
            "archived_candidate_ids": archived_ids,
            "discarded_candidate_ids": discarded_ids,
        }
    )


def append_evaluation_event(candidate: CandidateRecord, event_type: str, notes: list[str] | None = None) -> CandidateRecord:
    """Append one structured evaluation event to a candidate."""

    event = CandidateEvaluationEvent(
        event_type=event_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        notes=list(notes or []),
    )
    return candidate.model_copy(update={"evaluation_history": [*candidate.evaluation_history, event]})


def append_decision_event(candidate: CandidateRecord, decision: str, reason: str) -> CandidateRecord:
    """Append one structured decision event to a candidate."""

    event = CandidateDecisionEvent(
        decision=decision,
        reason=reason,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return candidate.model_copy(update={"decision_history": [*candidate.decision_history, event]})


def summarize_candidate(candidate: CandidateRecord) -> CandidateSummary:
    """Build a compact candidate summary for SearchState."""

    feasible_probability = candidate.predicted_feasibility.overall_feasibility if candidate.predicted_feasibility else 0.0
    return CandidateSummary(
        candidate_id=candidate.candidate_id,
        world_state_ref=candidate.world_state_ref,
        feasible_probability=feasible_probability,
        priority_score=round(candidate.priority_score, 6),
        lifecycle_status=candidate.lifecycle_status,
    )


def frontier_candidates(pool: CandidatePoolState) -> list[CandidateRecord]:
    """Return candidates eligible for further expansion."""

    return [
        candidate
        for candidate in pool.candidates
        if candidate.lifecycle_status in {"proposed", "frontier", "best_infeasible", "best_feasible"}
    ]

