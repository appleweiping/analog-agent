"""Selection and ranking helpers for the planning layer."""

from __future__ import annotations

from libs.schema.planning import CandidateRecord, SelectionPolicy, SimulationSelectionResponse


def apply_priority_scores(
    candidates: list[CandidateRecord],
    *,
    ranking_scores: dict[str, float],
    policy: SelectionPolicy,
) -> list[CandidateRecord]:
    """Update candidate priority scores from world-model ranking outputs."""

    updated: list[CandidateRecord] = []
    for candidate in candidates:
        feasibility = candidate.predicted_feasibility.overall_feasibility if candidate.predicted_feasibility else 0.0
        uncertainty = candidate.predicted_uncertainty.uncertainty_score if candidate.predicted_uncertainty else 1.0
        simulation_value = candidate.simulation_value_estimate.estimated_value if candidate.simulation_value_estimate else 0.0
        base = ranking_scores.get(candidate.world_state_ref, ranking_scores.get(candidate.candidate_id, candidate.priority_score))
        if policy.prioritize_feasibility:
            base += feasibility
        base -= uncertainty * policy.uncertainty_penalty
        base += simulation_value * policy.simulation_value_weight
        dominance = "nondominated"
        if feasibility >= max(policy.min_feasible_probability, 0.8):
            dominance = "boundary_feasible"
        elif feasibility >= 0.45:
            dominance = "boundary_infeasible"
        elif feasibility < 0.2:
            dominance = "dominated"
        updated.append(candidate.model_copy(update={"priority_score": round(base, 6), "dominance_status": dominance}))
    return sorted(updated, key=lambda item: item.priority_score, reverse=True)


def choose_best_known(candidates: list[CandidateRecord], *, feasible: bool) -> CandidateRecord | None:
    """Choose the current best feasible or infeasible candidate."""

    filtered = []
    for candidate in candidates:
        probability = candidate.predicted_feasibility.overall_feasibility if candidate.predicted_feasibility else 0.0
        if feasible and probability >= 0.8:
            filtered.append(candidate)
        if not feasible and probability < 0.8:
            filtered.append(candidate)
    if not filtered:
        return None
    return sorted(filtered, key=lambda item: item.priority_score, reverse=True)[0]

