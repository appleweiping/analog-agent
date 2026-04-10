"""Selection and ranking helpers for the planning layer."""

from __future__ import annotations

import math

from libs.schema.planning import CandidateRecord, SelectionPolicy, SimulationSelectionResponse


def objective_performance_score(candidate: CandidateRecord) -> float:
    """Compute an explicit performance term from predicted metrics."""

    if candidate.predicted_metrics is None:
        return 0.0
    score = 0.0
    for estimate in candidate.predicted_metrics.metrics:
        value = max(abs(float(estimate.value)), 1e-12)
        normalized = math.log10(value + 1.0) / 10.0
        if estimate.metric in {"power_w", "input_referred_noise_nv_sqrt_hz", "settling_time_s"}:
            score -= normalized
        else:
            score += normalized
    return round(score, 6)


def score_candidate(candidate: CandidateRecord, *, ranking_score: float, policy: SelectionPolicy) -> float:
    """Explicit fourth-layer scoring function used for candidate ranking."""

    feasibility = candidate.predicted_feasibility.overall_feasibility if candidate.predicted_feasibility else 0.0
    uncertainty = candidate.predicted_uncertainty.uncertainty_score if candidate.predicted_uncertainty else 1.0
    simulation_value = candidate.simulation_value_estimate.estimated_value if candidate.simulation_value_estimate else 0.0
    performance = objective_performance_score(candidate)

    score = ranking_score
    if policy.prioritize_feasibility:
        score += 2.0 * feasibility
    score += performance
    score -= uncertainty * policy.uncertainty_penalty
    score += simulation_value * policy.simulation_value_weight
    return round(score, 6)


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
        base = ranking_scores.get(candidate.world_state_ref, ranking_scores.get(candidate.candidate_id, candidate.priority_score))
        base = score_candidate(candidate, ranking_score=base, policy=policy)
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
