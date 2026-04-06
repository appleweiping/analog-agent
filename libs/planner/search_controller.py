"""Search loop coordination helpers."""

from __future__ import annotations

from libs.planner.service import PlanningService
from libs.schema.planning import SearchState


def next_iteration(iteration: int) -> int:
    """Advance the planner iteration counter."""

    return iteration + 1


def run_search_iteration(service: PlanningService, search_state: SearchState) -> SearchState:
    """Run one deterministic search iteration over the formal planning service."""

    proposed = service.propose_candidates(search_state).search_state
    evaluated = service.evaluate_candidates(proposed).search_state
    selected = service.select_for_simulation(evaluated).search_state
    advanced = service.advance_phase(selected).search_state
    return advanced
