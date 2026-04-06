"""Model predictive control rollout helpers."""

from __future__ import annotations

from libs.planner.service import PlanningService
from libs.schema.planning import ActionPlanResponse, SearchState


def rollout(service: PlanningService, search_state: SearchState, horizon: int) -> ActionPlanResponse:
    """Plan a short model-predictive action chain."""

    return service.plan_next_actions(search_state, horizon=horizon)
