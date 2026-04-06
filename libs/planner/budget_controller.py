"""Budget-control helpers for the planning layer."""

from __future__ import annotations

from libs.schema.planning import BudgetController, BudgetState


def initialize_budget_state(controller: BudgetController) -> BudgetState:
    """Create an initial runtime budget state from bundle policy."""

    return BudgetState(
        proxy_evaluation_budget=controller.max_proxy_evaluations,
        proxy_evaluations_used=0,
        rollout_budget=controller.max_rollouts,
        rollouts_used=0,
        simulation_budget=controller.max_real_simulations,
        simulations_used=0,
        calibration_budget=controller.max_calibration_updates,
        calibrations_used=0,
        budget_pressure=0.0,
    )


def _remaining(total: int, used: int) -> int:
    return max(total - used, 0)


def _budget_pressure(state: BudgetState) -> float:
    proxy_pressure = state.proxy_evaluations_used / max(1, state.proxy_evaluation_budget)
    rollout_pressure = state.rollouts_used / max(1, state.rollout_budget)
    simulation_pressure = state.simulations_used / max(1, state.simulation_budget)
    calibration_pressure = state.calibrations_used / max(1, state.calibration_budget)
    return round(max(proxy_pressure, rollout_pressure, simulation_pressure, calibration_pressure), 4)


def consume_proxy_evaluations(state: BudgetState, count: int = 1) -> BudgetState:
    """Consume planner proxy-evaluation budget."""

    updated = state.model_copy(update={"proxy_evaluations_used": state.proxy_evaluations_used + max(count, 0)})
    return updated.model_copy(update={"budget_pressure": _budget_pressure(updated)})


def consume_rollouts(state: BudgetState, count: int = 1) -> BudgetState:
    """Consume rollout budget."""

    updated = state.model_copy(update={"rollouts_used": state.rollouts_used + max(count, 0)})
    return updated.model_copy(update={"budget_pressure": _budget_pressure(updated)})


def consume_simulations(state: BudgetState, count: int = 1) -> BudgetState:
    """Consume real-simulation budget."""

    updated = state.model_copy(update={"simulations_used": state.simulations_used + max(count, 0)})
    return updated.model_copy(update={"budget_pressure": _budget_pressure(updated)})


def consume_calibrations(state: BudgetState, count: int = 1) -> BudgetState:
    """Consume calibration budget."""

    updated = state.model_copy(update={"calibrations_used": state.calibrations_used + max(count, 0)})
    return updated.model_copy(update={"budget_pressure": _budget_pressure(updated)})


def remaining_proxy_evaluations(state: BudgetState) -> int:
    """Return remaining proxy-evaluation budget."""

    return _remaining(state.proxy_evaluation_budget, state.proxy_evaluations_used)


def remaining_rollouts(state: BudgetState) -> int:
    """Return remaining rollout budget."""

    return _remaining(state.rollout_budget, state.rollouts_used)


def remaining_simulations(state: BudgetState) -> int:
    """Return remaining real-simulation budget."""

    return _remaining(state.simulation_budget, state.simulations_used)


def remaining_calibrations(state: BudgetState) -> int:
    """Return remaining calibration budget."""

    return _remaining(state.calibration_budget, state.calibrations_used)

