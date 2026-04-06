"""Phase-control helpers for the planning layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.planning import PhaseController, PhaseState


def determine_initial_phase(task: DesignTask) -> str:
    """Resolve the initial planning phase from task hints."""

    if task.solver_hint.needs_feasibility_first or task.difficulty_profile.expected_feasibility in {"low", "unknown"}:
        return "feasibility_bootstrapping"
    return "performance_refinement"


def initialize_phase_state(controller: PhaseController) -> PhaseState:
    """Create the initial runtime phase state."""

    return PhaseState(
        current_phase=controller.initial_phase,
        phase_iteration=0,
        successful_phase_transitions=0,
        stagnation_counter=0,
        last_phase_change_step=0,
    )


def advance_phase_state(state: PhaseState, next_phase: str, step_index: int) -> PhaseState:
    """Create a new phase state after a legal transition."""

    return state.model_copy(
        update={
            "current_phase": next_phase,
            "phase_iteration": 0,
            "successful_phase_transitions": state.successful_phase_transitions + 1,
            "stagnation_counter": 0,
            "last_phase_change_step": step_index,
        }
    )


def increment_phase_iteration(state: PhaseState, *, improved: bool) -> PhaseState:
    """Advance the iteration counter within the current phase."""

    return state.model_copy(
        update={
            "phase_iteration": state.phase_iteration + 1,
            "stagnation_counter": 0 if improved else state.stagnation_counter + 1,
        }
    )

