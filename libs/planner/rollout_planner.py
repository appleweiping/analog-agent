"""Rollout planning helpers for the planning layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.planning import RolloutConfig
from libs.schema.world_model import DesignAction, WorldState
from libs.world_model.action_builder import build_design_action


def build_candidate_actions(task: DesignTask, state: WorldState, phase: str, max_actions: int) -> list[DesignAction]:
    """Build deterministic candidate actions for one search state."""

    factors = {
        "feasibility_bootstrapping": (1.25, 0.85),
        "performance_refinement": (1.1, 0.92),
        "robustness_verification": (1.05, 0.97),
        "calibration_recovery": (1.02, 0.98),
    }
    up_factor, down_factor = factors.get(phase, (1.1, 0.9))
    preferred_names = list(task.difficulty_profile.sensitivity_hint)
    if not preferred_names:
        preferred_names = [variable.name for variable in task.design_space.variables]

    actions: list[DesignAction] = []
    for variable_name in preferred_names:
        variable = next((item for item in task.design_space.variables if item.name == variable_name), None)
        if variable is None or variable.kind not in {"continuous", "integer"}:
            continue
        scope = ["feasibility"] if phase == "feasibility_bootstrapping" else ["operating_point", "power"]
        actions.append(
            build_design_action(
                task,
                action_family="parameter_update",
                target_kind="variable",
                variable_names=[variable.name],
                operator="scale",
                payload={"factor": up_factor},
                expected_scope=scope,
                source="planner",
            )
        )
        actions.append(
            build_design_action(
                task,
                action_family="parameter_update",
                target_kind="variable",
                variable_names=[variable.name],
                operator="scale",
                payload={"factor": down_factor},
                expected_scope=scope,
                source="planner",
            )
        )
        if len(actions) >= max_actions:
            break
    if not actions:
        variable = task.design_space.variables[0]
        actions = [
            build_design_action(
                task,
                action_family="parameter_update",
                target_kind="variable",
                variable_names=[variable.name],
                operator="scale",
                payload={"factor": 1.1},
                expected_scope=["operating_point"],
                source="planner",
            )
        ]
    return actions[:max_actions]


def build_rollout_action_chain(task: DesignTask, state: WorldState, phase: str, config: RolloutConfig) -> list[DesignAction]:
    """Build one deterministic rollout action chain."""

    return build_candidate_actions(task, state, phase, max_actions=max(config.horizon, 1))[: config.horizon]

