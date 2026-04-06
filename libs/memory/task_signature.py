"""Task-signature helpers for the memory layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.memory import TaskSignature
from libs.utils.hashing import stable_hash


def build_task_signature(task: DesignTask) -> TaskSignature:
    """Build the deterministic sixth-layer task signature from DesignTask."""

    constraint_vector = sorted(
        f"{constraint.metric}:{constraint.relation}:{constraint.threshold if constraint.threshold is not None else f'{constraint.lower_threshold},{constraint.upper_threshold}'}"
        for constraint in task.constraints.hard_constraints
    )
    environment_profile = [
        f"corners:{task.evaluation_plan.corners_policy.mode}:{','.join(map(str, task.evaluation_plan.corners_policy.values))}",
        f"temperature:{task.evaluation_plan.temperature_policy.mode}:{','.join(map(str, task.evaluation_plan.temperature_policy.values))}",
        f"load:{task.evaluation_plan.load_policy.mode}:{','.join(map(str, task.evaluation_plan.load_policy.values))}",
        f"budget:{task.evaluation_plan.simulation_budget_class}",
    ]
    evaluation_profile = [
        f"{analysis.analysis_type}:{analysis.estimated_cost}:{','.join(sorted(analysis.required_metrics))}"
        for analysis in task.evaluation_plan.analyses
    ]
    design_space_shape = [
        f"dim:{len(task.design_space.variables)}",
        f"frozen:{len(task.design_space.frozen_variables)}",
        f"conditional:{len(task.design_space.conditional_variables)}",
        f"topology_mode:{task.topology.topology_mode}",
        *[f"{variable.name}:{variable.kind}:{variable.scale}:{variable.units}" for variable in task.design_space.variables],
    ]
    difficulty_hash = stable_hash(task.difficulty_profile.model_dump_json())
    return TaskSignature(
        circuit_family=task.circuit_family,
        task_type=task.task_type,
        constraint_vector=constraint_vector,
        environment_profile=environment_profile,
        evaluation_profile=evaluation_profile,
        design_space_shape=design_space_shape,
        difficulty_profile_hash=difficulty_hash,
    )
