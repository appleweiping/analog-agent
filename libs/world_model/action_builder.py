"""Design-action builder for the world model layer."""

from __future__ import annotations

from datetime import datetime, timezone

from libs.schema.design_task import DesignTask
from libs.schema.world_model import ActionTarget, ActionValidityGuard, DesignAction
from libs.utils.hashing import stable_hash


def build_design_action(
    task: DesignTask,
    *,
    action_family: str,
    target_kind: str,
    variable_names: list[str] | None = None,
    topology_slot: str | None = None,
    operator: str,
    payload: dict[str, float | int | str | bool] | None = None,
    expected_scope: list[str] | None = None,
    source: str = "manual",
) -> DesignAction:
    """Build a formal DesignAction object."""

    timestamp = datetime.now(timezone.utc).isoformat()
    variable_names = variable_names or []
    signature = stable_hash(
        "|".join(
            [
                task.task_id,
                action_family,
                operator,
                ",".join(sorted(variable_names)),
                topology_slot or "",
                ",".join(f"{key}={value}" for key, value in sorted((payload or {}).items())),
            ]
        )
    )
    if action_family in {"topology_switch", "template_slot_mutation"}:
        allowed_topology_modes = ["template_family", "search_space"]
    else:
        allowed_topology_modes = [task.topology.topology_mode]
    return DesignAction(
        action_id=f"act_{signature[:12]}",
        task_id=task.task_id,
        action_family=action_family,
        action_target=ActionTarget(
            target_kind=target_kind,
            variable_names=variable_names,
            topology_slot=topology_slot,
            metadata={},
        ),
        action_operator=operator,
        action_payload=payload or {},
        expected_scope=expected_scope or ["operating_point"],
        validity_guard=ActionValidityGuard(
            requires_domain_membership=True,
            requires_coupling_integrity=True,
            blocked_when_frozen=True,
            allowed_topology_modes=allowed_topology_modes,
            allowed_task_types=[task.task_type],
        ),
        source=source,
        timestamp=timestamp,
    )
