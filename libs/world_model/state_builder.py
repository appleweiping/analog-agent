"""World-state builder for the third layer."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from libs.schema.design_task import DesignTask
from libs.schema.world_model import (
    ConstraintObservation,
    EnvironmentState,
    EvaluationContext,
    GraphStatistics,
    HistoryContext,
    ParameterValue,
    ProvenanceRecord,
    StructuralFeatures,
    TopologyContext,
    UncertaintyContext,
    UncertaintyFieldState,
    WorldState,
)
from libs.utils.hashing import stable_hash
from libs.world_model.design_task_adapter import build_graph_statistics, resolve_active_family
from libs.world_model.feature_projection import build_metric_estimates, build_operating_point, evaluate_constraints, project_metrics


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_value(variable, value: float | int | str | bool) -> float:
    if variable.kind in {"categorical", "binary"}:
        choices = list(variable.domain.choices)
        if not choices:
            return 0.0
        try:
            return round(choices.index(value) / max(1, len(choices) - 1), 4)
        except ValueError:
            return 1.2
    if variable.domain.lower is None or variable.domain.upper is None:
        return 0.0
    lower = float(variable.domain.lower)
    upper = float(variable.domain.upper)
    numeric = float(value)
    if variable.scale == "log":
        lower = max(lower, 1e-18)
        upper = max(upper, lower * 1.0001)
        numeric = max(numeric, lower * 1e-6)
        return round((math.log10(numeric) - math.log10(lower)) / (math.log10(upper) - math.log10(lower)), 4)
    return round((numeric - lower) / max(1e-18, upper - lower), 4)


def _symmetry_groups(task: DesignTask) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for variable in task.design_space.variables:
        if variable.coupling_group:
            groups.setdefault(variable.coupling_group, []).append(variable.name)
    return [sorted(names) for names in groups.values() if len(names) > 1]


def build_world_state(
    task: DesignTask,
    parameter_values: dict[str, float | int | str | bool] | None = None,
    *,
    corner: str | None = None,
    temperature_c: float | None = None,
    analysis_fidelity: str = "quick_screening",
    analysis_intent: str | None = None,
    output_load_ohm: float | None = None,
    provenance_type: str = "offline_dataset",
    provenance_stage: str = "initial",
    artifact_refs: list[str] | None = None,
    recent_actions=None,
) -> WorldState:
    """Build a formal WorldState from a DesignTask and optional overrides."""

    parameter_values = parameter_values or {}
    timestamp = datetime.now(timezone.utc).isoformat()
    graph_stats: GraphStatistics = build_graph_statistics(task)
    topology_context = TopologyContext(
        topology_mode=task.topology.topology_mode,
        template_id=task.topology.template_id,
        template_version=task.topology.template_version,
        port_names=[port.name for port in task.topology.ports],
        instance_names=[instance.name for instance in task.topology.instances_schema],
        instance_roles=[instance.role for instance in task.topology.instances_schema],
        topology_constraints=[constraint.name for constraint in task.topology.topology_constraints],
        graph_statistics=graph_stats,
    )

    resolved_parameters: list[ParameterValue] = []
    for variable in task.design_space.variables:
        value = parameter_values.get(variable.name, variable.default)
        if value is None:
            if variable.kind == "categorical" and variable.domain.choices:
                value = variable.domain.choices[0]
            elif variable.kind == "binary":
                value = 0
            else:
                value = 0.0
        if isinstance(value, str) and variable.kind in {"continuous", "integer"}:
            value = float(value)
        resolved_parameters.append(
            ParameterValue(
                variable_name=variable.name,
                value=value,
                normalized_value=_normalize_value(variable, value),
                is_frozen=variable.name in task.design_space.frozen_variables,
                coupling_group=variable.coupling_group,
                is_active=True,
            )
        )

    environment = EnvironmentState(
        corner=corner or (task.evaluation_plan.corners_policy.values[0] if task.evaluation_plan.corners_policy.values else "tt"),
        temperature_c=float(temperature_c if temperature_c is not None else (task.evaluation_plan.temperature_policy.values[0] if task.evaluation_plan.temperature_policy.values else 27.0)),
        supply_voltage_v=task.initial_state.template_defaults.get("supply_voltage_v")
        or (1.2 if task.circuit_family.endswith("ota") or task.circuit_family == "comparator" else 1.8),
        load_cap_f=float(task.evaluation_plan.load_policy.values[0]) if task.evaluation_plan.load_policy.values else None,
        output_load_ohm=output_load_ohm,
        bias_mode="nominal",
        analysis_assumptions=[f"fidelity={analysis_fidelity}", f"task_type={task.task_type}"],
    )

    evaluation_context = EvaluationContext(
        analysis_intent=analysis_intent or ("tt_ac_screening" if analysis_fidelity == "quick_screening" else "task_conditioned_prediction"),
        analysis_fidelity=analysis_fidelity,
        target_metrics=list(task.objective.reporting_metrics),
        constraint_groups=[group.name for group in task.constraints.constraint_groups],
        objective_mode=task.objective.objective_mode,
    )
    skeleton_state = WorldState.model_construct(
        state_id="placeholder",
        task_id=task.task_id,
        topology_context=topology_context,
        parameter_state=resolved_parameters,
        environment_state=environment,
        evaluation_context=evaluation_context,
        operating_point_state=None,
        structural_features=None,
        performance_observation=[],
        constraint_observation=[],
        history_context=HistoryContext(recent_actions=list(recent_actions or []), trajectory_depth=len(list(recent_actions or [])), last_outcome=None),
        uncertainty_context=None,
        provenance=ProvenanceRecord(
            state_origin=provenance_type,
            source_stage=provenance_stage,
            analysis_fidelity=analysis_fidelity,
            artifact_refs=list(artifact_refs or []),
            created_at=timestamp,
        ),
    )

    metric_values, auxiliary = project_metrics(task, skeleton_state)
    ood_score = max((abs(parameter.normalized_value - _clip(parameter.normalized_value, 0.0, 1.0)) for parameter in resolved_parameters), default=0.0)
    epistemic = min(1.0, 0.15 + 0.45 * ood_score + 0.1 * (1 if task.circuit_family == "unknown" else 0))
    aleatoric = min(1.0, 0.12 + 0.15 * auxiliary["load_penalty"])
    confidence = _clip(1.0 - (0.6 * epistemic + 0.4 * aleatoric), 0.05, 0.98)
    performance = build_metric_estimates(metric_values, 1.0 - confidence, confidence)
    constraints: list[ConstraintObservation] = evaluate_constraints(task, metric_values)
    op_state = build_operating_point(task, skeleton_state, metric_values)
    structural = StructuralFeatures(
        module_roles=[instance.role for instance in task.topology.instances_schema],
        symmetry_groups=_symmetry_groups(task),
        key_subcircuits=[instance.name for instance in task.topology.instances_schema[:3]],
        graph_statistics=graph_stats,
    )
    uncertainty_context = UncertaintyContext(
        field_states=[
            UncertaintyFieldState(field_name="performance_observation", source="prediction", confidence=confidence),
            UncertaintyFieldState(field_name="constraint_observation", source="prediction", confidence=confidence),
            UncertaintyFieldState(field_name="operating_point_state", source="prediction", confidence=max(0.05, confidence - 0.08)),
        ],
        epistemic_score=epistemic,
        aleatoric_score=aleatoric,
        ood_score=_clip(ood_score, 0.0, 1.0),
    )
    signature = stable_hash(
        "|".join(
            [
                task.task_id,
                resolve_active_family(task, parameter_values),
                environment.corner,
                str(environment.temperature_c),
                analysis_fidelity,
                ",".join(f"{parameter.variable_name}={parameter.value}" for parameter in resolved_parameters),
            ]
        )
    )
    return WorldState(
        state_id=f"state_{signature[:12]}",
        task_id=task.task_id,
        topology_context=topology_context,
        parameter_state=resolved_parameters,
        environment_state=environment,
        evaluation_context=evaluation_context,
        operating_point_state=op_state,
        structural_features=structural,
        performance_observation=performance,
        constraint_observation=constraints,
        history_context=HistoryContext(recent_actions=list(recent_actions or []), trajectory_depth=len(list(recent_actions or [])), last_outcome=None),
        uncertainty_context=uncertainty_context,
        provenance=ProvenanceRecord(
            state_origin=provenance_type,
            source_stage=provenance_stage,
            analysis_fidelity=analysis_fidelity,
            artifact_refs=list(artifact_refs or []),
            created_at=timestamp,
        ),
    )


def build_world_state_from_design_task(
    task: DesignTask,
    parameter_values: dict[str, float | int | str | bool] | None = None,
    *,
    corner: str | None = None,
    temperature_c: float | None = None,
    analysis_fidelity: str = "quick_screening",
    analysis_intent: str | None = None,
    output_load_ohm: float | None = None,
    provenance_stage: str = "initial",
    artifact_refs: list[str] | None = None,
    recent_actions=None,
) -> WorldState:
    """Formal public builder aligned to DesignTask semantics."""

    provenance_map = {
        "initial": "offline_dataset",
        "predicted": "model_rollout",
        "simulated": "real_simulation",
    }
    return build_world_state(
        task,
        parameter_values=parameter_values,
        corner=corner,
        temperature_c=temperature_c,
        analysis_fidelity=analysis_fidelity,
        analysis_intent=analysis_intent,
        output_load_ohm=output_load_ohm,
        provenance_type=provenance_map[provenance_stage],
        provenance_stage=provenance_stage,
        artifact_refs=artifact_refs,
        recent_actions=recent_actions,
    )
