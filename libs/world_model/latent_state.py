"""Deterministic latent-state diagnostics for world-model rollouts.

This module is intentionally a scaffold, not a learned JEPA-style model. It
creates a compact, comparable state vector from the formal WorldState so the
planner can measure rollout drift and decide when real simulation evidence is
needed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from libs.schema.world_model import RolloutResponse, WorldState

LATENT_STATE_VERSION = "latent-state-v1"

METRIC_SCALES = {
    "area_um2": 1000.0,
    "dc_gain_db": 100.0,
    "delay_ns": 10.0,
    "gbw_hz": 1.0e8,
    "input_common_mode_v": 1.0,
    "input_referred_noise_nv_per_sqrt_hz": 20.0,
    "line_regulation_mv_per_v": 5.0,
    "load_regulation_mv_per_ma": 5.0,
    "noise_nv_per_sqrt_hz": 20.0,
    "offset_mv": 10.0,
    "output_swing_v": 1.0,
    "phase_margin_deg": 90.0,
    "power_w": 1.0e-3,
    "psrr_db": 60.0,
    "slew_rate_v_per_us": 10.0,
    "temperature_coefficient_ppm_per_c": 25.0,
}


@dataclass(frozen=True)
class LatentState:
    """A deterministic feature view of a WorldState."""

    state_id: str
    task_id: str
    version: str
    feature_vector: dict[str, float]
    feature_groups: dict[str, list[str]]
    provenance: dict[str, str]


@dataclass(frozen=True)
class LatentStateComparison:
    """Distance summary between two latent states."""

    version: str
    lhs_state_id: str
    rhs_state_id: str
    feature_count: int
    changed_feature_count: int
    l1_distance: float
    l2_distance: float
    cosine_similarity: float
    max_abs_delta: float
    top_deltas: list[tuple[str, float]]


def _finite(value: float | int | None, default: float = 0.0) -> float:
    if value is None:
        return default
    numeric = float(value)
    if not math.isfinite(numeric):
        return default
    return numeric


def _rounded(value: float) -> float:
    return round(_finite(value), 8)


def _signed_log1p(value: float) -> float:
    numeric = _finite(value)
    return math.copysign(math.log1p(abs(numeric)), numeric)


def _scaled(value: float | int | None, scale: float) -> float:
    return _rounded(_finite(value) / max(abs(scale), 1.0e-18))


def _scaled_log(value: float | int | None, scale: float) -> float:
    return _rounded(_signed_log1p(_finite(value) / max(abs(scale), 1.0e-18)))


def _add_feature(
    features: dict[str, float],
    groups: dict[str, list[str]],
    group: str,
    name: str,
    value: float,
) -> None:
    key = f"{group}:{name}"
    features[key] = _rounded(value)
    groups.setdefault(group, []).append(key)


def _add_numeric_map(
    features: dict[str, float],
    groups: dict[str, list[str]],
    group: str,
    values: dict[str, float],
    *,
    scale: float,
    log_scale: bool = False,
) -> None:
    for name, value in sorted(values.items()):
        encoded = _scaled_log(value, scale) if log_scale else _scaled(value, scale)
        _add_feature(features, groups, group, name, encoded)


def _group_count(groups: Iterable[object]) -> float:
    return float(len(list(groups)))


def build_latent_state(state: WorldState) -> LatentState:
    """Project a WorldState into a compact deterministic feature vector."""

    features: dict[str, float] = {}
    groups: dict[str, list[str]] = {}

    for parameter in sorted(state.parameter_state, key=lambda item: item.variable_name):
        _add_feature(features, groups, "param", parameter.variable_name, parameter.normalized_value)
        _add_feature(features, groups, "param_active", parameter.variable_name, 1.0 if parameter.is_active else 0.0)
        if parameter.is_frozen:
            _add_feature(features, groups, "param_frozen", parameter.variable_name, 1.0)

    env = state.environment_state
    _add_feature(features, groups, "env", "temperature_c", _scaled(env.temperature_c, 150.0))
    _add_feature(features, groups, "env", "supply_voltage_v", _scaled(env.supply_voltage_v, 5.0))
    _add_feature(features, groups, "env", "load_cap_pf", _scaled(env.load_cap_f, 1.0e-12))
    _add_feature(features, groups, "env", "output_load_log", _scaled_log(env.output_load_ohm, 1000.0))

    graph = state.topology_context.graph_statistics
    _add_feature(features, groups, "graph", "node_count", _scaled(graph.node_count, 100.0))
    _add_feature(features, groups, "graph", "edge_count", _scaled(graph.edge_count, 200.0))
    _add_feature(features, groups, "graph", "symmetry_group_count", _scaled(graph.symmetry_group_count, 20.0))
    _add_feature(features, groups, "graph", "fanout_estimate", _scaled(graph.fanout_estimate, 20.0))

    structural = state.structural_features
    _add_feature(features, groups, "structure", "module_role_count", _group_count(structural.module_roles) / 20.0)
    _add_feature(features, groups, "structure", "key_subcircuit_count", _group_count(structural.key_subcircuits) / 20.0)
    _add_feature(features, groups, "structure", "symmetry_group_count", _group_count(structural.symmetry_groups) / 20.0)

    for metric in sorted(state.performance_observation, key=lambda item: item.metric):
        _add_feature(
            features,
            groups,
            "metric",
            metric.metric,
            _scaled_log(metric.value, METRIC_SCALES.get(metric.metric, 1.0)),
        )
        _add_feature(features, groups, "metric_uncertainty", metric.metric, metric.uncertainty)

    for constraint in sorted(state.constraint_observation, key=lambda item: (item.constraint_group, item.constraint_name)):
        name = f"{constraint.constraint_group}.{constraint.constraint_name}"
        _add_feature(features, groups, "constraint_margin", name, _scaled_log(constraint.margin, 1.0))
        _add_feature(features, groups, "constraint_satisfaction", name, constraint.satisfied_probability)

    op = state.operating_point_state
    _add_numeric_map(features, groups, "op_node_v", op.node_voltage_summary, scale=2.0)
    _add_numeric_map(features, groups, "op_gm", op.gm, scale=1.0e-3, log_scale=True)
    _add_numeric_map(features, groups, "op_gds", op.gds, scale=1.0e-4, log_scale=True)
    _add_numeric_map(features, groups, "op_ro", op.ro, scale=1.0e4, log_scale=True)
    _add_numeric_map(features, groups, "op_current", op.drain_current_a, scale=1.0e-4, log_scale=True)
    _add_numeric_map(features, groups, "op_overdrive", op.overdrive_v, scale=0.5)
    _add_numeric_map(features, groups, "op_gm_over_id", op.gm_over_id, scale=20.0)
    _add_numeric_map(features, groups, "op_branch_balance", op.branch_balance, scale=1.0)
    _add_numeric_map(features, groups, "op_stability", op.stability_proxy, scale=90.0)

    uncertainty = state.uncertainty_context
    _add_feature(features, groups, "uncertainty", "epistemic_score", uncertainty.epistemic_score)
    _add_feature(features, groups, "uncertainty", "aleatoric_score", uncertainty.aleatoric_score)
    _add_feature(features, groups, "uncertainty", "ood_score", uncertainty.ood_score)

    normalized_groups = {group: sorted(keys) for group, keys in sorted(groups.items())}
    return LatentState(
        state_id=state.state_id,
        task_id=state.task_id,
        version=LATENT_STATE_VERSION,
        feature_vector={key: features[key] for key in sorted(features)},
        feature_groups=normalized_groups,
        provenance={
            "state_origin": state.provenance.state_origin,
            "source_stage": state.provenance.source_stage,
            "analysis_fidelity": state.provenance.analysis_fidelity,
            "corner": state.environment_state.corner,
        },
    )


def _coerce_latent(state: WorldState | LatentState) -> LatentState:
    if isinstance(state, LatentState):
        return state
    return build_latent_state(state)


def compare_latent_states(
    lhs: WorldState | LatentState,
    rhs: WorldState | LatentState,
    *,
    top_k: int = 8,
) -> LatentStateComparison:
    """Compare two states in the deterministic latent feature space."""

    lhs_latent = _coerce_latent(lhs)
    rhs_latent = _coerce_latent(rhs)
    keys = sorted(set(lhs_latent.feature_vector) | set(rhs_latent.feature_vector))
    deltas = {
        key: rhs_latent.feature_vector.get(key, 0.0) - lhs_latent.feature_vector.get(key, 0.0)
        for key in keys
    }
    l1 = sum(abs(value) for value in deltas.values())
    l2 = math.sqrt(sum(value * value for value in deltas.values()))
    lhs_norm = math.sqrt(sum(lhs_latent.feature_vector.get(key, 0.0) ** 2 for key in keys))
    rhs_norm = math.sqrt(sum(rhs_latent.feature_vector.get(key, 0.0) ** 2 for key in keys))
    dot = sum(lhs_latent.feature_vector.get(key, 0.0) * rhs_latent.feature_vector.get(key, 0.0) for key in keys)
    if lhs_norm == 0.0 and rhs_norm == 0.0:
        cosine = 1.0
    elif lhs_norm == 0.0 or rhs_norm == 0.0:
        cosine = 0.0
    else:
        cosine = dot / (lhs_norm * rhs_norm)
    top_deltas = sorted(deltas.items(), key=lambda item: (abs(item[1]), item[0]), reverse=True)[:top_k]
    changed = [value for value in deltas.values() if abs(value) > 1.0e-9]
    return LatentStateComparison(
        version=LATENT_STATE_VERSION,
        lhs_state_id=lhs_latent.state_id,
        rhs_state_id=rhs_latent.state_id,
        feature_count=len(keys),
        changed_feature_count=len(changed),
        l1_distance=_rounded(l1),
        l2_distance=_rounded(l2),
        cosine_similarity=_rounded(max(-1.0, min(1.0, cosine))),
        max_abs_delta=_rounded(max((abs(value) for value in deltas.values()), default=0.0)),
        top_deltas=[(key, _rounded(value)) for key, value in top_deltas if abs(value) > 1.0e-9],
    )


def rollout_error_summary(
    rollout: RolloutResponse,
    *,
    initial_state: WorldState | None = None,
    truth_terminal_state: WorldState | None = None,
) -> dict[str, object]:
    """Summarize latent rollout drift and optional terminal truth error."""

    path_states = []
    if initial_state is not None:
        path_states.append(initial_state)
    path_states.extend(step.transition.next_state for step in rollout.steps)
    path_distances = [
        compare_latent_states(previous, current).l2_distance
        for previous, current in zip(path_states, path_states[1:])
    ]
    terminal_error = None
    terminal_top_deltas: list[tuple[str, float]] = []
    if truth_terminal_state is not None:
        comparison = compare_latent_states(rollout.terminal_state, truth_terminal_state)
        terminal_error = comparison.l2_distance
        terminal_top_deltas = comparison.top_deltas

    return {
        "version": LATENT_STATE_VERSION,
        "initial_state_id": rollout.initial_state_id,
        "horizon": rollout.horizon,
        "step_count": len(rollout.steps),
        "terminal_state_id": rollout.terminal_state.state_id,
        "terminal_service_tier": rollout.trust_assessment.service_tier,
        "terminal_trust_reasons": list(rollout.trust_assessment.reasons),
        "step_simulation_decisions": [step.simulation_value.decision for step in rollout.steps],
        "must_escalate_step_count": sum(1 for step in rollout.steps if step.simulation_value.trust_assessment.must_escalate),
        "path_distance_count": len(path_distances),
        "path_l2_distances": path_distances,
        "terminal_l2_error": terminal_error,
        "terminal_top_deltas": terminal_top_deltas,
    }
