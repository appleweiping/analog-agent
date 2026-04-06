"""Helpers for adapting DesignTask objects into world-model contracts."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.world_model import (
    EncodingSpec,
    GraphStatistics,
    PredictionHeads,
    HeadDefinition,
    StateSchema,
    ActionSchema,
    WORLD_MODEL_METRICS,
)

FAMILY_EXTRA_METRICS: dict[str, list[str]] = {
    "two_stage_ota": ["dc_gain_db", "gbw_hz", "phase_margin_deg", "slew_rate_v_per_us", "power_w", "area_um2", "noise_nv_per_sqrt_hz"],
    "folded_cascode_ota": ["dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w", "noise_nv_per_sqrt_hz", "area_um2"],
    "telescopic_ota": ["dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w", "output_swing_v", "input_common_mode_v"],
    "comparator": ["delay_ns", "offset_mv", "power_w", "area_um2"],
    "ldo": ["phase_margin_deg", "power_w", "psrr_db", "line_regulation_mv_per_v", "load_regulation_mv_per_ma", "output_swing_v"],
    "bandgap": ["power_w", "temperature_coefficient_ppm_per_c", "line_regulation_mv_per_v", "noise_nv_per_sqrt_hz"],
    "unknown": ["power_w", "area_um2"],
}


def infer_supported_families(task: DesignTask) -> list[str]:
    """Infer supported families from the task definition."""

    if task.circuit_family != "unknown":
        return [task.circuit_family]

    for variable in task.design_space.variables:
        if variable.name == "topology_template_choice" and variable.domain.choices:
            return [str(choice) for choice in variable.domain.choices]
    return ["unknown"]


def resolve_active_family(task: DesignTask, parameter_values: dict[str, float | int | str | bool] | None = None) -> str:
    """Resolve the effective family for a particular state."""

    if task.circuit_family != "unknown":
        return task.circuit_family

    parameter_values = parameter_values or {}
    topology_choice = parameter_values.get("topology_template_choice")
    if topology_choice:
        return str(topology_choice)
    supported = infer_supported_families(task)
    return supported[0] if supported else "unknown"


def collect_supported_metrics(task: DesignTask) -> list[str]:
    """Collect task-aligned and family-aligned metrics for the bundle."""

    metrics = set(task.objective.reporting_metrics)
    metrics.update(constraint.metric for constraint in task.constraints.hard_constraints)
    for family in infer_supported_families(task):
        metrics.update(FAMILY_EXTRA_METRICS.get(family, []))
    return [metric for metric in WORLD_MODEL_METRICS if metric in metrics]


def build_graph_statistics(task: DesignTask) -> GraphStatistics:
    """Create a compact structural summary from the task topology."""

    graph = task.topology.optional_graph_repr
    node_count = len(graph.nodes) if graph else len(task.topology.instances_schema)
    edge_count = len(graph.edges) if graph else len(task.topology.connectivity_schema)
    symmetry_group_count = 1 if task.circuit_family.endswith("ota") else 0
    fanout_estimate = round(max(1.0, edge_count / max(1, node_count)), 4)
    return GraphStatistics(
        node_count=node_count,
        edge_count=edge_count,
        symmetry_group_count=symmetry_group_count,
        fanout_estimate=fanout_estimate,
    )


def build_state_schema() -> StateSchema:
    """Return the fixed third-layer state schema."""

    return StateSchema(
        topology_encoding=EncodingSpec(strategy="template_or_graph", fields=["topology_context", "structural_features"]),
        parameter_encoding=EncodingSpec(strategy="normalized_design_space", fields=["parameter_state"]),
        environment_encoding=EncodingSpec(strategy="task_conditioned_environment", fields=["environment_state", "evaluation_context"]),
        op_encoding=EncodingSpec(strategy="proxy_operating_point_latent", fields=["operating_point_state"]),
        history_encoding=EncodingSpec(strategy="recent_transition_summary", fields=["history_context"]),
        uncertainty_encoding=EncodingSpec(strategy="explicit_field_provenance", fields=["uncertainty_context", "provenance"]),
    )


def build_action_schema() -> ActionSchema:
    """Return the fixed third-layer action schema."""

    return ActionSchema(
        target_selection=EncodingSpec(strategy="structured_target", fields=["action_target"]),
        operator_encoding=EncodingSpec(strategy="operator_token", fields=["action_family", "action_operator"]),
        magnitude_encoding=EncodingSpec(strategy="payload_numeric_or_symbolic", fields=["action_payload"]),
        scope_encoding=EncodingSpec(strategy="expected_effect_axes", fields=["expected_scope"]),
        validity_encoding=EncodingSpec(strategy="design_space_guard", fields=["validity_guard"]),
    )


def build_prediction_heads(supported_metrics: list[str]) -> PredictionHeads:
    """Build the bundle prediction-head registry."""

    return PredictionHeads(
        metric_prediction_head=HeadDefinition(
            head_name="metric_predictor",
            output_schema="MetricsPrediction",
            supported_metrics=supported_metrics,
            notes=["predicts task-aligned terminal metrics and auxiliary proxies"],
        ),
        feasibility_head=HeadDefinition(
            head_name="feasibility_predictor",
            output_schema="FeasibilityPrediction",
            supported_metrics=supported_metrics,
            notes=["predicts overall feasibility and per-constraint-group margins"],
        ),
        transition_head=HeadDefinition(
            head_name="transition_predictor",
            output_schema="TransitionPrediction",
            supported_metrics=supported_metrics,
            notes=["predicts next-state proxies and metric deltas under a design action"],
        ),
        uncertainty_head=HeadDefinition(
            head_name="uncertainty_estimator",
            output_schema="TrustAssessment",
            supported_metrics=supported_metrics,
            notes=["reports epistemic and aleatoric risk for screening and rollout"],
        ),
        trust_head=HeadDefinition(
            head_name="trust_estimator",
            output_schema="TrustAssessment",
            supported_metrics=supported_metrics,
            notes=["maps uncertainty and OOD evidence to service tiers"],
        ),
        simulation_value_head=HeadDefinition(
            head_name="simulation_value_estimator",
            output_schema="SimulationValueEstimate",
            supported_metrics=supported_metrics,
            notes=["estimates the expected value of escalating a candidate to real simulation"],
        ),
    )
