"""Deterministic feature projection and proxy prediction utilities."""

from __future__ import annotations

import math

from libs.schema.design_task import ConstraintSpec, DesignTask
from libs.schema.world_model import (
    ConstraintObservation,
    EnvironmentState,
    MetricEstimate,
    OperatingPointState,
    WORLD_MODEL_METRICS,
    WorldState,
)
from libs.tasking.constraint_resolver import group_for_metric
from libs.world_model.design_task_adapter import resolve_active_family

CORNER_FACTOR = {"tt": 1.0, "ss": 0.9, "ff": 1.08, "sf": 0.95, "fs": 1.02}

FAMILY_BASES: dict[str, dict[str, float]] = {
    "two_stage_ota": {
        "dc_gain_db": 68.0,
        "gbw_hz": 1.2e8,
        "phase_margin_deg": 63.0,
        "slew_rate_v_per_us": 8.0,
        "power_w": 7.5e-4,
        "area_um2": 1600.0,
        "noise_nv_per_sqrt_hz": 22.0,
        "output_swing_v": 0.9,
    },
    "folded_cascode_ota": {
        "dc_gain_db": 74.0,
        "gbw_hz": 1.0e8,
        "phase_margin_deg": 70.0,
        "slew_rate_v_per_us": 6.0,
        "power_w": 9.0e-4,
        "area_um2": 2100.0,
        "noise_nv_per_sqrt_hz": 18.0,
        "output_swing_v": 1.0,
    },
    "telescopic_ota": {
        "dc_gain_db": 72.0,
        "gbw_hz": 1.6e8,
        "phase_margin_deg": 58.0,
        "slew_rate_v_per_us": 10.0,
        "power_w": 6.5e-4,
        "area_um2": 1400.0,
        "noise_nv_per_sqrt_hz": 16.0,
        "output_swing_v": 0.8,
        "input_common_mode_v": 0.6,
    },
    "comparator": {
        "delay_ns": 2.4,
        "offset_mv": 5.0,
        "power_w": 3.0e-4,
        "area_um2": 1200.0,
    },
    "ldo": {
        "phase_margin_deg": 64.0,
        "power_w": 1.2e-3,
        "psrr_db": 52.0,
        "line_regulation_mv_per_v": 5.0,
        "load_regulation_mv_per_ma": 3.2,
        "output_swing_v": 1.1,
    },
    "bandgap": {
        "power_w": 1.8e-4,
        "temperature_coefficient_ppm_per_c": 22.0,
        "line_regulation_mv_per_v": 1.4,
        "noise_nv_per_sqrt_hz": 28.0,
        "area_um2": 1800.0,
    },
    "unknown": {"power_w": 1.0e-3, "area_um2": 2000.0},
}


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _numeric_parameter_vectors(state: WorldState) -> tuple[dict[str, float], dict[str, float]]:
    raw: dict[str, float] = {}
    normalized: dict[str, float] = {}
    for parameter in state.parameter_state:
        if isinstance(parameter.value, bool):
            raw[parameter.variable_name] = 1.0 if parameter.value else 0.0
            normalized[parameter.variable_name] = parameter.normalized_value
        elif isinstance(parameter.value, (float, int)):
            raw[parameter.variable_name] = float(parameter.value)
            normalized[parameter.variable_name] = float(parameter.normalized_value)
    return raw, normalized


def _summary_scores(state: WorldState) -> dict[str, float]:
    raw, norm = _numeric_parameter_vectors(state)
    widths = [value for key, value in norm.items() if key.startswith("w_")]
    lengths = [value for key, value in norm.items() if key.startswith("l_")]
    currents = [value for key, value in norm.items() if "ibias" in key]
    compensation = [value for key, value in norm.items() if key.startswith("c_") or key in {"cc", "c_comp"}]
    resistors = [value for key, value in norm.items() if key.startswith("r")]
    ratios = [value for key, value in norm.items() if "ratio" in key]
    pass_devices = [value for key, value in norm.items() if "pass" in key]

    supply = state.environment_state.supply_voltage_v or 1.2
    load_cap = state.environment_state.load_cap_f or 0.0
    output_load = state.environment_state.output_load_ohm or 0.0
    temperature = state.environment_state.temperature_c
    corner_factor = CORNER_FACTOR.get(state.environment_state.corner, 1.0)
    load_penalty = min(1.5, load_cap * 1e12 / 5.0 + (1000.0 / max(output_load, 1000.0) if output_load else 0.0))
    temperature_factor = _clip(1.0 - max(0.0, temperature - 27.0) / 250.0, 0.55, 1.1)

    def mean(values: list[float], default: float = 0.5) -> float:
        return sum(values) / len(values) if values else default

    return {
        "supply": supply,
        "corner_factor": corner_factor,
        "temperature_factor": temperature_factor,
        "load_penalty": load_penalty,
        "width_score": mean(widths),
        "length_score": mean(lengths),
        "bias_score": mean(currents, 0.4),
        "comp_score": mean(compensation, 0.2),
        "resistor_score": mean(resistors, 0.5),
        "ratio_score": mean(ratios, 0.5),
        "pass_score": mean(pass_devices, 0.5),
        "raw_bias_current": max([value for key, value in raw.items() if "ibias" in key], default=50e-6),
        "raw_common_mode": raw.get("vcm", 0.6 * supply),
    }


def project_metrics(task: DesignTask, state: WorldState) -> tuple[dict[str, float], dict[str, float]]:
    """Project a world state into deterministic metric proxies."""

    family = resolve_active_family(task, {parameter.variable_name: parameter.value for parameter in state.parameter_state})
    base = FAMILY_BASES.get(family, FAMILY_BASES["unknown"])
    score = _summary_scores(state)
    metrics: dict[str, float] = {}

    if family.endswith("ota"):
        gain = base.get("dc_gain_db", 65.0) + 18.0 * score["length_score"] + 8.0 * score["width_score"] - 5.5 * score["bias_score"] - 4.0 * score["load_penalty"]
        gbw = base.get("gbw_hz", 1e8) * (0.55 + 0.9 * score["width_score"] + 1.2 * score["bias_score"]) * score["corner_factor"] * score["temperature_factor"] / (0.8 + 0.9 * score["comp_score"] + score["load_penalty"])
        phase_margin = base.get("phase_margin_deg", 60.0) + 15.0 * score["comp_score"] + 9.0 * score["length_score"] - 8.5 * score["bias_score"] - 10.0 * score["load_penalty"]
        slew = base.get("slew_rate_v_per_us", 8.0) * (0.4 + 1.6 * score["bias_score"]) / (0.7 + score["comp_score"] + score["load_penalty"])
        power = base.get("power_w", 8e-4) * (0.65 + 1.2 * score["bias_score"]) * (score["supply"] / 1.2)
        noise = base.get("noise_nv_per_sqrt_hz", 20.0) * (1.0 + max(0.0, 1.0 - score["temperature_factor"])) / (0.8 + score["width_score"] + 0.35 * score["bias_score"])
        area = base.get("area_um2", 1500.0) * (0.65 + score["width_score"] + 0.45 * score["length_score"] + 0.2 * score["comp_score"])
        output_swing = (base.get("output_swing_v", 0.9) * score["supply"]) / max(1.0, 1.0 + 0.18 * score["load_penalty"])
        metrics.update(
            {
                "dc_gain_db": gain,
                "gbw_hz": gbw,
                "phase_margin_deg": phase_margin,
                "slew_rate_v_per_us": slew,
                "power_w": power,
                "noise_nv_per_sqrt_hz": noise,
                "area_um2": area,
                "output_swing_v": output_swing,
            }
        )
        if family == "telescopic_ota":
            metrics["input_common_mode_v"] = 0.25 * score["supply"] + 0.55 * score["raw_common_mode"]

    elif family == "comparator":
        metrics["delay_ns"] = base.get("delay_ns", 2.0) / (0.6 + 0.9 * score["width_score"] + 1.3 * score["bias_score"])
        metrics["offset_mv"] = base.get("offset_mv", 4.0) / (0.8 + score["width_score"] + 0.5 * score["length_score"])
        metrics["power_w"] = base.get("power_w", 2.5e-4) * (0.6 + 1.4 * score["bias_score"]) * (score["supply"] / 1.2)
        metrics["area_um2"] = base.get("area_um2", 1200.0) * (0.6 + score["width_score"] + 0.5 * score["length_score"])

    elif family == "ldo":
        regulation_strength = 0.7 + 1.1 * score["pass_score"] + 0.4 * score["comp_score"]
        metrics["phase_margin_deg"] = base.get("phase_margin_deg", 62.0) + 11.0 * score["comp_score"] + 6.0 * score["bias_score"] - 8.0 * score["load_penalty"]
        metrics["power_w"] = base.get("power_w", 1.2e-3) * (0.8 + 1.2 * score["bias_score"] + 0.4 * score["pass_score"]) * (score["supply"] / 1.8)
        metrics["psrr_db"] = base.get("psrr_db", 48.0) + 8.0 * score["length_score"] + 6.0 * score["comp_score"] - 6.5 * score["load_penalty"]
        metrics["line_regulation_mv_per_v"] = base.get("line_regulation_mv_per_v", 4.0) / regulation_strength
        metrics["load_regulation_mv_per_ma"] = base.get("load_regulation_mv_per_ma", 3.0) / (regulation_strength + 0.35 * score["bias_score"])
        metrics["output_swing_v"] = max(0.1, score["supply"] - 0.08 - 0.2 / max(0.2, regulation_strength))

    elif family == "bandgap":
        balance = 0.5 * score["ratio_score"] + 0.5 * score["resistor_score"]
        metrics["power_w"] = base.get("power_w", 2.0e-4) * (0.7 + 1.1 * score["bias_score"])
        metrics["temperature_coefficient_ppm_per_c"] = base.get("temperature_coefficient_ppm_per_c", 24.0) / (0.75 + balance + 0.25 * score["length_score"])
        metrics["line_regulation_mv_per_v"] = base.get("line_regulation_mv_per_v", 1.5) / (0.8 + score["bias_score"] + 0.25 * score["resistor_score"])
        metrics["noise_nv_per_sqrt_hz"] = base.get("noise_nv_per_sqrt_hz", 30.0) / (0.75 + score["resistor_score"] + 0.2 * score["bias_score"])
        metrics["area_um2"] = base.get("area_um2", 1800.0) * (0.7 + score["resistor_score"] + 0.3 * score["ratio_score"])

    else:
        metrics["power_w"] = base.get("power_w", 1e-3) * (0.8 + score["bias_score"])
        metrics["area_um2"] = base.get("area_um2", 2000.0) * (0.8 + score["width_score"] + 0.2 * score["length_score"])

    auxiliary = {
        "corner_factor": score["corner_factor"],
        "temperature_factor": score["temperature_factor"],
        "load_penalty": score["load_penalty"],
        "width_score": score["width_score"],
        "length_score": score["length_score"],
        "bias_score": score["bias_score"],
        "comp_score": score["comp_score"],
    }
    return metrics, auxiliary


def build_metric_estimates(metric_values: dict[str, float], uncertainty_score: float, confidence: float) -> list[MetricEstimate]:
    """Convert raw metric values into structured estimates."""

    estimates: list[MetricEstimate] = []
    trust_level = "high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low"
    width = max(1e-12, uncertainty_score)
    for metric, value in sorted(metric_values.items()):
        span = abs(value) * max(0.03, 0.18 * width) + (0.1 if abs(value) < 1.0 else 0.0)
        estimates.append(
            MetricEstimate(
                metric=metric,
                value=float(value),
                lower_bound=float(value - span),
                upper_bound=float(value + span),
                uncertainty=round(width, 4),
                trust_level=trust_level,
                source="prediction",
            )
        )
    return estimates


def evaluate_constraints(task: DesignTask, metric_values: dict[str, float], source: str = "prediction") -> list[ConstraintObservation]:
    """Evaluate task constraints against a metric dictionary."""

    observations: list[ConstraintObservation] = []
    for constraint in task.constraints.hard_constraints:
        metric_value = metric_values.get(constraint.metric, 0.0)
        margin = constraint_margin(constraint, metric_value)
        observations.append(
            ConstraintObservation(
                constraint_name=constraint.name,
                constraint_group=group_for_metric(constraint.metric),
                satisfied_probability=_clip(0.5 + margin / (abs(constraint.threshold or 1.0) + 1.0), 0.0, 1.0),
                margin=margin,
                violation_severity=max(0.0, -margin),
                source=source,
            )
        )
    return observations


def constraint_margin(constraint: ConstraintSpec, metric_value: float) -> float:
    """Compute signed margin for one constraint."""

    if constraint.relation == ">=":
        return metric_value - float(constraint.threshold or 0.0)
    if constraint.relation == "<=":
        return float(constraint.threshold or 0.0) - metric_value
    if constraint.relation == "==":
        return float(constraint.tolerance) - abs(metric_value - float(constraint.threshold or 0.0))
    return min(metric_value - float(constraint.lower_threshold or 0.0), float(constraint.upper_threshold or 0.0) - metric_value)


def build_operating_point(task: DesignTask, state: WorldState, metric_values: dict[str, float]) -> OperatingPointState:
    """Construct an operating-point proxy view from parameters and projected metrics."""

    score = _summary_scores(state)
    family = resolve_active_family(task, {parameter.variable_name: parameter.value for parameter in state.parameter_state})
    supply = score["supply"]
    gm_core = 1e-3 * (0.6 + score["width_score"] + 0.8 * score["bias_score"])
    gds_core = 5e-5 * (0.8 + score["width_score"]) / (0.8 + score["length_score"])
    ro_core = 1.0 / max(gds_core, 1e-8)
    id_core = score["raw_bias_current"] * (1.0 if family != "ldo" else 2.0)
    vov = 0.12 + 0.18 * (1.0 - score["length_score"]) + 0.08 * score["bias_score"]
    gm_id = gm_core / max(id_core, 1e-9)
    node_vout = min(supply, metric_values.get("output_swing_v", 0.55 * supply))
    node_vin = metric_values.get("input_common_mode_v", 0.45 * supply)
    return OperatingPointState(
        node_voltage_summary={"vout": node_vout, "vin_cm": node_vin, "v_tail": 0.18 * supply},
        device_region_map={"core": "saturation", "bias": "saturation" if score["bias_score"] > 0.2 else "subthreshold"},
        gm={"core": gm_core},
        gds={"core": gds_core},
        ro={"core": ro_core},
        drain_current_a={"core": id_core},
        overdrive_v={"core": vov},
        gm_over_id={"core": gm_id},
        branch_balance={"main_branch": 1.0 - abs(score["width_score"] - score["bias_score"]) * 0.2},
        pole_zero_summary={"dominant_pole_hz": metric_values.get("gbw_hz", 1e6) / max(1.0, 10 ** (metric_values.get("phase_margin_deg", 60.0) / 60.0))},
        stability_proxy={"phase_margin_deg": metric_values.get("phase_margin_deg", 60.0)},
    )
