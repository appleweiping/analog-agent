"""Constraint-normalization helpers for the task formalization layer."""

from __future__ import annotations


from libs.schema.design_spec import MetricRange

METRIC_STAGE_MAP = {
    "dc_gain_db": "ac",
    "gbw_hz": "ac",
    "phase_margin_deg": "ac",
    "slew_rate_v_per_us": "tran",
    "power_w": "op",
    "area_um2": "op",
    "noise_nv_per_sqrt_hz": "noise",
    "input_referred_noise_nv_per_sqrt_hz": "noise",
    "output_swing_v": "tran",
    "input_common_mode_v": "op",
    "temperature_coefficient_ppm_per_c": "op",
    "line_regulation_mv_per_v": "tran",
}

METRIC_GROUP_MAP = {
    "dc_gain_db": "gain",
    "gbw_hz": "bandwidth",
    "phase_margin_deg": "stability",
    "slew_rate_v_per_us": "transient",
    "power_w": "power",
    "area_um2": "area",
    "noise_nv_per_sqrt_hz": "noise",
    "input_referred_noise_nv_per_sqrt_hz": "noise",
    "output_swing_v": "swing",
    "input_common_mode_v": "input_range",
    "temperature_coefficient_ppm_per_c": "temperature_stability",
    "line_regulation_mv_per_v": "line_regulation",
}


def stage_for_metric(metric: str) -> str:
    """Return the canonical evaluation stage for a metric."""

    return METRIC_STAGE_MAP.get(metric, "op")


def group_for_metric(metric: str) -> str:
    """Return the semantic group used for reporting and validation."""

    return METRIC_GROUP_MAP.get(metric, "general")


def normalize_metric_range(metric: str, metric_range: MetricRange) -> dict[str, float | str]:
    """Convert a MetricRange into a canonical relation payload."""

    if metric_range.min is not None and metric_range.max is not None:
        return {
            "relation": "in_range",
            "lower_threshold": float(metric_range.min),
            "upper_threshold": float(metric_range.max),
        }
    if metric_range.min is not None:
        return {
            "relation": ">=",
            "threshold": float(metric_range.min),
        }
    if metric_range.max is not None:
        return {
            "relation": "<=",
            "threshold": float(metric_range.max),
        }

    target = float(metric_range.target or 0.0)
    return {
        "relation": "==",
        "threshold": target,
        "tolerance": max(abs(target) * 0.05, 1e-12),
    }
