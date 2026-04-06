"""Deterministic ground-truth proxy used by fifth-layer backends."""

from __future__ import annotations

from math import log10, sqrt

from libs.schema.design_task import DesignTask
from libs.schema.planning import CandidateRecord
from libs.schema.simulation import AnalysisStatement, NetlistInstance


def _param_map(candidate: CandidateRecord) -> dict[str, float]:
    values: dict[str, float] = {}
    for parameter in candidate.world_state_snapshot.parameter_state:
        value = parameter.value
        if isinstance(value, bool):
            values[parameter.variable_name] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            values[parameter.variable_name] = float(value)
    return values


def _value(values: dict[str, float], *names: str, default: float) -> float:
    for name in names:
        if name in values:
            return float(values[name])
    return default


def _family_factor(task: DesignTask) -> float:
    factors = {
        "two_stage_ota": 1.05,
        "folded_cascode_ota": 0.98,
        "telescopic_ota": 1.08,
        "comparator": 0.92,
        "ldo": 0.88,
        "bandgap": 0.84,
        "unknown": 0.75,
    }
    return factors.get(task.circuit_family, 0.8)


def _corner_factor(corner: str) -> float:
    return {"tt": 1.0, "ff": 1.08, "ss": 0.9, "sf": 0.94, "fs": 1.02}.get(corner, 0.96)


def _temperature_factor(temperature_c: float) -> float:
    return max(0.75, 1.0 - abs(temperature_c - 27.0) / 280.0)


def _load_factor(load_cap_f: float | None) -> float:
    load = load_cap_f or 2e-12
    return max(0.65, min(1.25, 2.4e-12 / max(load, 1e-15)))


def compute_truth_metrics(
    task: DesignTask,
    candidate: CandidateRecord,
    *,
    analysis_type: str,
    corner: str,
    temperature_c: float,
    load_cap_f: float | None,
) -> dict[str, float]:
    """Return deterministic per-analysis truth metrics."""

    values = _param_map(candidate)
    supply = candidate.world_state_snapshot.environment_state.supply_voltage_v or 1.2
    w_in = _value(values, "w_in", "w_input", default=8e-6)
    l_in = _value(values, "l_in", "l_input", default=1e-6)
    w_tail = _value(values, "w_tail", default=max(w_in * 0.75, 3e-6))
    l_tail = _value(values, "l_tail", default=l_in)
    ibias = _value(values, "ibias", "bias_current", default=5e-5)
    cc = _value(values, "cc", "comp_cap", default=1e-12)

    family = _family_factor(task)
    corner_scale = _corner_factor(corner)
    temp_scale = _temperature_factor(temperature_c)
    load_scale = _load_factor(load_cap_f)

    gm_factor = max(0.2, (w_in / max(l_in, 1e-9)) / 8.0)
    tail_factor = max(0.2, (w_tail / max(l_tail, 1e-9)) / 6.0)
    bias_factor = max(0.2, ibias / 5e-5)
    comp_factor = max(0.35, 1e-12 / max(cc, 1e-15))
    aggregate = family * corner_scale * temp_scale * load_scale

    dc_gain_db = 48.0 + 9.0 * log10(max(gm_factor * tail_factor * family, 1e-6)) - 2.0 * log10(max(bias_factor, 1e-6))
    gbw_hz = 1.4e8 * gm_factor * sqrt(bias_factor) * comp_factor * aggregate
    phase_margin_deg = 57.0 + 11.0 * log10(max(cc / max(load_cap_f or 2e-12, 1e-15), 1e-6)) + 2.5 * family - 3.0 * max(gm_factor - 3.0, 0.0)
    slew_rate_v_per_us = (ibias / max(load_cap_f or 2e-12, 1e-15)) / 1e6
    power_w = supply * ibias * (1.25 if task.circuit_family.endswith("ota") else 1.05)
    area_um2 = (w_in * l_in * 4.0 + w_tail * l_tail * 2.0 + cc * 7.5e6) * 1e12
    noise = 12.0 / max(sqrt(max(gm_factor * bias_factor, 1e-6)), 1e-3)
    output_swing_v = max(0.12, min(supply * 0.92, supply - 0.16 - 0.03 * max(gm_factor - 2.0, 0.0)))
    psrr_db = 56.0 + 6.0 * log10(max(cc / 5e-13, 1e-6)) + 2.0 * family
    line_regulation_mv_per_v = max(0.5, 7.5 / max(family * tail_factor, 0.2))
    load_regulation_mv_per_ma = max(0.8, 8.2 / max(gm_factor * family, 0.25))
    offset_mv = 2.5 / max(sqrt(max(w_in / 8e-6, 1e-6)), 1e-3)
    delay_ns = max(0.5, 18.0 / max(gbw_hz / 1e8, 0.1))
    temperature_coefficient_ppm_per_c = 14.0 / max(family * temp_scale, 0.2)

    base = {
        "dc_gain_db": round(dc_gain_db, 6),
        "gbw_hz": round(gbw_hz, 6),
        "phase_margin_deg": round(phase_margin_deg, 6),
        "slew_rate_v_per_us": round(slew_rate_v_per_us, 6),
        "power_w": round(power_w, 12),
        "area_um2": round(area_um2, 6),
        "noise_nv_per_sqrt_hz": round(noise, 6),
        "input_referred_noise_nv_per_sqrt_hz": round(noise * 1.08, 6),
        "output_swing_v": round(output_swing_v, 6),
        "input_common_mode_v": round(supply * 0.5, 6),
        "psrr_db": round(psrr_db, 6),
        "line_regulation_mv_per_v": round(line_regulation_mv_per_v, 6),
        "load_regulation_mv_per_ma": round(load_regulation_mv_per_ma, 6),
        "offset_mv": round(offset_mv, 6),
        "delay_ns": round(delay_ns, 6),
        "temperature_coefficient_ppm_per_c": round(temperature_coefficient_ppm_per_c, 6),
    }

    by_analysis = {
        "op": ["power_w", "area_um2", "output_swing_v", "input_common_mode_v"],
        "ac": ["dc_gain_db", "gbw_hz", "phase_margin_deg", "psrr_db"],
        "tran": ["slew_rate_v_per_us", "delay_ns", "output_swing_v"],
        "noise": ["noise_nv_per_sqrt_hz", "input_referred_noise_nv_per_sqrt_hz"],
        "pvt_sweep": ["gbw_hz", "phase_margin_deg", "power_w"],
        "load_sweep": ["gbw_hz", "phase_margin_deg", "load_regulation_mv_per_ma"],
        "temperature_sweep": ["offset_mv", "temperature_coefficient_ppm_per_c", "line_regulation_mv_per_v"],
        "monte_carlo": ["offset_mv", "gbw_hz", "phase_margin_deg"],
    }
    selected = by_analysis.get(analysis_type, list(base))
    return {metric: base[metric] for metric in selected}


def analysis_payload(
    task: DesignTask,
    candidate: CandidateRecord,
    *,
    netlist: NetlistInstance,
    analysis: AnalysisStatement,
    corner: str,
    temperature_c: float,
    load_cap_f: float | None,
) -> dict[str, object]:
    """Build a backend-neutral payload for one executed analysis."""

    metrics = compute_truth_metrics(
        task,
        candidate,
        analysis_type=analysis.analysis_type,
        corner=corner,
        temperature_c=temperature_c,
        load_cap_f=load_cap_f,
    )
    regions = {
        instance.name: ("saturation" if metrics.get("phase_margin_deg", 70.0) >= 48.0 else "linear")
        for instance in task.topology.instances_schema
    }
    return {
        "netlist_id": netlist.netlist_id,
        "analysis_type": analysis.analysis_type,
        "corner": corner,
        "temperature_c": temperature_c,
        "load_cap_f": load_cap_f,
        "metrics": metrics,
        "op_diagnostics": {
            "device_region_map": regions,
            "stability_proxy": {
                "phase_margin_deg": metrics.get("phase_margin_deg", 0.0),
                "dc_gain_db": metrics.get("dc_gain_db", 0.0),
            },
        },
    }
