"""Default testbench planning helpers."""

from __future__ import annotations

from libs.schema.design_spec import DesignSpec, TESTBENCH_ORDER

AC_METRICS = {"dc_gain_db", "gbw_hz", "phase_margin_deg"}
TRAN_METRICS = {"slew_rate_v_per_us"}
NOISE_METRICS = {"input_referred_noise_nv_per_sqrt_hz", "noise_nv_per_sqrt_hz"}


def _ordered(plan: set[str]) -> list[str]:
    return [step for step in TESTBENCH_ORDER if step in plan]


def plan_testbenches(spec: DesignSpec) -> list[str]:
    """Derive a deterministic testbench plan from the current DesignSpec."""

    metrics = set(spec.hard_constraints)
    metrics.update(spec.objectives.maximize)
    metrics.update(spec.objectives.minimize)

    plan = {"op"}
    if metrics & AC_METRICS:
        plan.add("ac")
    if metrics & TRAN_METRICS:
        plan.add("tran")
    if metrics & NOISE_METRICS:
        plan.add("noise")
    return _ordered(plan)
