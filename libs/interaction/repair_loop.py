"""Repair loop helpers for invalid or incomplete specs."""

from __future__ import annotations

from libs.schema.design_spec import (
    DESIGN_VARIABLES_BY_FAMILY,
    DesignSpec,
)
from libs.interaction.validator import validate_design_spec
from libs.tasking.default_testbench_planner import plan_testbenches


def suggest_repairs(errors: list[str]) -> list[str]:
    """Convert validation errors into simple repair hints."""
    return [f"repair:{error}" for error in errors]


def _recompute_missing_information(spec: DesignSpec) -> list[str]:
    missing = set(spec.missing_information)
    if spec.circuit_family == "unknown":
        missing.add("circuit_family")
    if spec.process_node is None:
        missing.add("process_node")
    if spec.environment.load_cap_f is None and spec.circuit_family in {
        "two_stage_ota",
        "folded_cascode_ota",
        "telescopic_ota",
        "comparator",
        "unknown",
    }:
        missing.add("load_cap_f")
    return list(missing)


def repair_design_spec(spec: DesignSpec) -> DesignSpec:
    """Apply deterministic repair rules to an invalid DesignSpec."""

    payload = spec.model_dump()
    payload.setdefault("objectives", {"maximize": [], "minimize": []})
    payload.setdefault("hard_constraints", {})
    payload.setdefault(
        "environment",
        {
            "temperature_c": [],
            "corners": [],
            "load_cap_f": None,
            "output_load_ohm": None,
            "supply_voltage_v": None,
        },
    )
    payload.setdefault("testbench_plan", ["op"])
    payload.setdefault("design_variables", [])
    payload.setdefault("missing_information", [])
    payload.setdefault("notes", [])
    payload.setdefault("compile_confidence", 0.0)

    for metric_range in payload["hard_constraints"].values():
        if metric_range.get("min") is not None and metric_range.get("max") is not None and metric_range["min"] > metric_range["max"]:
            metric_range["min"], metric_range["max"] = metric_range["max"], metric_range["min"]
        for key in ("min", "max", "target"):
            if key in metric_range and metric_range[key] is not None and key not in {"min", "max", "target"}:
                metric_range[key] = metric_range[key]

    payload["compile_confidence"] = min(max(float(payload["compile_confidence"]), 0.0), 1.0)
    repaired = DesignSpec.model_validate(payload)

    if repaired.process_node:
        repaired.process_node = repaired.process_node.replace(" ", "").lower()
    if repaired.environment.supply_voltage_v is None and repaired.supply_voltage_v is not None:
        repaired.environment.supply_voltage_v = repaired.supply_voltage_v
    if repaired.supply_voltage_v is None and repaired.environment.supply_voltage_v is not None:
        repaired.supply_voltage_v = repaired.environment.supply_voltage_v

    repaired.objectives.maximize = [metric for metric in repaired.objectives.maximize if metric not in repaired.objectives.minimize]

    for metric_range in repaired.hard_constraints.values():
        if metric_range.min is not None and metric_range.max is not None and metric_range.min > metric_range.max:
            metric_range.min, metric_range.max = metric_range.max, metric_range.min

    repaired.testbench_plan = plan_testbenches(repaired)
    repaired.design_variables = DESIGN_VARIABLES_BY_FAMILY[repaired.circuit_family]
    repaired.missing_information = _recompute_missing_information(repaired)
    repaired.compile_confidence = min(max(repaired.compile_confidence, 0.0), 1.0)
    return DesignSpec.model_validate(repaired.model_dump())


def run_repair_loop(spec: DesignSpec, max_rounds: int = 3) -> tuple[DesignSpec, int]:
    """Run deterministic repairs until the spec validates or the budget is exhausted."""

    repaired = spec
    attempts = 0
    while attempts < max_rounds:
        report = validate_design_spec(repaired)
        if report.valid:
            break
        repaired = repair_design_spec(repaired)
        attempts += 1
    return repaired, attempts
