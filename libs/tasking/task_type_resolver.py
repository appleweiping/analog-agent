"""Resolve the formal task type for a DesignSpec."""

from __future__ import annotations

from dataclasses import dataclass

from libs.schema.design_spec import CIRCUIT_FAMILIES, DesignSpec
from libs.schema.design_task import TASK_TYPES, TOPOLOGY_MODES

AMPLIFIER_METRICS = {
    "dc_gain_db",
    "gbw_hz",
    "phase_margin_deg",
    "slew_rate_v_per_us",
    "input_referred_noise_nv_per_sqrt_hz",
    "output_swing_v",
    "input_common_mode_v",
}

KNOWN_FAMILIES = tuple(family for family in CIRCUIT_FAMILIES if family != "unknown")
AMPLIFIER_FAMILIES = ("two_stage_ota", "folded_cascode_ota", "telescopic_ota", "comparator")
REFERENCE_FAMILIES = ("bandgap",)
REGULATOR_FAMILIES = ("ldo",)


@dataclass(frozen=True)
class TaskTypeResolution:
    """Deterministic task-type resolution result."""

    task_type: str
    topology_mode: str
    candidate_families: list[str]
    unresolved_dependencies: list[str]
    assumptions: list[str]


def _metric_fingerprint(spec: DesignSpec) -> set[str]:
    return {
        *spec.objectives.maximize,
        *spec.objectives.minimize,
        *spec.hard_constraints.keys(),
    }


def _candidate_families(spec: DesignSpec) -> tuple[list[str], list[str]]:
    if spec.circuit_family != "unknown":
        return [spec.circuit_family], []

    metrics = _metric_fingerprint(spec)
    note_text = " ".join(spec.notes).lower()
    assumptions: list[str] = []

    if metrics & AMPLIFIER_METRICS or "ota" in note_text or "amplifier" in note_text:
        assumptions.append("candidate circuit families narrowed to amplifier templates from metric fingerprint")
        return list(AMPLIFIER_FAMILIES), assumptions
    if "ldo" in note_text or "regulator" in note_text:
        assumptions.append("candidate circuit families narrowed to regulator templates from notes")
        return list(REGULATOR_FAMILIES), assumptions
    if "bandgap" in note_text or "reference" in note_text:
        assumptions.append("candidate circuit families narrowed to reference templates from notes")
        return list(REFERENCE_FAMILIES), assumptions
    assumptions.append("candidate circuit families left broad because the upstream spec does not identify a family")
    return list(KNOWN_FAMILIES), assumptions


def resolve_task_type(spec: DesignSpec, task_type_hint: str | None = None) -> TaskTypeResolution:
    """Resolve the task family and topology mode without mutating user semantics."""

    if task_type_hint is not None and task_type_hint not in TASK_TYPES:
        raise ValueError(f"unsupported task type hint: {task_type_hint}")

    candidate_families, assumptions = _candidate_families(spec)
    unresolved = sorted(set(spec.missing_information))

    if spec.supply_voltage_v is None:
        unresolved.append("supply_voltage_v")
    if spec.circuit_family == "unknown":
        unresolved.append("circuit_family")

    task_type = task_type_hint
    if task_type is None:
        task_type = "topology_sizing" if spec.circuit_family == "unknown" else "sizing"

    if task_type == "sizing":
        topology_mode = "fixed" if spec.circuit_family != "unknown" else "template_family"
    elif task_type == "topology_sizing":
        topology_mode = "template_family" if spec.circuit_family != "unknown" else "search_space"
    else:
        topology_mode = "fixed" if spec.circuit_family != "unknown" else "template_family"

    if topology_mode not in TOPOLOGY_MODES:
        raise ValueError(f"resolved unsupported topology mode: {topology_mode}")

    return TaskTypeResolution(
        task_type=task_type,
        topology_mode=topology_mode,
        candidate_families=list(dict.fromkeys(candidate_families)),
        unresolved_dependencies=list(dict.fromkeys(unresolved)),
        assumptions=assumptions,
    )
