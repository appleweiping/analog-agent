"""Normalization helpers for interaction-layer parsing."""

from __future__ import annotations

from libs.interaction.parser import ParsedSpecification
from libs.schema.design_spec import CORNER_ORDER, MISSING_INFO_ORDER, OBJECTIVE_METRICS


def normalize_keys(payload: dict[str, object]) -> dict[str, object]:
    """Normalize keys to lower snake-ish names."""
    return {key.strip().lower().replace(" ", "_"): value for key, value in payload.items()}


def normalize_parsed_spec(parsed: ParsedSpecification) -> ParsedSpecification:
    """Canonicalize parser output into a deterministic intermediate form."""

    parsed.objectives_maximize = sorted(set(parsed.objectives_maximize), key=OBJECTIVE_METRICS.index)
    parsed.objectives_minimize = sorted(set(parsed.objectives_minimize), key=OBJECTIVE_METRICS.index)

    corners = sorted(set(parsed.environment["corners"]), key=CORNER_ORDER.index)
    parsed.environment["corners"] = corners
    parsed.environment["temperature_c"] = sorted({float(value) for value in parsed.environment["temperature_c"]})

    if parsed.process_node:
        parsed.process_node = parsed.process_node.replace(" ", "").lower()

    ordered_missing = sorted(set(parsed.missing_information), key=lambda item: (MISSING_INFO_ORDER.index(item), item))
    parsed.missing_information = set(ordered_missing)
    parsed.notes = list(dict.fromkeys(parsed.notes))
    parsed.ambiguities = list(dict.fromkeys(parsed.ambiguities))

    if parsed.supply_voltage_v is not None and parsed.environment["supply_voltage_v"] is None:
        parsed.environment["supply_voltage_v"] = parsed.supply_voltage_v
    if parsed.supply_voltage_v is None and parsed.environment["supply_voltage_v"] is not None:
        parsed.supply_voltage_v = float(parsed.environment["supply_voltage_v"])

    return parsed
