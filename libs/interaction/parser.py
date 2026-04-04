"""Deterministic natural-language parser for interaction-layer requests."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

_COMPARATOR_ALIASES = {
    ">=": "min",
    ">": "min",
    "大于": "min",
    "大於": "min",
    "大于等于": "min",
    "大於等於": "min",
    "不少于": "min",
    "不少於": "min",
    "不小于": "min",
    "不小於": "min",
    "至少": "min",
    "以上": "min",
    "<=": "max",
    "<": "max",
    "小于": "max",
    "小於": "max",
    "小于等于": "max",
    "小於等於": "max",
    "不超过": "max",
    "不超過": "max",
    "低于": "max",
    "低於": "max",
    "以下": "max",
    "=": "target",
    "约": "target",
    "約": "target",
    "约等于": "target",
    "約等於": "target",
}

_CIRCUIT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("folded_cascode_ota", (r"folded[\s\-]*cascode", r"折叠共栅", r"折疊共柵")),
    ("telescopic_ota", (r"telescopic", r"望远镜", r"望遠鏡")),
    ("two_stage_ota", (r"two[\s\-]*stage", r"\b2[\s\-]*stage\b", r"两级", r"兩級")),
    ("comparator", (r"comparator", r"比较器", r"比較器")),
    ("ldo", (r"\bldo\b", r"低压差", r"低壓差")),
    ("bandgap", (r"band[\s\-]*gap", r"带隙", r"帶隙")),
]

_GENERIC_OTA_PATTERNS = (r"\bota\b", r"运算跨导放大器", r"運算跨導放大器")
_GENERIC_AMPLIFIER_PATTERNS = (r"amplifier", r"放大器")

_OBJECTIVE_HINTS: list[tuple[tuple[str, ...], str, str]] = [
    ((r"高速", r"high[\s\-]*speed", r"fast"), "maximize", "gbw_hz"),
    ((r"低功耗", r"low[\s\-]*power"), "minimize", "power_w"),
    ((r"高增益", r"high[\s\-]*gain"), "maximize", "dc_gain_db"),
    ((r"低噪声", r"低噪聲", r"low[\s\-]*noise"), "minimize", "noise_nv_per_sqrt_hz"),
    ((r"大相位裕度", r"高相位裕度", r"stable", r"stability"), "maximize", "phase_margin_deg"),
]


@dataclass(slots=True)
class ParsedSpecification:
    """Intermediate parse result before normalization and schema construction."""

    raw_text: str
    circuit_family: str | None = None
    process_node: str | None = None
    supply_voltage_v: float | None = None
    objectives_maximize: list[str] = field(default_factory=list)
    objectives_minimize: list[str] = field(default_factory=list)
    hard_constraints: dict[str, dict[str, float | None | str]] = field(default_factory=dict)
    environment: dict[str, object] = field(
        default_factory=lambda: {
            "temperature_c": [],
            "corners": [],
            "load_cap_f": None,
            "output_load_ohm": None,
            "supply_voltage_v": None,
        }
    )
    missing_information: set[str] = field(default_factory=set)
    ambiguities: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    field_sources: dict[str, str] = field(default_factory=dict)
    parser_errors: list[str] = field(default_factory=list)


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _normalize_comparator(raw: str | None) -> str:
    if raw is None:
        return "target"
    return _COMPARATOR_ALIASES.get(raw.strip(), "target")


def _canonicalize_frequency(value: float, unit: str) -> float:
    factors = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
    return value * factors[unit.lower()]


def _canonicalize_power(value: float, unit: str) -> float:
    factors = {"w": 1.0, "mw": 1e-3, "uw": 1e-6, "μw": 1e-6, "nw": 1e-9}
    return value * factors[unit.lower()]


def _canonicalize_voltage(value: float, unit: str) -> float:
    factors = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "μv": 1e-6}
    return value * factors[unit.lower()]


def _canonicalize_capacitance(value: float, unit: str) -> float:
    factors = {"f": 1.0, "mf": 1e-3, "uf": 1e-6, "μf": 1e-6, "nf": 1e-9, "pf": 1e-12}
    return value * factors[unit.lower()]


def _canonicalize_resistance(value: float, unit: str) -> float:
    factors = {"ohm": 1.0, "kohm": 1e3, "mohm": 1e6}
    return value * factors[unit.lower()]


def _canonicalize_slew_rate(value: float, unit: str) -> float:
    factors = {"v/us": 1.0, "v/μs": 1.0, "v/ns": 1e3, "v/ms": 1e-3}
    return value * factors[unit.lower()]


def _canonicalize_noise(value: float, unit: str) -> float:
    factors = {"nv/sqrthz": 1.0, "uv/sqrthz": 1e3, "μv/sqrthz": 1e3}
    return value * factors[unit.lower()]


def _update_constraint(
    constraints: dict[str, dict[str, float | None | str]],
    metric: str,
    comparator: str,
    value: float,
) -> None:
    constraint = constraints.setdefault(metric, {"min": None, "max": None, "target": None, "priority": "hard"})
    if comparator == "min":
        existing = constraint["min"]
        constraint["min"] = value if existing is None else max(float(existing), value)
    elif comparator == "max":
        existing = constraint["max"]
        constraint["max"] = value if existing is None else min(float(existing), value)
    else:
        constraint["target"] = value


def _parse_metric_constraints(text: str, parsed: ParsedSpecification) -> None:
    metric_specs = [
        (
            "gbw_hz",
            r"(?:gbw|ugb|unity[\s\-]*gain[\s\-]*bandwidth|bandwidth|增益带宽积|增益帶寬積|带宽|帶寬)",
            r"(hz|khz|mhz|ghz)",
            _canonicalize_frequency,
        ),
        (
            "phase_margin_deg",
            r"(?:phase[\s\-]*margin|\bpm\b|相位裕度)",
            r"(?:deg(?:ree)?s?|°)",
            lambda value, _unit: value,
        ),
        (
            "dc_gain_db",
            r"(?:dc[\s\-]*gain|gain|增益)",
            r"(db)",
            lambda value, _unit: value,
        ),
        (
            "power_w",
            r"(?:power|功耗)",
            r"(w|mw|uw|μw|nw)",
            _canonicalize_power,
        ),
        (
            "slew_rate_v_per_us",
            r"(?:slew[\s\-]*rate|\bsr\b|压摆率|壓擺率|转换速率|轉換速率)",
            r"(v/us|v/μs|v/ns|v/ms)",
            _canonicalize_slew_rate,
        ),
        (
            "input_referred_noise_nv_per_sqrt_hz",
            r"(?:input[\s\-]*referred[\s\-]*noise|noise|输入参考噪声|輸入參考噪聲|噪声|噪聲)",
            r"(nv/sqrthz|uv/sqrthz|μv/sqrthz)",
            _canonicalize_noise,
        ),
        (
            "output_swing_v",
            r"(?:output[\s\-]*swing|輸出擺幅|输出摆幅)",
            r"(v|mv)",
            _canonicalize_voltage,
        ),
        (
            "input_common_mode_v",
            r"(?:input[\s\-]*common[\s\-]*mode|common[\s\-]*mode|输入共模|輸入共模)",
            r"(v|mv)",
            _canonicalize_voltage,
        ),
    ]

    comparator_pattern = (
        r"(?P<cmp>>=|<=|>|<|=|大于等于|大於等於|大于|大於|小于等于|小於等於|小于|小於|"
        r"不少于|不少於|不小于|不小於|至少|以上|以下|不超过|不超過|低于|低於|约等于|約等於|约|約)?"
    )

    for metric, alias_pattern, unit_pattern, converter in metric_specs:
        pattern = re.compile(
            rf"{alias_pattern}\s*(?:[:=]|为|為)?\s*{comparator_pattern}\s*"
            rf"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>{unit_pattern})",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            value = converter(float(match.group("value")), match.group("unit"))
            if metric not in {"output_swing_v", "input_common_mode_v"} and value < 0:
                parsed.parser_errors.append(f"{metric} cannot be negative")
                parsed.notes.append(f"ignored invalid negative constraint for {metric}")
                continue
            _update_constraint(parsed.hard_constraints, metric, _normalize_comparator(match.group("cmp")), value)


def _parse_environment(text: str, parsed: ParsedSpecification) -> None:
    process_match = re.search(r"(?P<node>\d+)\s*nm", text, flags=re.IGNORECASE)
    if process_match:
        parsed.process_node = f"{process_match.group('node')}nm"
        parsed.field_sources["process_node"] = "user_provided"

    supply_patterns = [
        re.compile(
            r"(?:supply|vdd|供电|供電)\s*(?:voltage)?\s*(?:[:=]|为|為)?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>v|mv)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>v|mv)\s*(?:supply|vdd|供电|供電)",
            flags=re.IGNORECASE,
        ),
    ]
    for pattern in supply_patterns:
        supply_match = pattern.search(text)
        if supply_match:
            supply = _canonicalize_voltage(float(supply_match.group("value")), supply_match.group("unit"))
            parsed.supply_voltage_v = supply
            parsed.environment["supply_voltage_v"] = supply
            parsed.field_sources["supply_voltage_v"] = "user_provided"
            break

    for corner in ("tt", "ss", "ff", "sf", "fs"):
        if re.search(rf"\b{corner}\b", text, flags=re.IGNORECASE):
            parsed.environment["corners"].append(corner)
    if parsed.environment["corners"]:
        parsed.field_sources["environment"] = "user_provided"

    for temp_match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*°?\s*c\b", text, flags=re.IGNORECASE):
        parsed.environment["temperature_c"].append(float(temp_match.group(1)))
    if parsed.environment["temperature_c"]:
        parsed.field_sources["environment"] = "user_provided"

    load_cap_match = re.search(
        r"(?:load[\s\-]*cap(?:acitance)?|cload|負載電容|负载电容)\s*(?:[:=]|为|為)?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>f|mf|uf|μf|nf|pf)",
        text,
        flags=re.IGNORECASE,
    )
    if load_cap_match:
        parsed.environment["load_cap_f"] = _canonicalize_capacitance(
            float(load_cap_match.group("value")),
            load_cap_match.group("unit"),
        )
        parsed.field_sources["environment"] = "user_provided"

    output_load_match = re.search(
        r"(?:output[\s\-]*load|load[\s\-]*resistance|負載電阻|负载电阻)\s*(?:[:=]|为|為)?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>ohm|kohm|mohm)",
        text,
        flags=re.IGNORECASE,
    )
    if output_load_match:
        parsed.environment["output_load_ohm"] = _canonicalize_resistance(
            float(output_load_match.group("value")),
            output_load_match.group("unit"),
        )
        parsed.field_sources["environment"] = "user_provided"


def _parse_objective_hints(text: str, parsed: ParsedSpecification) -> None:
    for patterns, direction, metric in _OBJECTIVE_HINTS:
        if _contains_any(text, patterns):
            if direction == "maximize":
                parsed.objectives_maximize.append(metric)
            else:
                parsed.objectives_minimize.append(metric)
            parsed.notes.append(f"inferred objective {direction}:{metric} from qualitative wording")


def _parse_circuit_family(text: str, parsed: ParsedSpecification) -> None:
    for circuit_family, patterns in _CIRCUIT_PATTERNS:
        if _contains_any(text, patterns):
            parsed.circuit_family = circuit_family
            parsed.field_sources["circuit_family"] = "user_provided"
            return

    if _contains_any(text, _GENERIC_OTA_PATTERNS):
        parsed.circuit_family = "unknown"
        parsed.missing_information.add("circuit_family")
        parsed.ambiguities.append("generic OTA request does not specify a topology family")
        parsed.notes.append("parsed a generic OTA request without a concrete topology")
        parsed.field_sources["circuit_family"] = "system_inferred"
        return

    if _contains_any(text, _GENERIC_AMPLIFIER_PATTERNS):
        parsed.circuit_family = "unknown"
        parsed.missing_information.add("circuit_family")
        parsed.ambiguities.append("amplifier request is missing a supported circuit family")
        parsed.field_sources["circuit_family"] = "system_inferred"


def parse_specification(text: str) -> ParsedSpecification:
    """Parse natural-language design requirements into a structured intermediate form."""

    parsed = ParsedSpecification(raw_text=text)
    lowered = text.lower()

    _parse_circuit_family(lowered, parsed)
    _parse_environment(lowered, parsed)
    _parse_metric_constraints(lowered, parsed)
    _parse_objective_hints(lowered, parsed)

    if not parsed.circuit_family:
        parsed.circuit_family = "unknown"
        parsed.missing_information.add("circuit_family")
        parsed.field_sources["circuit_family"] = "system_inferred"

    if not parsed.process_node:
        parsed.missing_information.add("process_node")

    return parsed
