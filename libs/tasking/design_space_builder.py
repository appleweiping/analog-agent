"""Design-space construction for formalized design tasks."""

from __future__ import annotations

from dataclasses import dataclass

from libs.schema.design_task import (
    ConditionalVariable,
    DesignSpace,
    DesignVariable,
    NormalizationPolicy,
    VariableDomain,
    VariableRelationConstraint,
)
from libs.tasking.task_type_resolver import TaskTypeResolution


@dataclass(frozen=True)
class VariableTemplate:
    """Compact variable template before process-aware instantiation."""

    name: str
    role: str
    kind: str
    dtype: str
    units: str
    scale: str
    lower_multiplier: float | None = None
    upper_multiplier: float | None = None
    absolute_lower: float | int | None = None
    absolute_upper: float | int | None = None
    choices: list[str | float | int] | None = None
    default: float | int | str | bool | None = None
    coupling_group: str | None = None


VARIABLE_LIBRARY: dict[str, list[VariableTemplate]] = {
    "two_stage_ota": [
        VariableTemplate("w_in", "input_pair_width", "continuous", "float", "m", "log", 2.0, 400.0, default=8e-6, coupling_group="input_pair"),
        VariableTemplate("l_in", "input_pair_length", "continuous", "float", "m", "log", 1.0, 20.0, default=1e-6, coupling_group="input_pair"),
        VariableTemplate("w_tail", "tail_device_width", "continuous", "float", "m", "log", 2.0, 320.0, default=6e-6, coupling_group="tail"),
        VariableTemplate("l_tail", "tail_device_length", "continuous", "float", "m", "log", 1.0, 20.0, default=1e-6, coupling_group="tail"),
        VariableTemplate("ibias", "bias_current", "continuous", "float", "A", "log", absolute_lower=1e-6, absolute_upper=2e-3, default=50e-6, coupling_group="bias"),
        VariableTemplate("cc", "compensation_capacitance", "continuous", "float", "F", "log", absolute_lower=0.1e-12, absolute_upper=20e-12, default=1e-12, coupling_group="compensation"),
    ],
    "folded_cascode_ota": [
        VariableTemplate("w_in", "input_pair_width", "continuous", "float", "m", "log", 2.0, 400.0, default=10e-6, coupling_group="input_pair"),
        VariableTemplate("l_in", "input_pair_length", "continuous", "float", "m", "log", 1.0, 20.0, default=1.5e-6, coupling_group="input_pair"),
        VariableTemplate("w_cas", "cascode_width", "continuous", "float", "m", "log", 2.0, 400.0, default=8e-6, coupling_group="cascode"),
        VariableTemplate("l_cas", "cascode_length", "continuous", "float", "m", "log", 1.0, 24.0, default=1.5e-6, coupling_group="cascode"),
        VariableTemplate("ibias", "bias_current", "continuous", "float", "A", "log", absolute_lower=1e-6, absolute_upper=3e-3, default=80e-6, coupling_group="bias"),
        VariableTemplate("cc", "stability_capacitance", "continuous", "float", "F", "log", absolute_lower=0.05e-12, absolute_upper=10e-12, default=0.5e-12, coupling_group="compensation"),
    ],
    "telescopic_ota": [
        VariableTemplate("w_in", "input_pair_width", "continuous", "float", "m", "log", 2.0, 320.0, default=8e-6, coupling_group="input_pair"),
        VariableTemplate("l_in", "input_pair_length", "continuous", "float", "m", "log", 1.0, 20.0, default=1e-6, coupling_group="input_pair"),
        VariableTemplate("w_cas", "cascode_width", "continuous", "float", "m", "log", 2.0, 320.0, default=6e-6, coupling_group="cascode"),
        VariableTemplate("l_cas", "cascode_length", "continuous", "float", "m", "log", 1.0, 24.0, default=1.2e-6, coupling_group="cascode"),
        VariableTemplate("ibias", "bias_current", "continuous", "float", "A", "log", absolute_lower=1e-6, absolute_upper=2e-3, default=60e-6, coupling_group="bias"),
        VariableTemplate("vcm", "common_mode_bias", "continuous", "float", "V", "linear", absolute_lower=0.2, absolute_upper=1.2, default=0.6, coupling_group="bias"),
    ],
    "comparator": [
        VariableTemplate("w_in", "input_pair_width", "continuous", "float", "m", "log", 2.0, 240.0, default=6e-6, coupling_group="input_pair"),
        VariableTemplate("l_in", "input_pair_length", "continuous", "float", "m", "log", 1.0, 12.0, default=0.5e-6, coupling_group="input_pair"),
        VariableTemplate("w_latch", "latch_width", "continuous", "float", "m", "log", 2.0, 240.0, default=5e-6, coupling_group="latch"),
        VariableTemplate("l_latch", "latch_length", "continuous", "float", "m", "log", 1.0, 12.0, default=0.5e-6, coupling_group="latch"),
        VariableTemplate("ibias", "tail_current", "continuous", "float", "A", "log", absolute_lower=1e-6, absolute_upper=1e-3, default=20e-6, coupling_group="bias"),
    ],
    "ldo": [
        VariableTemplate("w_pass", "pass_device_width", "continuous", "float", "m", "log", 20.0, 8000.0, default=200e-6, coupling_group="pass_device"),
        VariableTemplate("l_pass", "pass_device_length", "continuous", "float", "m", "log", 1.0, 8.0, default=0.5e-6, coupling_group="pass_device"),
        VariableTemplate("w_err", "error_amplifier_width", "continuous", "float", "m", "log", 2.0, 320.0, default=8e-6, coupling_group="error_amp"),
        VariableTemplate("l_err", "error_amplifier_length", "continuous", "float", "m", "log", 1.0, 20.0, default=1e-6, coupling_group="error_amp"),
        VariableTemplate("ibias", "quiescent_bias_current", "continuous", "float", "A", "log", absolute_lower=1e-6, absolute_upper=5e-3, default=100e-6, coupling_group="bias"),
        VariableTemplate("c_comp", "compensation_capacitance", "continuous", "float", "F", "log", absolute_lower=0.1e-12, absolute_upper=100e-12, default=5e-12, coupling_group="compensation"),
    ],
    "bandgap": [
        VariableTemplate("area_ratio", "emitter_area_ratio", "integer", "int", "ratio", "linear", absolute_lower=1, absolute_upper=32, default=8, coupling_group="core_ratio"),
        VariableTemplate("r1", "ptat_resistance", "continuous", "float", "ohm", "log", absolute_lower=1e3, absolute_upper=5e5, default=12e3, coupling_group="resistors"),
        VariableTemplate("r2", "ctat_resistance", "continuous", "float", "ohm", "log", absolute_lower=1e3, absolute_upper=5e5, default=36e3, coupling_group="resistors"),
        VariableTemplate("w_core", "core_device_width", "continuous", "float", "m", "log", 2.0, 120.0, default=4e-6, coupling_group="core"),
        VariableTemplate("l_core", "core_device_length", "continuous", "float", "m", "log", 1.0, 20.0, default=1e-6, coupling_group="core"),
        VariableTemplate("ibias", "startup_bias_current", "continuous", "float", "A", "log", absolute_lower=100e-9, absolute_upper=500e-6, default=5e-6, coupling_group="bias"),
    ],
}

GLOBAL_CONSTRAINT_LIBRARY = {
    "two_stage_ota": [
        ("w_tail", ">=", "w_in", "tail device should not undersize the input pair"),
        ("l_tail", "==", "l_in", "tail device shares the channel-length anchor with the input pair"),
    ],
    "folded_cascode_ota": [
        ("l_cas", "==", "l_in", "cascode and input devices share a stable channel-length policy"),
    ],
    "telescopic_ota": [
        ("l_cas", "==", "l_in", "cascode stack reuses the input-stage channel length"),
    ],
    "comparator": [
        ("l_latch", "==", "l_in", "latch and input pair share the same minimum reliable channel length"),
    ],
    "ldo": [
        ("w_pass", ">=", "w_err", "pass device must dominate the small-signal error amplifier size"),
    ],
    "bandgap": [
        ("r2", ">=", "r1", "CTAT resistor should not be smaller than the PTAT resistor in the default template"),
    ],
}

DERIVED_VARIABLE_LIBRARY = {
    "two_stage_ota": ["estimated_area_um2", "gm_over_id_input_estimate"],
    "folded_cascode_ota": ["estimated_area_um2", "cascode_headroom_margin"],
    "telescopic_ota": ["estimated_area_um2", "headroom_margin"],
    "comparator": ["estimated_area_um2", "regeneration_strength_proxy"],
    "ldo": ["estimated_area_um2", "pass_device_overdrive_proxy"],
    "bandgap": ["estimated_area_um2", "ptat_ctat_ratio_proxy"],
}


@dataclass(frozen=True)
class DesignSpaceResolution:
    """Resolved design-space payload plus provenance."""

    design_space: DesignSpace
    derived_fields: list[str]
    assumptions: list[str]


def _process_lmin(process_node: str | None) -> tuple[float, str]:
    if not process_node:
        return 0.18e-6, "expert_rule"
    return float(process_node[:-2]) * 1e-9, "process_rule"


def _domain_for_template(template: VariableTemplate, lmin: float) -> VariableDomain:
    if template.choices:
        return VariableDomain(choices=list(template.choices))
    if template.absolute_lower is not None and template.absolute_upper is not None:
        return VariableDomain(lower=template.absolute_lower, upper=template.absolute_upper)
    return VariableDomain(
        lower=round(float(template.lower_multiplier or 0.0) * lmin, 18),
        upper=round(float(template.upper_multiplier or 0.0) * lmin, 18),
    )


def _build_variable(template: VariableTemplate, process_node: str | None) -> DesignVariable:
    lmin, domain_source = _process_lmin(process_node)
    source = domain_source if template.kind in {"continuous", "integer"} else "expert_rule"
    return DesignVariable(
        name=template.name,
        role=template.role,
        kind=template.kind,
        dtype=template.dtype,
        domain=_domain_for_template(template, lmin),
        scale=template.scale,
        units=template.units,
        default=template.default,
        source=source,
        is_required=True,
        coupling_group=template.coupling_group,
    )


def _global_constraints_for_family(family: str) -> list[VariableRelationConstraint]:
    return [
        VariableRelationConstraint(left=left, relation=relation, right=right, reason=reason)
        for left, relation, right, reason in GLOBAL_CONSTRAINT_LIBRARY.get(family, [])
    ]


def _merge_templates(candidate_families: list[str], process_node: str | None) -> tuple[list[DesignVariable], list[ConditionalVariable]]:
    variables: list[DesignVariable] = []
    seen: set[str] = set()
    activation_map: dict[str, list[str]] = {}

    for family in candidate_families:
        for template in VARIABLE_LIBRARY.get(family, []):
            activation_map.setdefault(template.name, []).append(family)
            if template.name in seen:
                continue
            seen.add(template.name)
            variables.append(_build_variable(template, process_node))

    conditional_variables: list[ConditionalVariable] = []
    if len(candidate_families) > 1:
        for variable in variables:
            active_families = activation_map.get(variable.name, [])
            if len(active_families) != len(candidate_families):
                condition = "topology_template_choice in [" + ", ".join(f"'{family}'" for family in active_families) + "]"
                conditional_variables.append(ConditionalVariable(name=variable.name, active_when=condition))
    return variables, conditional_variables


def build_design_space(
    circuit_family: str,
    process_node: str | None,
    task_type_resolution: TaskTypeResolution,
) -> DesignSpaceResolution:
    """Build the formal design-space object."""

    assumptions: list[str] = []
    derived_fields = ["design_space"]

    if task_type_resolution.topology_mode == "fixed":
        variables = [_build_variable(template, process_node) for template in VARIABLE_LIBRARY[circuit_family]]
        global_constraints = _global_constraints_for_family(circuit_family)
        conditional_variables: list[ConditionalVariable] = []
        derived_variables = DERIVED_VARIABLE_LIBRARY.get(circuit_family, [])
    else:
        topology_choices = task_type_resolution.candidate_families
        structural_variable = DesignVariable(
            name="topology_template_choice",
            role="structural_template_choice",
            kind="categorical",
            dtype="string",
            domain=VariableDomain(choices=topology_choices),
            scale="linear",
            units="category",
            default=topology_choices[0],
            source="expert_rule",
            is_required=True,
            coupling_group="topology",
        )
        merged_variables, conditional_variables = _merge_templates(topology_choices, process_node)
        variables = [structural_variable, *merged_variables]
        global_constraints = []
        derived_variables = ["estimated_area_um2", "structure_complexity_score"]
        assumptions.append("topology search space exposes an explicit categorical structural variable")

    normalization_policy = NormalizationPolicy(
        continuous_strategy="mixed",
        categorical_strategy="one_hot",
        clip_to_domain=True,
    )
    design_space = DesignSpace(
        variables=variables,
        global_constraints=global_constraints,
        derived_variables=derived_variables,
        frozen_variables=[],
        conditional_variables=conditional_variables,
        normalization_policy=normalization_policy,
    )
    return DesignSpaceResolution(
        design_space=design_space,
        derived_fields=derived_fields,
        assumptions=assumptions,
    )
