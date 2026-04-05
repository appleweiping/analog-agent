"""Topology resolution for the task formalization layer."""

from __future__ import annotations

from dataclasses import dataclass

from libs.schema.design_task import (
    ConnectivityRule,
    GraphEdge,
    GraphNode,
    GraphRepresentation,
    InstanceSlot,
    TopologyConstraint,
    TopologyPort,
    TopologySpec,
)
from libs.tasking.task_type_resolver import TaskTypeResolution


def _port(name: str, role: str, direction: str) -> TopologyPort:
    return TopologyPort(name=name, role=role, direction=direction)


def _slot(name: str, role: str, device_type: str, tunable_parameters: list[str]) -> InstanceSlot:
    return InstanceSlot(
        name=name,
        role=role,
        device_type=device_type,
        tunable_parameters=tunable_parameters,
    )


def _link(source: str, target: str, relation: str) -> ConnectivityRule:
    return ConnectivityRule(source=source, target=target, relation=relation)


FIXED_TOPOLOGY_LIBRARY = {
    "two_stage_ota": {
        "template_id": "ota2_miller_basic_v1",
        "template_version": "1.0",
        "ports": [
            _port("vinp", "positive_input", "input"),
            _port("vinn", "negative_input", "input"),
            _port("vout", "single_output", "output"),
            _port("vdd", "positive_supply", "supply"),
            _port("vss", "ground_reference", "ground"),
            _port("ibias_ref", "bias_reference", "bias"),
        ],
        "instances": [
            _slot("m_in_pair", "input_pair", "nmos_pair", ["w_in", "l_in"]),
            _slot("m_tail", "tail_current_source", "nmos_current_source", ["w_tail", "l_tail", "ibias"]),
            _slot("m_second_stage", "second_gain_stage", "pmos_common_source", ["ibias"]),
            _slot("c_comp", "compensation_capacitor", "capacitor", ["cc"]),
        ],
        "connectivity": [
            _link("vinp/vinn", "m_in_pair", "drives"),
            _link("m_tail", "m_in_pair", "biases"),
            _link("m_in_pair", "m_second_stage", "feeds"),
            _link("c_comp", "m_second_stage", "stabilizes"),
            _link("m_second_stage", "vout", "drives"),
        ],
        "constraints": [
            TopologyConstraint(name="differential_input_required", description="template preserves differential OTA semantics"),
            TopologyConstraint(name="single_output_required", description="template emits a single-ended output node"),
        ],
    },
    "folded_cascode_ota": {
        "template_id": "folded_cascode_ota_basic_v1",
        "template_version": "1.0",
        "ports": [
            _port("vinp", "positive_input", "input"),
            _port("vinn", "negative_input", "input"),
            _port("vout", "single_output", "output"),
            _port("vdd", "positive_supply", "supply"),
            _port("vss", "ground_reference", "ground"),
            _port("ibias_ref", "bias_reference", "bias"),
        ],
        "instances": [
            _slot("m_in_pair", "input_pair", "pmos_pair", ["w_in", "l_in"]),
            _slot("m_folded_branch", "folded_cascode_branch", "nmos_cascode", ["w_cas", "l_cas"]),
            _slot("m_output_load", "output_load", "pmos_active_load", ["ibias"]),
            _slot("c_comp", "stability_capacitor", "capacitor", ["cc"]),
        ],
        "connectivity": [
            _link("vinp/vinn", "m_in_pair", "drives"),
            _link("m_in_pair", "m_folded_branch", "feeds"),
            _link("m_folded_branch", "m_output_load", "loads"),
            _link("c_comp", "vout", "stabilizes"),
        ],
        "constraints": [
            TopologyConstraint(name="folded_branch_required", description="cascode folding branch must be preserved"),
        ],
    },
    "telescopic_ota": {
        "template_id": "telescopic_ota_basic_v1",
        "template_version": "1.0",
        "ports": [
            _port("vinp", "positive_input", "input"),
            _port("vinn", "negative_input", "input"),
            _port("voutp", "positive_output", "output"),
            _port("voutn", "negative_output", "output"),
            _port("vdd", "positive_supply", "supply"),
            _port("vss", "ground_reference", "ground"),
            _port("vcm_ref", "common_mode_bias", "bias"),
        ],
        "instances": [
            _slot("m_in_pair", "input_pair", "nmos_pair", ["w_in", "l_in"]),
            _slot("m_cascode", "stacked_cascode_devices", "nmos_cascode", ["w_cas", "l_cas"]),
            _slot("m_bias", "bias_network", "bias_block", ["ibias", "vcm"]),
        ],
        "connectivity": [
            _link("vinp/vinn", "m_in_pair", "drives"),
            _link("m_bias", "m_cascode", "biases"),
            _link("m_in_pair", "m_cascode", "feeds"),
            _link("m_cascode", "voutp/voutn", "drives"),
        ],
        "constraints": [
            TopologyConstraint(name="headroom_sensitive_structure", description="stacked telescopic structure requires explicit headroom"),
        ],
    },
    "comparator": {
        "template_id": "comparator_regenerative_v1",
        "template_version": "1.0",
        "ports": [
            _port("vinp", "positive_input", "input"),
            _port("vinn", "negative_input", "input"),
            _port("clk", "clock_input", "input"),
            _port("outp", "positive_output", "output"),
            _port("outn", "negative_output", "output"),
            _port("vdd", "positive_supply", "supply"),
            _port("vss", "ground_reference", "ground"),
        ],
        "instances": [
            _slot("m_in_pair", "input_pair", "nmos_pair", ["w_in", "l_in"]),
            _slot("m_latch", "regenerative_latch", "cross_coupled_pair", ["w_latch", "l_latch"]),
            _slot("m_tail", "tail_device", "nmos_switch", ["ibias"]),
        ],
        "connectivity": [
            _link("vinp/vinn", "m_in_pair", "drives"),
            _link("clk", "m_tail", "gates"),
            _link("m_in_pair", "m_latch", "feeds"),
            _link("m_latch", "outp/outn", "drives"),
        ],
        "constraints": [
            TopologyConstraint(name="clocked_operation_required", description="comparator template requires an explicit clock path"),
        ],
    },
    "ldo": {
        "template_id": "ldo_pmos_compensated_v1",
        "template_version": "1.0",
        "ports": [
            _port("vin", "input_supply", "input"),
            _port("vout", "regulated_output", "output"),
            _port("vref", "reference_input", "input"),
            _port("vdd", "positive_supply", "supply"),
            _port("gnd", "ground_reference", "ground"),
        ],
        "instances": [
            _slot("m_pass", "pass_device", "pmos_pass_transistor", ["w_pass", "l_pass"]),
            _slot("m_err_amp", "error_amplifier", "ota_core", ["w_err", "l_err", "ibias"]),
            _slot("c_comp", "compensation_capacitor", "capacitor", ["c_comp"]),
        ],
        "connectivity": [
            _link("vref", "m_err_amp", "drives"),
            _link("m_err_amp", "m_pass", "controls"),
            _link("m_pass", "vout", "regulates"),
            _link("c_comp", "m_err_amp", "stabilizes"),
        ],
        "constraints": [
            TopologyConstraint(name="pass_device_required", description="LDO task requires an explicit pass device"),
            TopologyConstraint(name="regulation_feedback_required", description="LDO task requires a closed regulation loop"),
        ],
    },
    "bandgap": {
        "template_id": "bandgap_brokaw_core_v1",
        "template_version": "1.0",
        "ports": [
            _port("vdd", "positive_supply", "supply"),
            _port("gnd", "ground_reference", "ground"),
            _port("vref", "reference_output", "output"),
        ],
        "instances": [
            _slot("core_pair", "bandgap_core_pair", "bjt_or_subthreshold_pair", ["area_ratio"]),
            _slot("r_ptat", "ptat_resistor", "resistor", ["r1"]),
            _slot("r_ctat", "ctat_resistor", "resistor", ["r2"]),
            _slot("bias_core", "bias_transistor", "mos_bias_device", ["w_core", "l_core", "ibias"]),
        ],
        "connectivity": [
            _link("core_pair", "r_ptat", "generates_ptat"),
            _link("core_pair", "r_ctat", "balances_ctat"),
            _link("bias_core", "core_pair", "biases"),
            _link("r_ptat/r_ctat", "vref", "sums"),
        ],
        "constraints": [
            TopologyConstraint(name="ptat_ctat_balance_required", description="bandgap core must preserve PTAT and CTAT branches"),
        ],
    },
}

TOPOLOGY_FAMILY_IDS = {
    "two_stage_ota": "ota_family_bundle_v1",
    "folded_cascode_ota": "ota_family_bundle_v1",
    "telescopic_ota": "ota_family_bundle_v1",
    "comparator": "comparator_family_bundle_v1",
    "ldo": "ldo_family_bundle_v1",
    "bandgap": "bandgap_family_bundle_v1",
}


@dataclass(frozen=True)
class TopologyResolution:
    """Resolved topology anchor and provenance."""

    topology: TopologySpec
    derived_fields: list[str]
    assumptions: list[str]


def _graph_from_instances(instances: list[InstanceSlot], connectivity: list[ConnectivityRule]) -> GraphRepresentation:
    nodes = [GraphNode(node_id=slot.name, operation=slot.role, produces=slot.tunable_parameters) for slot in instances]
    edges = [
        GraphEdge(source=rule.source, target=rule.target, condition=rule.relation)
        for rule in connectivity
    ]
    return GraphRepresentation(nodes=nodes, edges=edges)


def resolve_topology(
    circuit_family: str,
    task_type_resolution: TaskTypeResolution,
) -> TopologyResolution:
    """Resolve the formal topology object from family and task-type context."""

    derived_fields = ["topology"]
    assumptions = list(task_type_resolution.assumptions)

    if task_type_resolution.topology_mode == "fixed":
        template = FIXED_TOPOLOGY_LIBRARY[circuit_family]
        topology = TopologySpec(
            topology_mode="fixed",
            template_id=template["template_id"],
            template_version=template["template_version"],
            ports=template["ports"],
            instances_schema=template["instances"],
            connectivity_schema=template["connectivity"],
            topology_constraints=template["constraints"],
            optional_graph_repr=_graph_from_instances(template["instances"], template["connectivity"]),
        )
        return TopologyResolution(topology=topology, derived_fields=derived_fields, assumptions=assumptions)

    if task_type_resolution.topology_mode == "template_family":
        candidate_families = task_type_resolution.candidate_families
        family_id = TOPOLOGY_FAMILY_IDS.get(candidate_families[0], "mixed_family_bundle_v1")
        instances = [
            _slot("candidate_topology", "topology_selector", "template_selector", ["topology_template_choice"]),
            _slot("shared_bias", "shared_bias_anchor", "bias_block", ["ibias"]),
        ]
        connectivity = [
            _link("candidate_topology", "shared_bias", "activates"),
        ]
        constraints = [
            TopologyConstraint(
                name="family_choice_bounded",
                description=f"template choice must remain within {candidate_families}",
            ),
        ]
        assumptions.append("topology family search remains bounded to a finite template bundle")
        topology = TopologySpec(
            topology_mode="template_family",
            template_id=family_id,
            template_version="1.0",
            ports=[],
            instances_schema=instances,
            connectivity_schema=connectivity,
            topology_constraints=constraints,
            optional_graph_repr=_graph_from_instances(instances, connectivity),
        )
        return TopologyResolution(topology=topology, derived_fields=derived_fields, assumptions=assumptions)

    candidate_families = task_type_resolution.candidate_families
    instances = [
        _slot("family_selector", "family_selector", "categorical_switch", ["topology_template_choice"]),
        _slot("core_block", "core_subcircuit", "template_slot", []),
        _slot("bias_block", "bias_subcircuit", "template_slot", ["ibias"]),
        _slot("compensation_block", "compensation_subcircuit", "template_slot", []),
    ]
    connectivity = [
        _link("family_selector", "core_block", "activates"),
        _link("core_block", "bias_block", "requires"),
        _link("core_block", "compensation_block", "may_require"),
    ]
    constraints = [
        TopologyConstraint(
            name="search_space_is_bounded",
            description=f"topology search is limited to the curated candidate families {candidate_families}",
        ),
        TopologyConstraint(
            name="search_space_requires_valid_ports",
            description="all candidate graphs must expose simulator-facing ports before execution",
        ),
    ]
    assumptions.append("open topology search is represented as a bounded curated search space instead of arbitrary graph synthesis")
    topology = TopologySpec(
        topology_mode="search_space",
        template_id="analog_task_search_space_v1",
        template_version="1.0",
        ports=[],
        instances_schema=instances,
        connectivity_schema=connectivity,
        topology_constraints=constraints,
        optional_graph_repr=_graph_from_instances(instances, connectivity),
    )
    return TopologyResolution(topology=topology, derived_fields=derived_fields, assumptions=assumptions)
