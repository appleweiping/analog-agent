"""Pre-defined topology catalog for known circuit families."""

from __future__ import annotations

from .graph import CircuitEdge, CircuitNode, CircuitTopology


def ota2_topology() -> CircuitTopology:
    """Two-stage OTA topology (behavioral model)."""
    return CircuitTopology(
        name="two_stage_ota_v1",
        family="two_stage_ota",
        nodes=[
            CircuitNode(id="vdd", node_type="supply", voltage=1.2),
            CircuitNode(id="gnd", node_type="ground", voltage=0.0),
            CircuitNode(id="vinp", node_type="input"),
            CircuitNode(id="vinn", node_type="input"),
            CircuitNode(id="vcm", node_type="internal"),
            CircuitNode(id="n1", node_type="internal"),
            CircuitNode(id="vout", node_type="output"),
        ],
        edges=[
            CircuitEdge("vinp", "n1", "transconductor", "Gm1", {"gm1": 1e-3}),
            CircuitEdge("n1", "vcm", "resistor", "R1", {"ro1": 100e3}),
            CircuitEdge("n1", "vcm", "capacitor", "Cp1", {"cp1": 0.1e-12}),
            CircuitEdge("n1", "vout", "transconductor", "Gm2", {"gm2": 5e-3}),
            CircuitEdge("vout", "vcm", "resistor", "R2", {"ro2": 50e3}),
            CircuitEdge("n1", "vout", "capacitor", "Cc", {"cc": 2e-12}),
            CircuitEdge("vout", "gnd", "capacitor", "Cload", {"cload": 2e-12}),
            CircuitEdge("vdd", "gnd", "current_source", "Ibias", {"ibias": 100e-6}),
        ],
    )


def folded_cascode_topology() -> CircuitTopology:
    """Folded cascode OTA topology."""
    return CircuitTopology(
        name="folded_cascode_v1",
        family="folded_cascode",
        nodes=[
            CircuitNode(id="vdd", node_type="supply", voltage=1.2),
            CircuitNode(id="gnd", node_type="ground", voltage=0.0),
            CircuitNode(id="vinp", node_type="input"),
            CircuitNode(id="vinn", node_type="input"),
            CircuitNode(id="n_tail", node_type="internal"),
            CircuitNode(id="n_casc_p", node_type="internal"),
            CircuitNode(id="n_casc_n", node_type="internal"),
            CircuitNode(id="vout", node_type="output"),
        ],
        edges=[
            CircuitEdge("vinp", "n_tail", "mosfet", "M1", {"w": 10e-6, "l": 0.5e-6}),
            CircuitEdge("vinn", "n_tail", "mosfet", "M2", {"w": 10e-6, "l": 0.5e-6}),
            CircuitEdge("vdd", "n_tail", "current_source", "Itail", {"ibias": 200e-6}),
            CircuitEdge("n_casc_p", "vout", "mosfet", "Mcasc_p", {"w": 20e-6, "l": 0.5e-6}),
            CircuitEdge("n_casc_n", "vout", "mosfet", "Mcasc_n", {"w": 20e-6, "l": 0.5e-6}),
            CircuitEdge("vout", "gnd", "capacitor", "Cload", {"cload": 2e-12}),
        ],
    )


TOPOLOGY_CATALOG: dict[str, callable] = {
    "two_stage_ota": ota2_topology,
    "folded_cascode": folded_cascode_topology,
}
