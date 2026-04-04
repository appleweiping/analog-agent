"""Graph encoding helpers for topology-aware models."""

from __future__ import annotations

from libs.schema.circuit_ir import CircuitIR


def encode_graph(circuit: CircuitIR) -> dict[str, object]:
    """Produce a lightweight graph summary."""
    return {
        "name": circuit.name,
        "node_count": len(circuit.nodes),
        "edge_count": len(circuit.edges),
    }
