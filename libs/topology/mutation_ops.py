"""Topology mutation primitives."""

from __future__ import annotations

from libs.schema.circuit_ir import CircuitIR


def rename_circuit(circuit: CircuitIR, suffix: str) -> CircuitIR:
    """Return a renamed circuit copy placeholder."""
    return CircuitIR(name=f"{circuit.name}_{suffix}", nodes=list(circuit.nodes), edges=list(circuit.edges))
