"""Netlist construction helpers."""

from __future__ import annotations

from libs.schema.circuit_ir import CircuitIR


def build_netlist(circuit: CircuitIR) -> str:
    """Render a minimal textual representation for downstream simulator work."""
    return f"* netlist for {circuit.name}\n.end"
