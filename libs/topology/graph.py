"""Circuit topology as a directed graph."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CircuitNode:
    """A node in the circuit topology graph."""

    id: str
    node_type: str  # "input", "output", "internal", "supply", "ground"
    voltage: float | None = None


@dataclass
class CircuitEdge:
    """A directed edge representing a component between two nodes."""

    source: str
    target: str
    component_type: str  # "mosfet", "resistor", "capacitor", "current_source", "voltage_source", "transconductor"
    component_id: str
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass
class CircuitTopology:
    """Complete circuit topology representation as a directed graph."""

    name: str
    family: str  # "two_stage_ota", "folded_cascode", "ldo", "bandgap"
    nodes: list[CircuitNode] = field(default_factory=list)
    edges: list[CircuitEdge] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def component_count(self) -> int:
        return len(self.edges)

    def get_design_variables(self) -> list[str]:
        """Extract tunable parameters from all edges."""
        variables: list[str] = []
        for edge in self.edges:
            for param_name in edge.parameters:
                if param_name not in variables:
                    variables.append(param_name)
        return variables

    def get_node(self, node_id: str) -> CircuitNode | None:
        """Look up a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_edges_from(self, node_id: str) -> list[CircuitEdge]:
        """Get all edges originating from a given node."""
        return [edge for edge in self.edges if edge.source == node_id]

    def get_edges_to(self, node_id: str) -> list[CircuitEdge]:
        """Get all edges terminating at a given node."""
        return [edge for edge in self.edges if edge.target == node_id]

    def get_component(self, component_id: str) -> CircuitEdge | None:
        """Look up a component (edge) by its ID."""
        for edge in self.edges:
            if edge.component_id == component_id:
                return edge
        return None

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        node_ids = {node.id for node in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge {edge.component_id}: source '{edge.source}' not in nodes")
            if edge.target not in node_ids:
                errors.append(f"Edge {edge.component_id}: target '{edge.target}' not in nodes")
        component_ids = [edge.component_id for edge in self.edges]
        seen: set[str] = set()
        for cid in component_ids:
            if cid in seen:
                errors.append(f"Duplicate component_id: {cid}")
            seen.add(cid)
        return errors
