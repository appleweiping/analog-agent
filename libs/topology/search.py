"""Topology search: enumerate and evaluate candidate topologies."""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import TOPOLOGY_CATALOG
from .graph import CircuitTopology


@dataclass
class TopologyCandidate:
    """A ranked topology candidate with feasibility and complexity scores."""

    topology: CircuitTopology
    estimated_feasibility: float  # 0-1, how likely this topology can meet specs
    complexity_score: float  # lower is simpler
    reason: str


class TopologySearcher:
    """Search for suitable topologies given a design specification."""

    def __init__(self) -> None:
        self.catalog = TOPOLOGY_CATALOG

    def search(self, spec: dict) -> list[TopologyCandidate]:
        """Given a spec (gain, bandwidth, power targets), rank topologies.

        Args:
            spec: Dictionary with keys like "dc_gain_db", "gbw_hz", "power_w",
                  each mapping to a dict with "min"/"max"/"target" fields.

        Returns:
            Sorted list of TopologyCandidate (best first).
        """
        candidates: list[TopologyCandidate] = []

        for family_name, factory in self.catalog.items():
            topology = factory()
            feasibility = self._estimate_feasibility(topology, spec)
            complexity = topology.component_count / 10.0  # normalize
            candidates.append(
                TopologyCandidate(
                    topology=topology,
                    estimated_feasibility=feasibility,
                    complexity_score=complexity,
                    reason=self._explain(family_name, spec),
                )
            )

        # Sort by feasibility (descending), then complexity (ascending)
        candidates.sort(key=lambda c: (-c.estimated_feasibility, c.complexity_score))
        return candidates

    def get_topology(self, family: str) -> CircuitTopology | None:
        """Get a specific topology by family name."""
        factory = self.catalog.get(family)
        if factory is None:
            return None
        return factory()

    def register_topology(self, family: str, factory: callable) -> None:
        """Register a new topology factory in the catalog."""
        self.catalog[family] = factory

    def _estimate_feasibility(self, topology: CircuitTopology, spec: dict) -> float:
        """Heuristic feasibility estimate based on topology capabilities."""
        score = 0.5  # baseline

        gain_min = spec.get("dc_gain_db", {}).get("min", 0)
        gbw_min = spec.get("gbw_hz", {}).get("min", 0)
        power_max = spec.get("power_w", {}).get("max", float("inf"))

        if topology.family == "two_stage_ota":
            # Two-stage: good for high gain (>60dB), moderate bandwidth
            if gain_min <= 80:
                score += 0.2
            if gbw_min <= 500e6:
                score += 0.2
            score += 0.1  # simplicity bonus

        elif topology.family == "folded_cascode":
            # Folded cascode: good for high bandwidth, moderate gain
            if gbw_min >= 100e6:
                score += 0.2
            if gain_min <= 60:
                score += 0.2
            # Better power efficiency for high-speed
            if power_max < 5e-3:
                score += 0.1
            else:
                score += 0.05

        return min(1.0, max(0.0, score))

    def _explain(self, family: str, spec: dict) -> str:
        """Generate a human-readable explanation for topology selection."""
        if family == "two_stage_ota":
            return "Two-stage OTA: high gain via cascaded stages, Miller compensation for stability"
        elif family == "folded_cascode":
            return "Folded cascode: single-stage high-speed, good PSRR, wider input range"
        return f"Topology: {family}"
