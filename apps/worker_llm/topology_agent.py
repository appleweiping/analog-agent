"""Topology agent placeholder."""

from __future__ import annotations


class TopologyAgent:
    """Propose circuit topologies compatible with the current objective."""

    def propose(self, target_spec: dict) -> dict:
        return {"agent": "topology", "status": "stub", "target_spec": target_spec}
