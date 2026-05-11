"""Topology agent role wrapper."""

from __future__ import annotations

import json

from apps.worker_llm.provider_adapter import complete_role


class TopologyAgent:
    """Propose circuit topologies compatible with the current objective."""

    def propose(self, target_spec: dict) -> dict:
        response = complete_role("topology", "propose_topology", {"target_spec": target_spec})
        return {
            "agent": "topology",
            "status": response.status,
            "provider": response.provider,
            "model": response.model,
            "content": _decode_json_or_text(response.text),
            "error": response.error,
        }


def _decode_json_or_text(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
