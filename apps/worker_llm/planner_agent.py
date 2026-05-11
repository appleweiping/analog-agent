"""Planner agent role wrapper."""

from __future__ import annotations

import json

from apps.worker_llm.provider_adapter import complete_role


class PlannerAgent:
    """Generate design actions from parsed specifications."""

    def plan(self, design_spec: dict) -> dict:
        response = complete_role("planner", "generate_design_actions", {"design_spec": design_spec})
        return {
            "agent": "planner",
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
