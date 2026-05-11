"""Reflection agent role wrapper."""

from __future__ import annotations

import json

from apps.worker_llm.provider_adapter import complete_role


class ReflectionAgent:
    """Summarize lessons from completed or failed runs."""

    def reflect(self, trajectory: dict) -> dict:
        response = complete_role("reflection", "summarize_design_trajectory", {"trajectory": trajectory})
        return {
            "agent": "reflection",
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
