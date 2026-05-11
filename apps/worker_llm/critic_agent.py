"""Critic agent role wrapper."""

from __future__ import annotations

import json

from apps.worker_llm.provider_adapter import complete_role


class CriticAgent:
    """Review outcomes and surface failure explanations."""

    def critique(self, run_summary: dict) -> dict:
        response = complete_role("critic", "critique_run_summary", {"run_summary": run_summary})
        return {
            "agent": "critic",
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
