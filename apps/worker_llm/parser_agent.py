"""Parser agent role wrapper."""

from __future__ import annotations

import json

from apps.worker_llm.provider_adapter import complete_role


class ParserAgent:
    """Parse human specifications into structured design intents."""

    def parse(self, prompt: str) -> dict:
        response = complete_role("parser", "parse_design_specification", {"prompt": prompt})
        return {
            "agent": "parser",
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
