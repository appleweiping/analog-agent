"""Critic agent placeholder."""

from __future__ import annotations


class CriticAgent:
    """Review outcomes and surface failure explanations."""

    def critique(self, run_summary: dict) -> dict:
        return {"agent": "critic", "status": "stub", "summary": run_summary}
