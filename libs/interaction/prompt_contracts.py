"""Prompt contract templates for LLM worker coordination."""

from __future__ import annotations


PROMPT_CONTRACTS = {
    "parser": "Return normalized JSON-like fields for objectives and constraints.",
    "planner": "Return candidate actions and a concise rationale.",
    "critic": "Return failure modes, missing evidence, and next-step priorities.",
}
