"""Canonical failure categories for analog-agent runs."""

from __future__ import annotations


FAILURE_TAXONOMY = {
    "spec_error": "Input specification is invalid or underspecified.",
    "sim_error": "Simulator invocation or parsing failed.",
    "constraint_violation": "Candidate failed one or more required targets.",
    "search_stall": "Planner stopped producing useful new candidates.",
}
