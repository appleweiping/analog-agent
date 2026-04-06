"""Canonical failure categories for analog-agent runs."""

from __future__ import annotations


FAILURE_TAXONOMY = {
    "constraint_violation": "Candidate failed one or more required targets.",
    "operating_region_failure": "Operating-point constraints were violated under truth verification.",
    "stability_failure": "Loop or transient stability margin collapsed near or beyond boundary.",
    "drive_bandwidth_failure": "Drive capability, bandwidth, or load handling was insufficient.",
    "robustness_failure": "Nominal solution did not hold under environment or Monte-Carlo variation.",
    "search_stall": "Planner stopped producing useful new candidates.",
    "budget_exhaustion": "Planning or simulation budget was exhausted before certification.",
    "trust_violation": "World-model trust or calibration assumptions broke in a high-impact region.",
    "calibration_disagreement": "Truth feedback exposed systematic world-model prediction bias.",
    "measurement_failure": "Simulation finished but metric extraction or verification failed.",
    "simulator_failure": "Backend invocation failed or returned invalid payloads.",
}
