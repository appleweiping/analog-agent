"""Spectre-compatible binding for the fifth-layer simulation service."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.planning import CandidateRecord
from libs.schema.simulation import AnalysisStatement, NetlistInstance
from libs.simulation.truth_model import analysis_payload


def run_spectre_compat(
    task: DesignTask,
    candidate: CandidateRecord,
    *,
    netlist: NetlistInstance,
    analysis: AnalysisStatement,
    corner: str,
    temperature_c: float,
    load_cap_f: float | None,
) -> dict[str, object]:
    """Execute one deterministic Spectre-compatible analysis payload."""

    payload = analysis_payload(
        task,
        candidate,
        netlist=netlist,
        analysis=analysis,
        corner=corner,
        temperature_c=temperature_c,
        load_cap_f=load_cap_f,
    )
    payload["backend"] = "spectre_compat"
    payload["runtime_ms"] = 24 + analysis.order * 8
    payload["status"] = "ok"
    return payload
