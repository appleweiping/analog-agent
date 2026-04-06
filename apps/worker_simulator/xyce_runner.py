"""Xyce binding for the fifth-layer simulation service."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.planning import CandidateRecord
from libs.schema.simulation import AnalysisStatement, NetlistInstance
from libs.simulation.truth_model import analysis_payload


def run_xyce(
    task: DesignTask,
    candidate: CandidateRecord,
    *,
    netlist: NetlistInstance,
    analysis: AnalysisStatement,
    corner: str,
    temperature_c: float,
    load_cap_f: float | None,
) -> dict[str, object]:
    """Execute one deterministic Xyce-compatible analysis payload."""

    payload = analysis_payload(
        task,
        candidate,
        netlist=netlist,
        analysis=analysis,
        corner=corner,
        temperature_c=temperature_c,
        load_cap_f=load_cap_f,
    )
    payload["backend"] = "xyce"
    payload["runtime_ms"] = 20 + analysis.order * 6
    payload["status"] = "ok"
    return payload
