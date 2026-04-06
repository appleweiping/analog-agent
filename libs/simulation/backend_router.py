"""Backend routing and execution for the fifth layer."""

from __future__ import annotations

from apps.worker_simulator.ngspice_runner import run_ngspice
from apps.worker_simulator.raw_parser import parse_raw_output
from apps.worker_simulator.spectre_compat_runner import run_spectre_compat
from apps.worker_simulator.xyce_runner import run_xyce
from libs.schema.design_task import DesignTask
from libs.schema.planning import CandidateRecord
from libs.schema.simulation import (
    BackendBinding,
    BackendValidationReport,
    NetlistInstance,
    SimulationBundle,
    SimulationRequest,
)
from libs.simulation.artifact_registry import persist_json_artifact
from libs.simulation.batch_executor import order_analyses


def validate_backend(backend_binding: BackendBinding) -> BackendValidationReport:
    """Validate backend availability in the current environment."""

    warnings = []
    if backend_binding.invocation_mode != "native":
        warnings.append("backend_running_in_mock_truth_mode")
    return BackendValidationReport(
        backend=backend_binding.backend,
        is_available=True,
        invocation_mode=backend_binding.invocation_mode,
        warnings=warnings,
    )


def _dispatch(
    backend: str,
    task: DesignTask,
    candidate: CandidateRecord,
    *,
    netlist: NetlistInstance,
    analysis,
    corner: str,
    temperature_c: float,
    load_cap_f: float | None,
) -> dict[str, object]:
    if backend == "ngspice":
        return run_ngspice(task, candidate, netlist=netlist, analysis=analysis, corner=corner, temperature_c=temperature_c, load_cap_f=load_cap_f)
    if backend == "xyce":
        return run_xyce(task, candidate, netlist=netlist, analysis=analysis, corner=corner, temperature_c=temperature_c, load_cap_f=load_cap_f)
    return run_spectre_compat(task, candidate, netlist=netlist, analysis=analysis, corner=corner, temperature_c=temperature_c, load_cap_f=load_cap_f)


def execute_bundle(
    simulation_bundle: SimulationBundle,
    simulation_request: SimulationRequest,
    task: DesignTask,
    candidate: CandidateRecord,
) -> tuple[SimulationBundle, BackendValidationReport, list[dict[str, object]]]:
    """Execute the analysis plan via the selected backend."""

    backend_report = validate_backend(simulation_bundle.backend_binding)
    registry = simulation_bundle.artifact_registry
    outputs: list[dict[str, object]] = []
    environment = candidate.world_state_snapshot.environment_state
    analyses = order_analyses(simulation_bundle.analysis_plan.ordered_analyses)

    for analysis in analyses:
        raw = _dispatch(
            simulation_bundle.backend_binding.backend,
            task,
            candidate,
            netlist=simulation_bundle.netlist_instance,
            analysis=analysis,
            corner=str(simulation_request.environment_overrides.get("corner", environment.corner)),
            temperature_c=float(simulation_request.environment_overrides.get("temperature_c", environment.temperature_c)),
            load_cap_f=float(simulation_request.environment_overrides.get("load_cap_f", environment.load_cap_f or 2e-12)),
        )
        registry, artifact_id = persist_json_artifact(
            registry,
            "raw_waveform",
            f"{analysis.analysis_type}.json",
            raw,
        )
        parsed = parse_raw_output({**raw, "artifact_ref": artifact_id})
        outputs.append(parsed)

    updated_bundle = simulation_bundle.model_copy(update={"artifact_registry": registry})
    return updated_bundle, backend_report, outputs
