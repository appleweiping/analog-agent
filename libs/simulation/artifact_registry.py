"""Artifact persistence and registry helpers for the fifth layer."""

from __future__ import annotations

import json
from pathlib import Path

from libs.schema.simulation import ArtifactExecutionContext, ArtifactRecord, ArtifactRegistry, SimulationProvenance, ValidationStatus
from libs.utils.hashing import stable_hash


ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / ".artifacts" / "simulation"


def build_artifact_execution_context(
    *,
    simulation_provenance: SimulationProvenance | None,
    validation_status: ValidationStatus | None,
    backend_error_type: str | None = None,
    execution_runtime_sec: float | None = None,
    replay_hint: str | None = None,
) -> ArtifactExecutionContext | None:
    """Build replay-oriented execution context for one artifact."""

    if simulation_provenance is None and validation_status is None and not replay_hint and backend_error_type is None:
        return None
    paper_mode = simulation_provenance.paper_mode if simulation_provenance is not None else False
    paper_safe = simulation_provenance.paper_safe if simulation_provenance is not None else False
    replayable = bool(
        simulation_provenance is not None
        and simulation_provenance.invocation_mode == "native"
        and simulation_provenance.resolved_simulator_binary
    )
    return ArtifactExecutionContext(
        paper_mode=paper_mode,
        paper_safe=paper_safe,
        replayable=replayable,
        resolved_simulator_binary=simulation_provenance.resolved_simulator_binary if simulation_provenance is not None else None,
        backend_error_type=backend_error_type,
        execution_runtime_sec=round(float(execution_runtime_sec), 6) if execution_runtime_sec is not None else None,
        replay_hint=replay_hint,
    )


def initialize_artifact_registry(simulation_id: str) -> ArtifactRegistry:
    """Create a writable artifact registry root for one simulation."""

    run_directory = ARTIFACT_ROOT / simulation_id
    run_directory.mkdir(parents=True, exist_ok=True)
    return ArtifactRegistry(run_directory=str(run_directory), records=[])


def persist_text_artifact(
    registry: ArtifactRegistry,
    artifact_type: str,
    name: str,
    content: str,
    *,
    simulation_provenance: SimulationProvenance | None = None,
    validation_status: ValidationStatus | None = None,
    execution_context: ArtifactExecutionContext | None = None,
) -> tuple[ArtifactRegistry, str]:
    """Persist a text artifact and append it to the registry."""

    run_directory = Path(registry.run_directory)
    path = run_directory / name
    path.write_text(content, encoding="utf-8")
    artifact_id = f"artifact_{stable_hash(f'{artifact_type}|{path.name}')[:12]}"
    updated = registry.model_copy(
        update={
            "records": [
                *registry.records,
                ArtifactRecord(
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    path=str(path),
                    simulation_provenance=simulation_provenance,
                    validation_status=validation_status,
                    execution_context=execution_context
                    or build_artifact_execution_context(
                        simulation_provenance=simulation_provenance,
                        validation_status=validation_status,
                        replay_hint=f"replay from {path.name}",
                    ),
                ),
            ]
        }
    )
    return updated, artifact_id


def persist_json_artifact(
    registry: ArtifactRegistry,
    artifact_type: str,
    name: str,
    payload: dict[str, object],
    *,
    simulation_provenance: SimulationProvenance | None = None,
    validation_status: ValidationStatus | None = None,
    execution_context: ArtifactExecutionContext | None = None,
) -> tuple[ArtifactRegistry, str]:
    """Persist a JSON artifact and append it to the registry."""

    run_directory = Path(registry.run_directory)
    path = run_directory / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    artifact_id = f"artifact_{stable_hash(f'{artifact_type}|{path.name}')[:12]}"
    updated = registry.model_copy(
        update={
            "records": [
                *registry.records,
                ArtifactRecord(
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    path=str(path),
                    simulation_provenance=simulation_provenance,
                    validation_status=validation_status,
                    execution_context=execution_context
                    or build_artifact_execution_context(
                        simulation_provenance=simulation_provenance,
                        validation_status=validation_status,
                        replay_hint=f"replay from {path.name}",
                    ),
                ),
            ]
        }
    )
    return updated, artifact_id


def register_artifact(
    registry: ArtifactRegistry,
    artifact_type: str,
    path: str,
    *,
    simulation_provenance: SimulationProvenance | None = None,
    validation_status: ValidationStatus | None = None,
    execution_context: ArtifactExecutionContext | None = None,
) -> tuple[ArtifactRegistry, str]:
    """Register an existing artifact path without copying it."""

    artifact_path = Path(path)
    artifact_id = f"artifact_{stable_hash(f'{artifact_type}|{artifact_path.name}')[:12]}"
    updated = registry.model_copy(
        update={
            "records": [
                *registry.records,
                ArtifactRecord(
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    path=str(artifact_path),
                    simulation_provenance=simulation_provenance,
                    validation_status=validation_status,
                    execution_context=execution_context
                    or build_artifact_execution_context(
                        simulation_provenance=simulation_provenance,
                        validation_status=validation_status,
                        replay_hint=f"replay from {artifact_path.name}",
                    ),
                ),
            ]
        }
    )
    return updated, artifact_id
