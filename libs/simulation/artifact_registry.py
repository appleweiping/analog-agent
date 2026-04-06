"""Artifact persistence and registry helpers for the fifth layer."""

from __future__ import annotations

import json
from pathlib import Path

from libs.schema.simulation import ArtifactRecord, ArtifactRegistry
from libs.utils.hashing import stable_hash


ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / ".artifacts" / "simulation"


def initialize_artifact_registry(simulation_id: str) -> ArtifactRegistry:
    """Create a writable artifact registry root for one simulation."""

    run_directory = ARTIFACT_ROOT / simulation_id
    run_directory.mkdir(parents=True, exist_ok=True)
    return ArtifactRegistry(run_directory=str(run_directory), records=[])


def persist_text_artifact(registry: ArtifactRegistry, artifact_type: str, name: str, content: str) -> tuple[ArtifactRegistry, str]:
    """Persist a text artifact and append it to the registry."""

    run_directory = Path(registry.run_directory)
    path = run_directory / name
    path.write_text(content, encoding="utf-8")
    artifact_id = f"artifact_{stable_hash(f'{artifact_type}|{path.name}')[:12]}"
    updated = registry.model_copy(
        update={
            "records": [
                *registry.records,
                ArtifactRecord(artifact_id=artifact_id, artifact_type=artifact_type, path=str(path)),
            ]
        }
    )
    return updated, artifact_id


def persist_json_artifact(registry: ArtifactRegistry, artifact_type: str, name: str, payload: dict[str, object]) -> tuple[ArtifactRegistry, str]:
    """Persist a JSON artifact and append it to the registry."""

    run_directory = Path(registry.run_directory)
    path = run_directory / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    artifact_id = f"artifact_{stable_hash(f'{artifact_type}|{path.name}')[:12]}"
    updated = registry.model_copy(
        update={
            "records": [
                *registry.records,
                ArtifactRecord(artifact_id=artifact_id, artifact_type=artifact_type, path=str(path)),
            ]
        }
    )
    return updated, artifact_id
