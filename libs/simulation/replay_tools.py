"""Replay-oriented helpers for persisted native-truth artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from libs.schema.simulation import ArtifactReplayManifest, SimulationBundle, SimulationRequest, ValidationStatus


def build_replay_manifest(
    simulation_bundle: SimulationBundle,
    simulation_request: SimulationRequest,
    *,
    validation_status: ValidationStatus,
    measurement_report_path: str | None,
    verification_report_path: str | None,
) -> ArtifactReplayManifest:
    """Build a replay manifest from one executed simulation bundle."""

    records = simulation_bundle.artifact_registry.records
    netlist_paths = [record.path for record in records if record.artifact_type == "netlist"]
    log_paths = [record.path for record in records if record.artifact_type == "stdout"]
    raw_output_paths = [record.path for record in records if record.artifact_type == "raw_waveform"]
    resolved_binary = simulation_bundle.simulation_provenance.resolved_simulator_binary
    replay_commands: list[str] = []
    for netlist_path in netlist_paths:
        if resolved_binary:
            netlist = Path(netlist_path)
            replay_commands.append(f'"{resolved_binary}" -b "{netlist}" -o "{netlist.with_suffix(".log")}"')

    missing_requirements: list[str] = []
    if simulation_bundle.simulation_provenance.invocation_mode != "native":
        missing_requirements.append("native_backend_execution_required")
    if not resolved_binary:
        missing_requirements.append("resolved_simulator_binary_missing")
    if not netlist_paths:
        missing_requirements.append("netlist_artifacts_missing")

    return ArtifactReplayManifest(
        simulation_id=simulation_bundle.simulation_id,
        candidate_id=simulation_bundle.candidate_id,
        request_id=simulation_request.request_id,
        run_directory=simulation_bundle.artifact_registry.run_directory,
        invocation_mode=simulation_bundle.simulation_provenance.invocation_mode,
        replayable=not missing_requirements,
        resolved_simulator_binary=resolved_binary,
        truth_level=validation_status.truth_level,
        validation_status=validation_status.validity_state,
        fidelity_level=simulation_bundle.execution_profile.resolved_fidelity,
        physical_claim_scope=simulation_bundle.metadata.physical_claim_scope,
        netlist_paths=netlist_paths,
        log_paths=log_paths,
        raw_output_paths=raw_output_paths,
        measurement_report_path=measurement_report_path,
        verification_report_path=verification_report_path,
        replay_commands=replay_commands,
        missing_requirements=missing_requirements,
    )


def _load_manifest(path: str | Path) -> ArtifactReplayManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ArtifactReplayManifest.model_validate(payload)


def assess_replay_manifest(path: str | Path) -> dict[str, object]:
    """Assess whether a replay manifest is materially rerunnable in the current environment."""

    manifest = _load_manifest(path)
    binary_exists = bool(manifest.resolved_simulator_binary and Path(manifest.resolved_simulator_binary).exists())
    netlists_exist = all(Path(item).exists() for item in manifest.netlist_paths)
    logs_exist = all(Path(item).exists() for item in manifest.log_paths)
    measurement_exists = bool(manifest.measurement_report_path and Path(manifest.measurement_report_path).exists())
    verification_exists = bool(manifest.verification_report_path and Path(manifest.verification_report_path).exists())
    rerunnable = manifest.replayable and binary_exists and netlists_exist
    missing: list[str] = []
    if not binary_exists:
        missing.append("resolved_simulator_binary_missing_or_unavailable")
    if not netlists_exist:
        missing.append("netlist_artifacts_missing")
    if not logs_exist:
        missing.append("log_artifacts_missing")
    if not measurement_exists:
        missing.append("measurement_report_missing")
    if not verification_exists:
        missing.append("verification_report_missing")

    return {
        "simulation_id": manifest.simulation_id,
        "candidate_id": manifest.candidate_id,
        "replayable": manifest.replayable,
        "rerunnable_now": rerunnable,
        "run_directory": manifest.run_directory,
        "truth_level": manifest.truth_level,
        "validation_status": manifest.validation_status,
        "nominal_profile": manifest.physical_claim_scope.nominal_profile if manifest.physical_claim_scope else "",
        "truth_claim_tier": manifest.physical_claim_scope.truth_claim_tier if manifest.physical_claim_scope else "",
        "binary_exists": binary_exists,
        "netlists_exist": netlists_exist,
        "logs_exist": logs_exist,
        "measurement_exists": measurement_exists,
        "verification_exists": verification_exists,
        "replay_commands": list(manifest.replay_commands),
        "missing_requirements": [*manifest.missing_requirements, *missing],
    }
