"""Explain which configured-truth upgrade steps remain user-managed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_configured_truth_readiness import build_status as build_configured_truth_status


def build_status() -> dict[str, object]:
    readiness = build_configured_truth_status()
    return {
        "repo_managed_capabilities": [
            "paper_truth_policy",
            "configured_truth_contract_validation",
            "artifact_replay_manifest",
            "native_artifact_rerunability_check",
            "container_runtime_contract_check",
            "open_pdk_readiness_check",
        ],
        "user_managed_requirements": [
            "stage_or_mount_structured_pdk_root",
            "supply_external_model_card",
            "confirm_local_storage_and_mount_paths",
            "install_or_expose_docker_cli_for_container_smoke",
        ],
        "current_user_actions_required": readiness["user_actions_required"],
        "current_claim_state": readiness["readiness_state"],
        "notes": [
            "the repository can enforce truth-policy boundaries and readiness contracts",
            "only the user can provide proprietary or external model sources and local runtime mounts",
            "configured truth must not be claimed until the user-managed inputs are actually present",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(build_status(), indent=2))
