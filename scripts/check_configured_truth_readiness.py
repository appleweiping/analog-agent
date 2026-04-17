"""Review whether configured-truth claims are structurally ready."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.worker_simulator.ngspice_runner import _load_ngspice_config, configured_pdk_root, external_model_card_path
from scripts.check_container_runtime import build_status as build_container_status
from scripts.check_open_pdk_ready import build_status as build_open_pdk_status


def build_status() -> dict[str, object]:
    config = _load_ngspice_config()
    pdk_status = build_open_pdk_status()
    container_status = build_container_status()
    configured_mode = str(config.get("configured_truth_mode", "disabled"))
    contract_name = str(config.get("configured_truth_contract", "sky130_open"))
    model_source_policy = str(config.get("configured_truth_model_source", "external_model_card_or_pdk_root"))
    pdk_root = configured_pdk_root()
    model_card = external_model_card_path()

    if configured_mode == "disabled":
        readiness_state = "demonstrator_only"
    elif pdk_status["ready"] and model_card and Path(model_card).exists():
        readiness_state = "configured_truth_ready"
    elif pdk_status["ready"]:
        readiness_state = "configured_truth_candidate_ready"
    else:
        readiness_state = "configured_truth_not_ready"

    blockers: list[str] = []
    if readiness_state == "configured_truth_not_ready":
        blockers.extend(str(item) for item in pdk_status["missing_required_subpaths"])
    if configured_mode != "disabled" and not model_card:
        blockers.append("external_model_card_missing")
    if configured_mode != "disabled" and not container_status["ready"]:
        blockers.append("container_runtime_contract_not_ready")

    user_actions_required: list[str] = []
    if configured_mode != "disabled" and not pdk_root:
        user_actions_required.append("stage_or_mount_structured_pdk_root")
    if configured_mode != "disabled" and not model_card:
        user_actions_required.append("supply_external_model_card")
    if configured_mode != "disabled" and not container_status["docker_cli_available"]:
        user_actions_required.append("install_or_expose_docker_cli_for_container_smoke")

    recommended_next_actions = list(dict.fromkeys([*pdk_status["recommended_actions"], *user_actions_required]))
    return {
        "readiness_state": readiness_state,
        "configured_truth_mode": configured_mode,
        "configured_truth_contract": contract_name,
        "configured_truth_model_source": model_source_policy,
        "pdk_root": str(pdk_root) if pdk_root else "",
        "pdk_root_present": bool(pdk_root and Path(pdk_root).exists()),
        "external_model_card": str(model_card) if model_card else "",
        "external_model_card_present": bool(model_card and Path(model_card).exists()),
        "open_pdk_status": pdk_status,
        "container_runtime_ready": container_status["ready"],
        "claim_blockers": blockers,
        "user_actions_required": user_actions_required,
        "recommended_next_actions": recommended_next_actions,
    }


if __name__ == "__main__":
    print(json.dumps(build_status(), indent=2))
