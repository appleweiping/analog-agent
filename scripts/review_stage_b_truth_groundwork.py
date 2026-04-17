"""Review Stage B configured-truth groundwork status in one structured report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_configured_truth_readiness import build_status as configured_truth_status
from scripts.check_native_artifact_rerunability import build_status as rerunability_status
from scripts.explain_user_action_boundary import build_status as user_boundary_status


def build_status() -> dict[str, object]:
    readiness = configured_truth_status()
    rerunability = rerunability_status(Path(__file__).resolve().parents[1] / ".artifacts" / "simulation")
    user_boundary = user_boundary_status()

    readiness_state = str(readiness["readiness_state"])
    if readiness_state == "demonstrator_only":
        stage_status = "stage_b_groundwork_complete"
    elif readiness_state == "configured_truth_candidate_ready":
        stage_status = "configured_truth_candidate_ready"
    elif readiness_state == "configured_truth_ready":
        stage_status = "configured_truth_ready"
    else:
        stage_status = "configured_truth_still_blocked"

    return {
        "stage": "Stage B",
        "stage_status": stage_status,
        "configured_truth_state": readiness_state,
        "artifact_rerunability": {
            "manifest_count": rerunability["manifest_count"],
            "rerunnable_now_count": rerunability["rerunnable_now_count"],
            "all_rerunnable_now": rerunability["all_rerunnable_now"],
        },
        "user_actions_required": user_boundary["current_user_actions_required"],
        "claim_blockers": readiness["claim_blockers"],
        "ready_for_stage_c": True,
        "notes": [
            "Stage B is considered complete once configured-truth boundaries are explicit and replay contracts exist.",
            "Configured truth itself may still remain blocked while no external PDK or model-card is staged.",
        ],
    }


def main() -> None:
    print(json.dumps(build_status(), indent=2))


if __name__ == "__main__":
    main()
