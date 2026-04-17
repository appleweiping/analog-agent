"""Review whether planner rollout evidence is real observed evidence or only structural placeholder output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.schema.paper_evidence import PlannerAblationEvidenceBundle


def build_status(bundle: PlannerAblationEvidenceBundle) -> dict[str, object]:
    summary = bundle.summary
    evidence_state = (
        "real_observed_short_horizon_rollout_evidence"
        if summary.rollout_evidence_real_not_placeholder
        else "placeholder_or_structural_only"
    )
    return {
        "stage": "Stage D",
        "task_id": bundle.task_id,
        "evidence_state": evidence_state,
        "paper_safe": summary.rollout_evidence_real_not_placeholder,
        "rollout_guidance_observable": summary.rollout_guidance_observable,
        "rollout_claim_status": summary.rollout_claim_status,
        "rollout_claim_scope": summary.rollout_claim_scope,
        "placeholder_risk": summary.rollout_placeholder_risk,
        "notes": [
            "Real rollout evidence requires both observed rollout guidance in traces and a direct with/without-rollout comparison in the planner evidence bundle.",
            "Even when real, the safe claim remains short-horizon world-model guidance rather than broad MPC superiority.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-json", required=True, help="Path to planner_evidence_bundle.json")
    args = parser.parse_args()
    bundle = PlannerAblationEvidenceBundle.model_validate_json(Path(args.bundle_json).read_text(encoding="utf-8"))
    print(json.dumps(build_status(bundle), indent=2))


if __name__ == "__main__":
    main()
