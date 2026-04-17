"""Audit whether planner evidence supports a rollout claim without overclaiming MPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.schema.paper_evidence import PlannerAblationEvidenceBundle


def build_audit(bundle: PlannerAblationEvidenceBundle) -> dict[str, object]:
    figure_ids = {figure.figure_id for figure in bundle.figures}
    table_ids = {table.table_id for table in bundle.tables}
    summary = bundle.summary
    evidence_present = {
        "rollout_guidance_rate_figure": "fig_planner_rollout_guidance_rate" in figure_ids,
        "rollout_claim_audit_figure": "fig_planner_rollout_claim_audit" in figure_ids,
        "rollout_claim_audit_table": "tbl_planner_rollout_claim_audit" in table_ids,
    }
    structural_ready = all(evidence_present.values())
    if not structural_ready or not summary.rollout_guidance_observable:
        claim_status = "do_not_claim_rollout_utility"
        recommended_claim = (
            "Current evidence supports rollout-capable planner infrastructure, but not a standalone utility claim."
        )
    elif summary.rollout_claim_supported_without_mpc_overclaim:
        claim_status = "support_short_horizon_rollout_guidance_claim"
        recommended_claim = (
            "Claim only that short-horizon world-model rollout guidance improves planner convergence under the current benchmark suite."
        )
    else:
        claim_status = "limit_to_observable_rollout_behavior"
        recommended_claim = (
            "Claim only that rollout guidance is observable in planner traces; avoid saying it is broadly beneficial unless stronger reruns are added."
        )
    return {
        "stage": "Stage D",
        "claim_scope": summary.rollout_claim_scope,
        "claim_status": claim_status,
        "structural_ready": structural_ready,
        "real_not_placeholder": summary.rollout_evidence_real_not_placeholder,
        "evidence_present": evidence_present,
        "rollout_guidance_observable": summary.rollout_guidance_observable,
        "rollout_guidance_effective": summary.rollout_guidance_effective,
        "rollout_guidance_improves_convergence": summary.rollout_guidance_improves_convergence,
        "rollout_guidance_preserves_or_improves_feasible_hit_rate": summary.rollout_guidance_preserves_or_improves_feasible_hit_rate,
        "recommended_claim": recommended_claim,
        "avoid_claims": [
            "general_mpc_superiority",
            "long_horizon_optimal_control",
            "global_optimality_under_unknown_dynamics",
        ],
        "notes": [
            "This audit intentionally limits the safe wording to short-horizon world-model-guidance claims.",
            "Use broader MPC language only after dedicated rollout studies show stronger task-level and multitask evidence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-json", required=True, help="Path to planner_evidence_bundle.json")
    args = parser.parse_args()
    bundle = PlannerAblationEvidenceBundle.model_validate_json(Path(args.bundle_json).read_text(encoding="utf-8"))
    print(json.dumps(build_audit(bundle), indent=2))


if __name__ == "__main__":
    main()
