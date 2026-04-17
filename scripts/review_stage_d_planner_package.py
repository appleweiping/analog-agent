"""Review Stage D planner-package readiness in one structured report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.schema.paper_evidence import PlannerAblationSummary, PlannerPaperLayoutBundle


def _has_fields(model, required: list[str]) -> bool:
    fields = getattr(model, "model_fields", {})
    return all(field in fields for field in required)


def build_status() -> dict[str, object]:
    planner_summary_ready = _has_fields(
        PlannerAblationSummary,
        [
            "planner_reduces_simulations_vs_top_k",
            "rollout_evidence_real_not_placeholder",
            "dominant_failure_mode",
            "efficiency_synthesis_ready",
        ],
    )
    planner_layout_ready = _has_fields(
        PlannerPaperLayoutBundle,
        [
            "main_figure_captions",
            "appendix_figure_captions",
            "main_table_captions",
            "appendix_table_captions",
        ],
    )

    script_paths = {
        "planner_evidence_script": REPO_ROOT / "scripts" / "generate_vertical_slice_planner_evidence.py",
        "planner_layout_script": REPO_ROOT / "scripts" / "build_planner_paper_layout.py",
        "rollout_claim_audit_script": REPO_ROOT / "scripts" / "audit_planner_rollout_claim.py",
        "rollout_evidence_review_script": REPO_ROOT / "scripts" / "review_planner_rollout_evidence.py",
    }
    scripts_ready = all(path.exists() for path in script_paths.values())

    stage_status = (
        "stage_d_planner_package_complete"
        if planner_summary_ready and planner_layout_ready and scripts_ready
        else "stage_d_planner_package_incomplete"
    )
    return {
        "stage": "Stage D",
        "stage_status": stage_status,
        "planner_summary_ready": planner_summary_ready,
        "planner_layout_ready": planner_layout_ready,
        "scripts_ready": scripts_ready,
        "expected_scripts": {key: str(path) for key, path in script_paths.items()},
        "ready_for_stage_e": stage_status == "stage_d_planner_package_complete",
        "notes": [
            "Stage D is considered complete when planner-ablation claims, rollout wording boundaries, failure/efficiency syntheses, and main/appendix packaging are all represented as code-backed artifacts.",
            "This review script checks structural readiness and does not claim that every benchmark rerun has already been regenerated in the current environment.",
        ],
    }


def main() -> None:
    print(json.dumps(build_status(), indent=2))


if __name__ == "__main__":
    main()
