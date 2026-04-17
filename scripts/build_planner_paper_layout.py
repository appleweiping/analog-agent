"""Build main-text and appendix planner figure/table layout from one planner evidence bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.eval.planner_evidence import build_planner_paper_layout_bundle
from libs.schema.paper_evidence import PlannerAblationEvidenceBundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-json", required=True, help="Path to planner_evidence_bundle.json")
    parser.add_argument("--profile", default="paper")
    parser.add_argument("--output-root", type=Path, default=Path("research/papers/planner_layout"))
    args = parser.parse_args()

    bundle = PlannerAblationEvidenceBundle.model_validate_json(Path(args.bundle_json).read_text(encoding="utf-8"))
    layout = build_planner_paper_layout_bundle(
        profile_name=args.profile,
        bundle=bundle,
        output_root=args.output_root,
    )
    print(layout.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
