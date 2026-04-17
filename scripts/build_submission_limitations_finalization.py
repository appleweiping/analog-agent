"""Build the Day 57 limitations finalization bundle for the submission package."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.eval.submission_package import build_submission_limitations_finalization_bundle


def main() -> None:
    bundle = build_submission_limitations_finalization_bundle(
        profile_name="paper",
        output_root=Path("research/papers/submission_package/limitations"),
    )
    print(bundle.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
