"""Build the Day 51 physical-validity boundary package."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.eval.submission_package import build_physical_validity_boundary_bundle


def main() -> None:
    bundle = build_physical_validity_boundary_bundle(
        output_root=Path("research/papers/physical_validity_boundaries"),
    )
    print(bundle.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
