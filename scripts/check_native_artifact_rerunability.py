"""Check whether persisted native-truth artifacts remain rerunnable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.simulation.replay_tools import assess_replay_manifest


def build_status(artifacts_root: str | Path) -> dict[str, object]:
    root = Path(artifacts_root)
    manifests = sorted(root.glob("*/replay_manifest.json"))
    per_run = [assess_replay_manifest(path) for path in manifests]
    rerunnable_now = sum(1 for item in per_run if item["rerunnable_now"])
    replayable = sum(1 for item in per_run if item["replayable"])
    return {
        "artifacts_root": str(root),
        "manifest_count": len(manifests),
        "replayable_manifest_count": replayable,
        "rerunnable_now_count": rerunnable_now,
        "all_rerunnable_now": bool(per_run) and rerunnable_now == len(per_run),
        "runs": per_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check rerunability of native-truth artifacts.")
    parser.add_argument(
        "--artifacts-root",
        default=str(Path(__file__).resolve().parents[1] / ".artifacts" / "simulation"),
        help="Artifact root to scan",
    )
    args = parser.parse_args()
    print(json.dumps(build_status(args.artifacts_root), indent=2))


if __name__ == "__main__":
    main()
