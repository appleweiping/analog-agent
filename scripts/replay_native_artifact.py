"""Inspect a replay manifest for one native-truth artifact run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.simulation.replay_tools import assess_replay_manifest


def _manifest_path(run_dir: str | None, manifest_path: str | None) -> Path:
    if manifest_path:
        return Path(manifest_path)
    if run_dir:
        return Path(run_dir) / "replay_manifest.json"
    raise ValueError("either --manifest-path or --run-dir must be provided")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect replay metadata for one native-truth artifact run.")
    parser.add_argument("--run-dir", help="Simulation artifact run directory containing replay_manifest.json")
    parser.add_argument("--manifest-path", help="Direct path to replay_manifest.json")
    args = parser.parse_args()

    manifest = _manifest_path(args.run_dir, args.manifest_path)
    if not manifest.exists():
        print(
            json.dumps(
                {
                    "manifest_present": False,
                    "manifest_path": str(manifest),
                    "error": "replay_manifest_not_found",
                    "recommended_action": "rerun verification with the upgraded replay-manifest pipeline",
                },
                indent=2,
            )
        )
        return
    print(json.dumps(assess_replay_manifest(manifest), indent=2))


if __name__ == "__main__":
    main()
