"""Generate repeated-episode memory evidence bundles for frozen vertical slices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.vertical_slices.ota2 import run_ota_memory_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repeated-episode memory evidence.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of repeated episodes per mode.")
    parser.add_argument("--steps", type=int, default=3, help="Maximum planning steps per episode.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/papers/ota2_v1"),
        help="Output root for figures, tables, and JSON evidence bundle.",
    )
    args = parser.parse_args()

    bundle = run_ota_memory_evidence(
        episodes=args.episodes,
        max_steps=args.steps,
        output_root=args.output_root,
    )
    print(bundle.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
