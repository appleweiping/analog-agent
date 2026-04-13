"""Generate cross-task memory transfer evidence bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.vertical_slices.memory_transfer import (
    run_memory_transfer_evidence,
    run_ota_to_bandgap_memory_transfer_suite,
    run_ota_to_folded_cascode_memory_transfer_suite,
    run_ota_to_ldo_memory_transfer_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cross-task memory transfer evidence.")
    parser.add_argument(
        "--target",
        choices=["folded_cascode", "ldo", "bandgap"],
        default="folded_cascode",
    )
    parser.add_argument("--source-episodes", type=int, default=3)
    parser.add_argument("--target-episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=Path("research/papers"))
    args = parser.parse_args()

    if args.target == "folded_cascode":
        suite = run_ota_to_folded_cascode_memory_transfer_suite(
            source_episodes=args.source_episodes,
            target_episodes=args.target_episodes,
            max_steps=args.steps,
        )
        output_root = args.output_root / "folded_cascode_v1"
    elif args.target == "ldo":
        suite = run_ota_to_ldo_memory_transfer_suite(
            source_episodes=args.source_episodes,
            target_episodes=args.target_episodes,
            max_steps=args.steps,
        )
        output_root = args.output_root / "ldo_v1"
    else:
        suite = run_ota_to_bandgap_memory_transfer_suite(
            source_episodes=args.source_episodes,
            target_episodes=args.target_episodes,
            max_steps=args.steps,
        )
        output_root = args.output_root / "bandgap_v1"

    bundle = run_memory_transfer_evidence(
        source_task_slug=suite.source_task_slug,
        target_task_slug=suite.target_task_slug,
        suite=suite,
        output_root=output_root,
    )
    print(bundle.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
