"""Generate repeated-episode memory evidence bundles for frozen vertical slices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.vertical_slices.folded_cascode import run_folded_cascode_memory_evidence
from libs.vertical_slices.ldo import run_ldo_memory_evidence
from libs.vertical_slices.ota2 import run_ota_memory_evidence
from libs.vertical_slices.bandgap import run_bandgap_memory_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repeated-episode memory evidence.")
    parser.add_argument(
        "--task",
        choices=["ota2", "folded_cascode", "ldo", "bandgap"],
        default="ota2",
        help="Frozen vertical slice to export repeated-episode memory evidence for.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of repeated episodes per mode.")
    parser.add_argument("--steps", type=int, default=3, help="Maximum planning steps per episode.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root for figures, tables, and JSON evidence bundle.",
    )
    args = parser.parse_args()

    if args.task == "ota2":
        output_root = args.output_root or Path("research/papers/ota2_v1")
        bundle = run_ota_memory_evidence(
            episodes=args.episodes,
            max_steps=args.steps,
            output_root=output_root,
        )
    elif args.task == "folded_cascode":
        output_root = args.output_root or Path("research/papers/folded_cascode_v1")
        bundle = run_folded_cascode_memory_evidence(
            episodes=args.episodes,
            max_steps=args.steps,
            output_root=output_root,
        )
    elif args.task == "ldo":
        output_root = args.output_root or Path("research/papers/ldo_v1")
        bundle = run_ldo_memory_evidence(
            episodes=args.episodes,
            max_steps=args.steps,
            output_root=output_root,
        )
    else:
        output_root = args.output_root or Path("research/papers/bandgap_v1")
        bundle = run_bandgap_memory_evidence(
            episodes=args.episodes,
            max_steps=args.steps,
            output_root=output_root,
        )
    print(bundle.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
