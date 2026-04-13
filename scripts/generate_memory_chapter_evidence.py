"""Generate chapter-level memory evidence across repeated and transfer studies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.eval.memory_evidence import (
    build_memory_ablation_evidence_bundle,
    build_memory_chapter_evidence_bundle,
)
from libs.vertical_slices.memory_transfer import (
    run_folded_cascode_to_ota_memory_transfer_suite,
    run_memory_transfer_evidence,
    run_ota_to_bandgap_memory_transfer_suite,
    run_ota_to_folded_cascode_memory_transfer_suite,
    run_ota_to_ldo_memory_transfer_suite,
)
from libs.vertical_slices.ota2 import run_ota_memory_ablation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chapter-level memory evidence bundle.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--transfer-episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("research/papers/memory_chapter"))
    args = parser.parse_args()

    repeated_suite = run_ota_memory_ablation_suite(episodes=args.episodes, max_steps=args.steps)
    repeated_bundle = build_memory_ablation_evidence_bundle(
        repeated_suite,
        figures_dir=args.output_root / "repeated_figs",
        tables_dir=args.output_root / "repeated_tables",
        json_output_path=args.output_root / "repeated_memory_bundle.json",
    )

    ota_to_folded = run_memory_transfer_evidence(
        source_task_slug="ota2-v1",
        target_task_slug="folded_cascode-v1",
        suite=run_ota_to_folded_cascode_memory_transfer_suite(
            source_episodes=args.transfer_episodes,
            target_episodes=args.transfer_episodes,
            max_steps=args.steps,
        ),
        output_root=args.output_root / "ota_to_folded_cascode",
    )
    folded_to_ota = run_memory_transfer_evidence(
        source_task_slug="folded_cascode-v1",
        target_task_slug="ota2-v1",
        suite=run_folded_cascode_to_ota_memory_transfer_suite(
            source_episodes=args.transfer_episodes,
            target_episodes=args.transfer_episodes,
            max_steps=args.steps,
        ),
        output_root=args.output_root / "folded_cascode_to_ota",
    )
    ota_to_ldo = run_memory_transfer_evidence(
        source_task_slug="ota2-v1",
        target_task_slug="ldo-v1",
        suite=run_ota_to_ldo_memory_transfer_suite(
            source_episodes=args.transfer_episodes,
            target_episodes=args.transfer_episodes,
            max_steps=args.steps,
        ),
        output_root=args.output_root / "ota_to_ldo",
    )
    ota_to_bandgap = run_memory_transfer_evidence(
        source_task_slug="ota2-v1",
        target_task_slug="bandgap-v1",
        suite=run_ota_to_bandgap_memory_transfer_suite(
            source_episodes=args.transfer_episodes,
            target_episodes=args.transfer_episodes,
            max_steps=args.steps,
        ),
        output_root=args.output_root / "ota_to_bandgap",
    )

    bundle = build_memory_chapter_evidence_bundle(
        repeated_bundle=repeated_bundle,
        same_family_bundles=[ota_to_folded, folded_to_ota],
        cross_family_bundles=[ota_to_ldo, ota_to_bandgap],
        figures_dir=args.output_root / "figs",
        tables_dir=args.output_root / "tables",
        json_output_path=args.output_root / "memory_chapter_evidence_bundle.json",
    )
    print(bundle.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
