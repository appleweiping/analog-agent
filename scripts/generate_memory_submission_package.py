"""Generate submission-facing memory chapter materials with a larger study profile."""

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
    build_memory_negative_transfer_case_studies,
    build_memory_paper_layout_bundle,
)
from libs.vertical_slices.bandgap import run_bandgap_memory_ablation_suite
from libs.vertical_slices.folded_cascode import run_folded_cascode_memory_ablation_suite
from libs.vertical_slices.ldo import run_ldo_memory_ablation_suite
from libs.vertical_slices.memory_transfer import (
    run_folded_cascode_to_bandgap_memory_transfer_suite,
    run_folded_cascode_to_ldo_memory_transfer_suite,
    run_folded_cascode_to_ota_memory_transfer_suite,
    run_memory_transfer_evidence,
    run_ota_to_bandgap_memory_transfer_suite,
    run_ota_to_folded_cascode_memory_transfer_suite,
    run_ota_to_ldo_memory_transfer_suite,
)
from libs.vertical_slices.ota2 import run_ota_memory_ablation_suite

PROFILES = {
    "paper": {"episodes": 10, "transfer_episodes": 4, "steps": 3},
    "fast": {"episodes": 4, "transfer_episodes": 2, "steps": 2},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate submission-facing memory chapter package.")
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="paper")
    parser.add_argument("--output-root", type=Path, default=Path("research/papers/memory_chapter"))
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    output_root = args.output_root

    repeated_bundles = [
        build_memory_ablation_evidence_bundle(
            run_ota_memory_ablation_suite(episodes=profile["episodes"], max_steps=profile["steps"]),
            figures_dir=output_root / "ota2_repeated_figs",
            tables_dir=output_root / "ota2_repeated_tables",
            json_output_path=output_root / "ota2_repeated_memory_bundle.json",
        ),
        build_memory_ablation_evidence_bundle(
            run_folded_cascode_memory_ablation_suite(episodes=profile["episodes"], max_steps=profile["steps"]),
            figures_dir=output_root / "folded_repeated_figs",
            tables_dir=output_root / "folded_repeated_tables",
            json_output_path=output_root / "folded_repeated_memory_bundle.json",
        ),
        build_memory_ablation_evidence_bundle(
            run_ldo_memory_ablation_suite(episodes=profile["episodes"], max_steps=profile["steps"]),
            figures_dir=output_root / "ldo_repeated_figs",
            tables_dir=output_root / "ldo_repeated_tables",
            json_output_path=output_root / "ldo_repeated_memory_bundle.json",
        ),
        build_memory_ablation_evidence_bundle(
            run_bandgap_memory_ablation_suite(episodes=profile["episodes"], max_steps=profile["steps"]),
            figures_dir=output_root / "bandgap_repeated_figs",
            tables_dir=output_root / "bandgap_repeated_tables",
            json_output_path=output_root / "bandgap_repeated_memory_bundle.json",
        ),
    ]

    same_family_bundles = [
        run_memory_transfer_evidence(
            source_task_slug="ota2-v1",
            target_task_slug="folded_cascode-v1",
            suite=run_ota_to_folded_cascode_memory_transfer_suite(
                source_episodes=profile["transfer_episodes"],
                target_episodes=profile["transfer_episodes"],
                max_steps=profile["steps"],
            ),
            output_root=output_root / "ota_to_folded_cascode",
        ),
        run_memory_transfer_evidence(
            source_task_slug="folded_cascode-v1",
            target_task_slug="ota2-v1",
            suite=run_folded_cascode_to_ota_memory_transfer_suite(
                source_episodes=profile["transfer_episodes"],
                target_episodes=profile["transfer_episodes"],
                max_steps=profile["steps"],
            ),
            output_root=output_root / "folded_cascode_to_ota",
        ),
    ]

    cross_family_bundles = [
        run_memory_transfer_evidence(
            source_task_slug="ota2-v1",
            target_task_slug="ldo-v1",
            suite=run_ota_to_ldo_memory_transfer_suite(
                source_episodes=profile["transfer_episodes"],
                target_episodes=profile["transfer_episodes"],
                max_steps=profile["steps"],
            ),
            output_root=output_root / "ota_to_ldo",
        ),
        run_memory_transfer_evidence(
            source_task_slug="ota2-v1",
            target_task_slug="bandgap-v1",
            suite=run_ota_to_bandgap_memory_transfer_suite(
                source_episodes=profile["transfer_episodes"],
                target_episodes=profile["transfer_episodes"],
                max_steps=profile["steps"],
            ),
            output_root=output_root / "ota_to_bandgap",
        ),
        run_memory_transfer_evidence(
            source_task_slug="folded_cascode-v1",
            target_task_slug="ldo-v1",
            suite=run_folded_cascode_to_ldo_memory_transfer_suite(
                source_episodes=profile["transfer_episodes"],
                target_episodes=profile["transfer_episodes"],
                max_steps=profile["steps"],
            ),
            output_root=output_root / "folded_cascode_to_ldo",
        ),
        run_memory_transfer_evidence(
            source_task_slug="folded_cascode-v1",
            target_task_slug="bandgap-v1",
            suite=run_folded_cascode_to_bandgap_memory_transfer_suite(
                source_episodes=profile["transfer_episodes"],
                target_episodes=profile["transfer_episodes"],
                max_steps=profile["steps"],
            ),
            output_root=output_root / "folded_cascode_to_bandgap",
        ),
    ]

    chapter_bundle = build_memory_chapter_evidence_bundle(
        repeated_bundles=repeated_bundles,
        same_family_bundles=same_family_bundles,
        cross_family_bundles=cross_family_bundles,
        figures_dir=output_root / "figs",
        tables_dir=output_root / "tables",
        json_output_path=output_root / "memory_chapter_evidence_bundle.json",
    )
    case_studies = build_memory_negative_transfer_case_studies(
        cross_family_bundles,
        output_root=output_root / "case_studies",
    )
    layout_bundle = build_memory_paper_layout_bundle(
        profile_name=args.profile,
        repeated_bundles=repeated_bundles,
        same_family_bundles=same_family_bundles,
        cross_family_bundles=cross_family_bundles,
        chapter_bundle=chapter_bundle,
        case_studies=case_studies,
        output_root=output_root / "layout",
    )
    print(layout_bundle.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
