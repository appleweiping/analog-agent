from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.eval.memory_evidence import (
    build_memory_ablation_evidence_bundle,
    build_memory_chapter_evidence_bundle,
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


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class MemoryChapterEvidenceTests(unittest.TestCase):
    def test_memory_chapter_bundle_exports_structured_figures_and_tables(self) -> None:
        repeated_suite = run_ota_memory_ablation_suite(episodes=3, max_steps=2)

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repeated_bundles = [
                build_memory_ablation_evidence_bundle(
                    repeated_suite,
                    figures_dir=root / "repeated_figs",
                    tables_dir=root / "repeated_tables",
                    json_output_path=root / "repeated_memory_bundle.json",
                ),
                build_memory_ablation_evidence_bundle(
                    run_folded_cascode_memory_ablation_suite(episodes=2, max_steps=2),
                    figures_dir=root / "folded_repeated_figs",
                    tables_dir=root / "folded_repeated_tables",
                    json_output_path=root / "folded_repeated_memory_bundle.json",
                ),
                build_memory_ablation_evidence_bundle(
                    run_ldo_memory_ablation_suite(episodes=2, max_steps=2),
                    figures_dir=root / "ldo_repeated_figs",
                    tables_dir=root / "ldo_repeated_tables",
                    json_output_path=root / "ldo_repeated_memory_bundle.json",
                ),
                build_memory_ablation_evidence_bundle(
                    run_bandgap_memory_ablation_suite(episodes=2, max_steps=2),
                    figures_dir=root / "bandgap_repeated_figs",
                    tables_dir=root / "bandgap_repeated_tables",
                    json_output_path=root / "bandgap_repeated_memory_bundle.json",
                ),
            ]
            ota_to_folded = run_memory_transfer_evidence(
                source_task_slug="ota2-v1",
                target_task_slug="folded_cascode-v1",
                suite=run_ota_to_folded_cascode_memory_transfer_suite(source_episodes=2, target_episodes=2, max_steps=2),
                output_root=root / "ota_to_folded",
            )
            folded_to_ota = run_memory_transfer_evidence(
                source_task_slug="folded_cascode-v1",
                target_task_slug="ota2-v1",
                suite=run_folded_cascode_to_ota_memory_transfer_suite(source_episodes=2, target_episodes=2, max_steps=2),
                output_root=root / "folded_to_ota",
            )
            ota_to_ldo = run_memory_transfer_evidence(
                source_task_slug="ota2-v1",
                target_task_slug="ldo-v1",
                suite=run_ota_to_ldo_memory_transfer_suite(source_episodes=2, target_episodes=2, max_steps=2),
                output_root=root / "ota_to_ldo",
            )
            ota_to_bandgap = run_memory_transfer_evidence(
                source_task_slug="ota2-v1",
                target_task_slug="bandgap-v1",
                suite=run_ota_to_bandgap_memory_transfer_suite(source_episodes=2, target_episodes=2, max_steps=2),
                output_root=root / "ota_to_bandgap",
            )
            folded_to_ldo = run_memory_transfer_evidence(
                source_task_slug="folded_cascode-v1",
                target_task_slug="ldo-v1",
                suite=run_folded_cascode_to_ldo_memory_transfer_suite(source_episodes=2, target_episodes=2, max_steps=2),
                output_root=root / "folded_to_ldo",
            )
            folded_to_bandgap = run_memory_transfer_evidence(
                source_task_slug="folded_cascode-v1",
                target_task_slug="bandgap-v1",
                suite=run_folded_cascode_to_bandgap_memory_transfer_suite(source_episodes=2, target_episodes=2, max_steps=2),
                output_root=root / "folded_to_bandgap",
            )

            chapter_bundle = build_memory_chapter_evidence_bundle(
                repeated_bundles=repeated_bundles,
                same_family_bundles=[ota_to_folded, folded_to_ota],
                cross_family_bundles=[ota_to_ldo, ota_to_bandgap, folded_to_ldo, folded_to_bandgap],
                figures_dir=root / "chapter_figs",
                tables_dir=root / "chapter_tables",
                json_output_path=root / "memory_chapter_evidence_bundle.json",
            )

            self.assertTrue((root / "memory_chapter_evidence_bundle.json").exists())
            self.assertTrue((root / "chapter_figs" / "memory_chapter_repeated_episode_calls.svg").exists())
            self.assertTrue((root / "chapter_figs" / "memory_chapter_repeated_episode_failures.svg").exists())
            self.assertTrue((root / "chapter_figs" / "memory_chapter_step_to_feasible.svg").exists())
            self.assertTrue((root / "chapter_figs" / "memory_chapter_prediction_gap.svg").exists())
            self.assertTrue((root / "chapter_figs" / "memory_chapter_same_family_transfer.svg").exists())
            self.assertTrue((root / "chapter_figs" / "memory_chapter_cross_family_governance.svg").exists())
            self.assertTrue((root / "chapter_tables" / "memory_chapter_transfer_summary.csv").exists())
            self.assertTrue((root / "chapter_tables" / "memory_chapter_negative_transfer.csv").exists())
            self.assertTrue(chapter_bundle.summary.repeated_episode_beneficial)
            self.assertTrue(chapter_bundle.summary.repeated_episode_generalizes_beyond_ota)
            self.assertTrue(chapter_bundle.summary.same_family_transfer_beneficial)
            self.assertTrue(chapter_bundle.summary.governance_blocks_cross_family_negative_transfer)
            self.assertTrue(chapter_bundle.summary.retrieval_utility_observable)


if __name__ == "__main__":
    unittest.main()
