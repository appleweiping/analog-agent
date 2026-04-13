from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.eval.memory_evidence import (
    build_memory_ablation_evidence_bundle,
    build_memory_chapter_evidence_bundle,
    build_memory_negative_transfer_case_studies,
    build_memory_paper_layout_bundle,
)
from libs.vertical_slices.folded_cascode import run_folded_cascode_memory_ablation_suite
from libs.vertical_slices.memory_transfer import (
    run_memory_transfer_evidence,
    run_ota_to_bandgap_memory_transfer_suite,
    run_ota_to_folded_cascode_memory_transfer_suite,
)
from libs.vertical_slices.ota2 import run_ota_memory_ablation_suite


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class MemorySubmissionPackageTests(unittest.TestCase):
    def test_submission_package_exports_layout_and_case_study(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repeated_bundles = [
                build_memory_ablation_evidence_bundle(
                    run_ota_memory_ablation_suite(episodes=3, max_steps=2),
                    figures_dir=root / "ota2_repeated_figs",
                    tables_dir=root / "ota2_repeated_tables",
                    json_output_path=root / "ota2_repeated_memory_bundle.json",
                ),
                build_memory_ablation_evidence_bundle(
                    run_folded_cascode_memory_ablation_suite(episodes=3, max_steps=2),
                    figures_dir=root / "folded_repeated_figs",
                    tables_dir=root / "folded_repeated_tables",
                    json_output_path=root / "folded_repeated_memory_bundle.json",
                ),
            ]
            same_family = [
                run_memory_transfer_evidence(
                    source_task_slug="ota2-v1",
                    target_task_slug="folded_cascode-v1",
                    suite=run_ota_to_folded_cascode_memory_transfer_suite(
                        source_episodes=2,
                        target_episodes=2,
                        max_steps=2,
                    ),
                    output_root=root / "ota_to_folded",
                ),
            ]
            cross_family = [
                run_memory_transfer_evidence(
                    source_task_slug="ota2-v1",
                    target_task_slug="bandgap-v1",
                    suite=run_ota_to_bandgap_memory_transfer_suite(
                        source_episodes=2,
                        target_episodes=2,
                        max_steps=2,
                    ),
                    output_root=root / "ota_to_bandgap",
                ),
            ]
            chapter_bundle = build_memory_chapter_evidence_bundle(
                repeated_bundles=repeated_bundles,
                same_family_bundles=same_family,
                cross_family_bundles=cross_family,
                figures_dir=root / "chapter_figs",
                tables_dir=root / "chapter_tables",
                json_output_path=root / "memory_chapter_evidence_bundle.json",
            )
            case_studies = build_memory_negative_transfer_case_studies(
                cross_family,
                output_root=root / "case_studies",
            )
            layout_bundle = build_memory_paper_layout_bundle(
                profile_name="fast",
                repeated_bundles=repeated_bundles,
                same_family_bundles=same_family,
                cross_family_bundles=cross_family,
                chapter_bundle=chapter_bundle,
                case_studies=case_studies,
                output_root=root / "layout",
            )

            self.assertTrue((root / "case_studies").exists())
            self.assertTrue(any(case.selected_as_primary_case for case in case_studies))
            self.assertTrue((root / "layout" / "memory_paper_layout.md").exists())
            self.assertTrue((root / "layout" / "memory_paper_layout_bundle.json").exists())
            self.assertEqual(len(layout_bundle.main_figures), 6)
            self.assertEqual(len(layout_bundle.main_tables), 6)
            self.assertGreaterEqual(len(layout_bundle.case_studies), 2)


if __name__ == "__main__":
    unittest.main()
