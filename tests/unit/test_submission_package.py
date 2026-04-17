from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from libs.eval.submission_package import (
    PAPERS_ROOT,
    build_final_internal_submission_package_bundle,
    build_physical_validity_boundary_bundle,
    build_submission_experiments_alignment_bundle,
    build_submission_appendix_allocation_bundle,
    build_submission_limitations_finalization_bundle,
    build_submission_main_figure_freeze_bundle,
    build_submission_manuscript_structure_freeze_bundle,
    build_submission_main_table_freeze_bundle,
    build_submission_protocol_finalization_bundle,
)


class SubmissionPackageTests(unittest.TestCase):
    def test_physical_validity_boundary_bundle_exports_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_physical_validity_boundary_bundle(output_root=root)

            self.assertEqual(len(bundle.tables), 2)
            self.assertTrue((root / "physical_validity_boundary_bundle.json").exists())
            self.assertTrue((root / "tables" / "submission_physical_validity_boundary.md").exists())
            self.assertIn("Top-tier positioning", bundle.summary_notes[1])

    def test_main_figure_freeze_tracks_ready_and_manual_assets(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_submission_main_figure_freeze_bundle(
                profile_name="unit",
                output_root=root,
                papers_root=PAPERS_ROOT,
            )

            self.assertEqual(bundle.asset_kind, "figure")
            self.assertEqual(bundle.pending_entry_count, 0)
            self.assertGreaterEqual(bundle.ready_entry_count, 9)
            self.assertTrue((root / "main_figures" / "fig_system_architecture.svg").exists())
            self.assertTrue((root / "main_figures" / "fig_ota_acceptance_trace.svg").exists())
            self.assertTrue((root / "main_figures" / "world_model_prediction_gap_vs_step.svg").exists())
            self.assertTrue((root / "submission_main_figure_freeze_bundle.md").exists())

    def test_main_table_freeze_exports_submission_summary_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_submission_main_table_freeze_bundle(
                profile_name="unit",
                output_root=root,
                papers_root=PAPERS_ROOT,
            )

            self.assertEqual(bundle.asset_kind, "table")
            self.assertEqual(bundle.ready_entry_count, 6)
            self.assertTrue((root / "main_tables" / "submission_benchmark_summary.md").exists())
            self.assertTrue((root / "main_tables" / "submission_memory_transfer_summary.csv").exists())

    def test_appendix_allocation_exports_supporting_assets(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_submission_appendix_allocation_bundle(
                profile_name="unit",
                output_root=root,
                papers_root=PAPERS_ROOT,
            )

            self.assertTrue((root / "submission_appendix_allocation_bundle.json").exists())
            self.assertGreater(len(bundle.appendix_figures), 5)
            self.assertGreater(len(bundle.appendix_tables), 10)
            self.assertTrue(any(path.endswith("submission_claim_boundary_policy.md") for path in bundle.appendix_tables))

    def test_protocol_finalization_bundle_freezes_tables_and_sections(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_submission_protocol_finalization_bundle(
                profile_name="unit",
                output_root=root,
                papers_root=PAPERS_ROOT,
            )

            self.assertEqual(bundle.bundle_kind, "protocol")
            self.assertEqual(bundle.pending_section_count, 0)
            self.assertGreaterEqual(bundle.ready_section_count, 8)
            self.assertTrue((root / "tables" / "submission_protocol_benchmark_matrix.md").exists())
            self.assertTrue((root / "frozen_docs" / "experimental_protocol.md").exists())

    def test_limitations_finalization_bundle_exports_boundary_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_submission_limitations_finalization_bundle(
                profile_name="unit",
                output_root=root,
                papers_root=PAPERS_ROOT,
            )

            self.assertEqual(bundle.bundle_kind, "limitations")
            self.assertEqual(bundle.pending_section_count, 0)
            self.assertTrue((root / "tables" / "submission_limitations_matrix.csv").exists())
            self.assertIn("Bandgap", bundle.summary_notes[1])

    def test_manuscript_structure_freeze_tracks_manual_sections(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_submission_manuscript_structure_freeze_bundle(
                profile_name="unit",
                output_root=root,
                papers_root=PAPERS_ROOT,
            )

            self.assertEqual(bundle.bundle_kind, "manuscript")
            self.assertEqual(bundle.pending_section_count, 0)
            self.assertGreaterEqual(bundle.ready_section_count, 10)
            self.assertTrue((root / "tables" / "submission_manuscript_structure_freeze.md").exists())

    def test_experiments_alignment_bundle_surfaces_manual_trace_gap(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_submission_experiments_alignment_bundle(
                profile_name="unit",
                output_root=root,
                papers_root=PAPERS_ROOT,
            )

            self.assertEqual(bundle.pending_entry_count, 0)
            self.assertEqual(bundle.aligned_entry_count, 5)
            self.assertTrue((root / "tables" / "submission_experiments_alignment.md").exists())

    def test_final_internal_submission_package_is_internal_review_ready(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_final_internal_submission_package_bundle(
                profile_name="unit",
                output_root=root,
                papers_root=PAPERS_ROOT,
            )

            self.assertTrue(bundle.internal_review_ready)
            self.assertTrue(bundle.external_submission_ready)
            self.assertEqual(bundle.unresolved_manual_asset_ids, [])
            self.assertTrue((root / "final_internal_submission_package_bundle.md").exists())


if __name__ == "__main__":
    unittest.main()
