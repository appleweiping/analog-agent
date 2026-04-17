from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from libs.eval.benchmark_reporting import (
    build_failure_mode_synthesis_bundle,
    build_family_summary_bundle,
    build_fidelity_corner_load_bundle,
    build_multitask_rollup_bundle,
    build_robustness_narrative_bundle,
)


class BenchmarkReportingTests(unittest.TestCase):
    def test_multitask_rollup_bundle_exports_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_multitask_rollup_bundle(output_root=root)

            self.assertEqual(bundle.scope, "multitask_rollup")
            self.assertEqual(len(bundle.tables), 2)
            self.assertTrue((root / "benchmark_multitask_rollup_bundle.json").exists())
            self.assertTrue((root / "benchmark_multitask_rollup_bundle.md").exists())
            self.assertTrue((root / "tables" / "benchmark_multitask_rollup.csv").exists())
            self.assertTrue((root / "tables" / "benchmark_protocol_roster.csv").exists())

    def test_family_summary_bundle_exports_family_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_family_summary_bundle(output_root=root)

            self.assertEqual(bundle.scope, "family_summary")
            self.assertEqual(len(bundle.tables), 2)
            self.assertTrue((root / "tables" / "benchmark_family_summary.csv").exists())
            self.assertTrue((root / "tables" / "benchmark_family_metric_coverage.csv").exists())

    def test_failure_mode_synthesis_bundle_exports_taxonomy_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_failure_mode_synthesis_bundle(output_root=root)

            self.assertEqual(bundle.scope, "failure_mode_synthesis")
            self.assertEqual(len(bundle.tables), 2)
            self.assertTrue((root / "tables" / "benchmark_failure_mode_taxonomy.csv").exists())
            self.assertTrue((root / "tables" / "benchmark_family_failure_focus.csv").exists())

    def test_robustness_bundle_keeps_current_suite_honest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_robustness_narrative_bundle(output_root=root)

            self.assertEqual(bundle.scope, "robustness_narrative")
            self.assertEqual(len(bundle.tables), 1)
            self.assertTrue((root / "tables" / "benchmark_robustness_scope.csv").exists())
            rows = bundle.tables[0].rows
            statuses = {row.values["robustness_claim_status"] for row in rows}
            self.assertTrue(statuses.issubset({"nominal_only_contract", "robustness_style_not_physical_validity_strong"}))
            self.assertNotIn("configured_robustness_candidate", statuses)

    def test_fidelity_corner_load_bundle_exports_condition_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_fidelity_corner_load_bundle(output_root=root)

            self.assertEqual(bundle.scope, "fidelity_corner_load_framing")
            self.assertEqual(len(bundle.tables), 2)
            self.assertTrue((root / "tables" / "benchmark_fidelity_framing.csv").exists())
            self.assertTrue((root / "tables" / "benchmark_condition_contract.csv").exists())
            self.assertTrue(bundle.narrative_sections)


if __name__ == "__main__":
    unittest.main()
