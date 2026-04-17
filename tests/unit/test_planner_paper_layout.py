from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from libs.eval.planner_evidence import (
    build_planner_ablation_evidence_bundle,
    build_planner_paper_layout_bundle,
)
from tests.unit.test_planner_evidence import PlannerEvidenceUnitTests


class PlannerPaperLayoutUnitTests(unittest.TestCase):
    def test_layout_splits_main_and_appendix_outputs(self) -> None:
        fixture = PlannerEvidenceUnitTests()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_planner_ablation_evidence_bundle(
                fixture._suite(),
                figures_dir=root / "figs",
                tables_dir=root / "tables",
                json_output_path=root / "planner_evidence_bundle.json",
            )
            layout = build_planner_paper_layout_bundle(
                profile_name="unit",
                bundle=bundle,
                output_root=root / "layout",
            )

            self.assertTrue((root / "layout" / "planner_paper_layout_bundle.json").exists())
            self.assertTrue((root / "layout" / "planner_paper_layout.md").exists())
            self.assertTrue((root / "layout" / "planner_caption_manifest.md").exists())
            self.assertTrue(any(path.endswith("planner_topk_efficiency.svg") for path in layout.main_figures))
            self.assertTrue(any(path.endswith("planner_failure_pressure.svg") for path in layout.main_figures))
            self.assertTrue(any(path.endswith("planner_rollout_claim_audit.svg") for path in layout.appendix_figures))
            self.assertTrue(any(path.endswith("planner_efficiency_synthesis.csv") for path in layout.main_tables))
            self.assertTrue(any(path.endswith("planner_rollout_claim_audit.csv") for path in layout.appendix_tables))
            self.assertIn("planner_topk_efficiency.svg", layout.main_figure_captions)
            self.assertIn("planner_rollout_claim_audit.svg", layout.appendix_figure_captions)
            self.assertIn("planner_efficiency_synthesis.csv", layout.main_table_captions)
            self.assertIn("planner_rollout_claim_audit.csv", layout.appendix_table_captions)


if __name__ == "__main__":
    unittest.main()
