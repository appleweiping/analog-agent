from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from libs.eval.planner_evidence import build_planner_ablation_evidence_bundle
from scripts.review_planner_rollout_evidence import build_status
from tests.unit.test_planner_evidence import PlannerEvidenceUnitTests


class PlannerRolloutEvidenceReviewUnitTests(unittest.TestCase):
    def test_review_marks_observed_rollout_evidence_as_real(self) -> None:
        fixture = PlannerEvidenceUnitTests()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_planner_ablation_evidence_bundle(
                fixture._suite(),
                figures_dir=root / "figs",
                tables_dir=root / "tables",
                json_output_path=root / "planner_evidence_bundle.json",
            )
            status = build_status(bundle)

            self.assertEqual(status["evidence_state"], "real_observed_short_horizon_rollout_evidence")
            self.assertTrue(status["paper_safe"])
            self.assertFalse(status["placeholder_risk"])
            self.assertEqual(status["rollout_claim_scope"], "short_horizon_world_model_guidance")


if __name__ == "__main__":
    unittest.main()
