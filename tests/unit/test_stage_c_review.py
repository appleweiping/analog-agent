from __future__ import annotations

import unittest

from scripts.review_stage_c_world_model_readiness import build_status


class StageCReviewTests(unittest.TestCase):
    def test_stage_c_review_reports_world_model_readiness(self) -> None:
        status = build_status()

        self.assertEqual(status["stage"], "Stage C")
        self.assertIn(status["stage_status"], {"stage_c_world_model_complete", "stage_c_world_model_incomplete"})
        self.assertIsInstance(status["training_reproducibility_ready"], bool)
        self.assertIsInstance(status["trained_serving_boundary_ready"], bool)
        self.assertIsInstance(status["paper_evidence_extended"], bool)
        self.assertIsInstance(status["ready_for_stage_d"], bool)
        self.assertTrue(status["expected_scripts"])
        self.assertTrue(status["notes"])


if __name__ == "__main__":
    unittest.main()
