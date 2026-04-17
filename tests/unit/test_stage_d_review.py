from __future__ import annotations

import unittest

from scripts.review_stage_d_planner_package import build_status


class StageDReviewTests(unittest.TestCase):
    def test_stage_d_review_reports_planner_package_readiness(self) -> None:
        status = build_status()

        self.assertEqual(status["stage"], "Stage D")
        self.assertIn(status["stage_status"], {"stage_d_planner_package_complete", "stage_d_planner_package_incomplete"})
        self.assertIsInstance(status["planner_summary_ready"], bool)
        self.assertIsInstance(status["planner_layout_ready"], bool)
        self.assertIsInstance(status["scripts_ready"], bool)
        self.assertIsInstance(status["ready_for_stage_e"], bool)
        self.assertTrue(status["expected_scripts"])
        self.assertTrue(status["notes"])


if __name__ == "__main__":
    unittest.main()
