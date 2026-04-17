from __future__ import annotations

import unittest

from scripts.review_stage_b_truth_groundwork import build_status


class StageBReviewTests(unittest.TestCase):
    def test_stage_b_review_is_structured(self) -> None:
        status = build_status()
        self.assertEqual(status["stage"], "Stage B")
        self.assertIn(status["stage_status"], {"stage_b_groundwork_complete", "configured_truth_candidate_ready", "configured_truth_ready", "configured_truth_still_blocked"})
        self.assertIn("configured_truth_state", status)
        self.assertIn("artifact_rerunability", status)
        self.assertTrue(status["ready_for_stage_c"])


if __name__ == "__main__":
    unittest.main()
