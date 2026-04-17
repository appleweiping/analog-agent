from __future__ import annotations

import unittest

from libs.eval.submission_package import build_stage_e_review_status


class StageEReviewTests(unittest.TestCase):
    def test_stage_e_review_reports_ready_for_stage_f(self) -> None:
        status = build_stage_e_review_status()

        self.assertEqual(status["stage"], "Stage E")
        self.assertTrue(status["benchmark_contract_ready"])
        self.assertTrue(status["scripts_ready"])
        self.assertTrue(status["artifact_bundles_present"])
        self.assertTrue(status["ready_for_stage_f"])


if __name__ == "__main__":
    unittest.main()
