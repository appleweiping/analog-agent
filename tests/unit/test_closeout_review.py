from __future__ import annotations

import unittest

from scripts.review_harness_closeout import build_status


class HarnessCloseoutReviewTests(unittest.TestCase):
    def test_closeout_review_reports_expected_docs_scripts_and_tests(self) -> None:
        status = build_status()

        self.assertEqual(status["stage"], "Harness Closeout")
        self.assertIn(status["closeout_status"], {"ready_for_writing", "need_more_evidence"})
        self.assertIn("AGENTS.md", status["required_docs"])
        self.assertIn("scripts/review_harness_closeout.py", status["required_review_scripts"])
        self.assertIn("AnalogGym", status["related_work"]["missing_anchors"] + ["AnalogGym"])
        self.assertIsInstance(status["ready_for_writing"], bool)
        self.assertIsInstance(status["blockers"], list)

    def test_closeout_review_treats_missing_ignored_outputs_as_non_blocking(self) -> None:
        status = build_status()

        self.assertNotIn("ignored_research_outputs_missing", status["blockers"])
        self.assertIn("benchmark_depth", status)
        self.assertIn("stop_decision", status)


if __name__ == "__main__":
    unittest.main()
