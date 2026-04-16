"""Focused tests for the open-PDK readiness workflow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.check_open_pdk_ready import build_status


class OpenPdkReadinessTests(unittest.TestCase):
    def test_missing_root_is_reported_structurally(self) -> None:
        with mock.patch("scripts.check_open_pdk_ready.configured_pdk_root", return_value=None):
            with mock.patch("scripts.check_open_pdk_ready.external_model_card_path", return_value=None):
                status = build_status()

        self.assertFalse(status["ready"])
        self.assertEqual(status["readiness_state"], "missing_root")
        self.assertFalse(status["pdk_root_present"])
        self.assertTrue(status["missing_required_subpaths"])
        self.assertIn("stage_or_mount_pdk_root_at:/pdk/sky130A", status["recommended_actions"])

    def test_partial_root_is_classified_as_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "libs.tech" / "ngspice").mkdir(parents=True, exist_ok=True)
            with mock.patch("scripts.check_open_pdk_ready.configured_pdk_root", return_value=root):
                with mock.patch("scripts.check_open_pdk_ready.external_model_card_path", return_value=None):
                    status = build_status()

        self.assertFalse(status["ready"])
        self.assertEqual(status["readiness_state"], "partial")
        self.assertTrue(status["pdk_root_present"])
        self.assertIn("libs.ref/sky130_fd_pr/spice", status["missing_required_subpaths"])

    def test_complete_root_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "libs.tech" / "ngspice").mkdir(parents=True, exist_ok=True)
            (root / "libs.ref" / "sky130_fd_pr" / "spice").mkdir(parents=True, exist_ok=True)
            card = root / "sky130_tt.spice"
            card.write_text("* external model\n", encoding="utf-8")
            with mock.patch("scripts.check_open_pdk_ready.configured_pdk_root", return_value=root):
                with mock.patch("scripts.check_open_pdk_ready.external_model_card_path", return_value=card):
                    status = build_status()

        self.assertTrue(status["ready"])
        self.assertEqual(status["readiness_state"], "ready")
        self.assertEqual(status["missing_required_subpaths"], [])
        self.assertTrue(status["external_model_card_present"])
        self.assertIn("external_model_card_present", status["recommended_actions"])


if __name__ == "__main__":
    unittest.main()
