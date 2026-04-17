"""Focused tests for configured-truth readiness and user-action boundary review."""

from __future__ import annotations

import unittest
from unittest import mock

from scripts.check_configured_truth_readiness import build_status as build_configured_truth_status
from scripts.explain_user_action_boundary import build_status as build_user_boundary_status


class ConfiguredTruthReadinessTests(unittest.TestCase):
    def test_disabled_mode_reports_demonstrator_only(self) -> None:
        fake_config = {
            "configured_truth_mode": "disabled",
            "configured_truth_contract": "sky130_open",
            "configured_truth_model_source": "external_model_card_or_pdk_root",
        }
        with mock.patch("scripts.check_configured_truth_readiness._load_ngspice_config", return_value=fake_config):
            with mock.patch(
                "scripts.check_configured_truth_readiness.build_open_pdk_status",
                return_value={"ready": False, "missing_required_subpaths": ["libs.tech/ngspice"], "recommended_actions": []},
            ):
                with mock.patch(
                    "scripts.check_configured_truth_readiness.build_container_status",
                    return_value={"ready": True, "docker_cli_available": False},
                ):
                    with mock.patch("scripts.check_configured_truth_readiness.configured_pdk_root", return_value=None):
                        with mock.patch("scripts.check_configured_truth_readiness.external_model_card_path", return_value=None):
                            status = build_configured_truth_status()

        self.assertEqual(status["readiness_state"], "demonstrator_only")
        self.assertEqual(status["user_actions_required"], [])

    def test_missing_inputs_report_not_ready(self) -> None:
        fake_config = {
            "configured_truth_mode": "external_pdk_root",
            "configured_truth_contract": "sky130_open",
            "configured_truth_model_source": "external_model_card_or_pdk_root",
        }
        with mock.patch("scripts.check_configured_truth_readiness._load_ngspice_config", return_value=fake_config):
            with mock.patch(
                "scripts.check_configured_truth_readiness.build_open_pdk_status",
                return_value={
                    "ready": False,
                    "missing_required_subpaths": ["libs.tech/ngspice", "libs.ref/sky130_fd_pr/spice"],
                    "recommended_actions": ["populate_required_subpath:libs.tech/ngspice"],
                },
            ):
                with mock.patch(
                    "scripts.check_configured_truth_readiness.build_container_status",
                    return_value={"ready": False, "docker_cli_available": False},
                ):
                    with mock.patch("scripts.check_configured_truth_readiness.configured_pdk_root", return_value=None):
                        with mock.patch("scripts.check_configured_truth_readiness.external_model_card_path", return_value=None):
                            status = build_configured_truth_status()

        self.assertEqual(status["readiness_state"], "configured_truth_not_ready")
        self.assertIn("external_model_card_missing", status["claim_blockers"])
        self.assertIn("stage_or_mount_structured_pdk_root", status["user_actions_required"])

    def test_user_boundary_exposes_current_required_actions(self) -> None:
        with mock.patch(
            "scripts.explain_user_action_boundary.build_configured_truth_status",
            return_value={
                "user_actions_required": ["stage_or_mount_structured_pdk_root", "supply_external_model_card"],
                "readiness_state": "configured_truth_not_ready",
            },
        ):
            status = build_user_boundary_status()

        self.assertIn("artifact_replay_manifest", status["repo_managed_capabilities"])
        self.assertEqual(status["current_claim_state"], "configured_truth_not_ready")
        self.assertIn("supply_external_model_card", status["current_user_actions_required"])


if __name__ == "__main__":
    unittest.main()
