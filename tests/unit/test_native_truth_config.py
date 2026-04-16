"""Focused tests for native-truth configuration resolution."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from apps.worker_simulator import ngspice_runner


class NativeTruthConfigTests(unittest.TestCase):
    def test_env_override_takes_precedence_for_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            fake_bin = Path(tempdir) / "ngspice.exe"
            fake_bin.write_text("", encoding="utf-8")
            with mock.patch.dict(os.environ, {"ANALOG_AGENT_NGSPICE_BIN": str(fake_bin)}, clear=False):
                resolved = ngspice_runner.ngspice_binary_path()
        self.assertEqual(resolved, fake_bin)

    def test_workspace_tool_candidate_is_discovered_without_machine_specific_config(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            expected = Path(tempdir) / "ngspice.exe"
            expected.write_text("", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=False):
                with mock.patch("apps.worker_simulator.ngspice_runner.shutil.which", return_value=None):
                    with mock.patch(
                        "apps.worker_simulator.ngspice_runner._workspace_tool_candidates",
                        return_value=[expected],
                    ):
                        resolved = ngspice_runner.ngspice_binary_path()
        self.assertEqual(resolved, expected)

    def test_external_model_card_can_be_resolved_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            fake_card = Path(tempdir) / "sky130_tt.spice"
            fake_card.write_text("* mock model card\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"ANALOG_AGENT_EXTERNAL_MODEL_CARD": str(fake_card)}, clear=False):
                resolved = ngspice_runner.external_model_card_path()
        self.assertEqual(resolved, fake_card)

    def test_configured_pdk_root_can_be_resolved_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            pdk_root = Path(tempdir)
            with mock.patch.dict(os.environ, {"ANALOG_AGENT_PDK_ROOT": str(pdk_root)}, clear=False):
                resolved = ngspice_runner.configured_pdk_root()
        self.assertEqual(resolved, pdk_root)
