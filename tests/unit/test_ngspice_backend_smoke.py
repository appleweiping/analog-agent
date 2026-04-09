"""Backend execution smoke tests for the native ngspice path."""

from __future__ import annotations

import unittest
from pathlib import Path

from apps.worker_simulator.ngspice_runner import native_ngspice_available, run_ngspice_batch


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class NgspiceBackendSmokeTests(unittest.TestCase):
    def test_minimal_batch_smoke_completes(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        netlist = repo_root / "scripts" / "smoke" / "test_ngspice_min.cir"
        log_path = repo_root / ".artifacts" / "ngspice_logs" / "test_ngspice_min.unit.log"
        result = run_ngspice_batch(netlist, log_path, timeout_sec=20)

        self.assertTrue(result["ok"])
        self.assertEqual(result["error_type"], "none")
        self.assertTrue(result["log_exists"])
        self.assertEqual(result["returncode"], 0)

    def test_missing_netlist_surfaces_structured_error(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        missing = repo_root / "scripts" / "smoke" / "does_not_exist.cir"
        log_path = repo_root / ".artifacts" / "ngspice_logs" / "missing.log"
        result = run_ngspice_batch(missing, log_path, timeout_sec=5)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "netlist_error")

