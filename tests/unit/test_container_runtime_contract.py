"""Focused tests for the container-first ngspice runtime contract."""

from __future__ import annotations

import unittest

from scripts.check_container_runtime import build_status


class ContainerRuntimeContractTests(unittest.TestCase):
    def test_container_runtime_contract_is_structurally_ready(self) -> None:
        status = build_status()

        self.assertTrue(status["service_present"])
        self.assertTrue(status["ready"])
        self.assertTrue(all(status["env_checks"].values()))
        self.assertTrue(all(status["volume_checks"].values()))
        self.assertTrue(all(status["dockerfile_checks"].values()))
        self.assertTrue(all(status["command_checks"].values()))

    def test_container_contract_matches_expected_paths(self) -> None:
        status = build_status()

        self.assertEqual(status["container_contract"]["workdir"], "/workspace")
        self.assertEqual(status["container_contract"]["ngspice_bin"], "/usr/bin/ngspice")
        self.assertEqual(status["container_contract"]["pdk_root"], "/pdk/sky130A")


if __name__ == "__main__":
    unittest.main()
