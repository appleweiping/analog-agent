from __future__ import annotations

from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]


class AgentHarnessDocsTests(unittest.TestCase):
    def test_agents_md_names_canonical_windows_test_command(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

        self.assertIn(r".\scripts\run_test_suite.ps1 -UseVenv -RequireApiDeps", readme)
        self.assertIn("scripts/run_system_closure_report.py", agents)

    def test_agents_md_warns_against_expanding_claims(self) -> None:
        text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

        self.assertIn("Do not claim signoff", text)
        self.assertIn("SPICE/configured-truth", text)
        self.assertIn("lightweight internal baselines", text)


if __name__ == "__main__":
    unittest.main()
