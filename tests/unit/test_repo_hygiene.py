from __future__ import annotations

from pathlib import Path
import re
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]


class RepoHygieneTests(unittest.TestCase):
    def test_agents_md_exists_and_mentions_local_output_boundaries(self) -> None:
        text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

        self.assertIn("Default Reading Set", text)
        self.assertIn("Complex Task Rule", text)
        self.assertIn("archive/", text)
        self.assertIn(".artifacts/", text)
        self.assertIn("Completion Report", text)

    def test_repo_map_references_existing_tracked_paths(self) -> None:
        text = (REPO_ROOT / "docs" / "repo-map.md").read_text(encoding="utf-8")
        references = sorted(set(re.findall(r"`([^`]+/)`", text)))

        self.assertIn("apps/api_server/", references)
        self.assertIn("libs/schema/", references)
        for reference in references:
            if reference in {"archive/", ".artifacts/", ".pdk/"}:
                continue
            self.assertTrue((REPO_ROOT / reference).exists(), reference)

    def test_hygiene_docs_do_not_require_ignored_research_outputs(self) -> None:
        docs = [
            REPO_ROOT / "AGENTS.md",
            REPO_ROOT / "docs" / "repo-map.md",
            REPO_ROOT / "docs" / "stop_conditions.md",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in docs)

        self.assertIn("archive/", combined)
        self.assertNotIn("must read `research/", combined)
        self.assertNotIn("must read `paper/", combined)

    def test_legacy_project_tree_snapshot_is_removed(self) -> None:
        self.assertFalse((REPO_ROOT / "project_tree").exists())


if __name__ == "__main__":
    unittest.main()
