from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.archive_local_outputs import REPO_ROOT, build_archive_plan, execute_archive_plan


class ArchivePolicyTests(unittest.TestCase):
    def test_archive_plan_only_selects_local_roots(self) -> None:
        plan = build_archive_plan(["paper", "research"])

        self.assertEqual(plan["archive_root"], "archive\\legacy" if "\\" in str(REPO_ROOT) else "archive/legacy")
        for move in plan["moves"]:
            self.assertIn(move["source"], {"paper", "research"})
            self.assertTrue(move["destination"].replace("\\", "/").startswith("archive/legacy/"))

    def test_archive_dry_run_does_not_move_files(self) -> None:
        with TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            temp_root = Path(tmpdir)
            local_dir = temp_root / "local_outputs"
            local_dir.mkdir()
            marker = local_dir / "marker.txt"
            marker.write_text("x", encoding="utf-8")

            plan = build_archive_plan([local_dir.relative_to(REPO_ROOT).as_posix()])
            result = execute_archive_plan(plan, dry_run=True)

            self.assertTrue(marker.exists())
            self.assertEqual(result["executed"], [])

    def test_archive_rejects_paths_outside_repo(self) -> None:
        with self.assertRaises(ValueError):
            build_archive_plan([str(REPO_ROOT.parent)])


if __name__ == "__main__":
    unittest.main()
