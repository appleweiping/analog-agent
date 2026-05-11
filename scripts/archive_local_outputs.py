"""Archive ignored local workspaces into the repository archive sink."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "archive" / "legacy"
DEFAULT_LOCAL_ROOTS = ["paper", "research"]


def _resolve_repo_path(path: str | Path) -> Path:
    resolved = (REPO_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not (resolved == REPO_ROOT or REPO_ROOT in resolved.parents):
        raise ValueError(f"path is outside repository: {path}")
    return resolved


def _safe_relative(path: Path) -> Path:
    return path.resolve().relative_to(REPO_ROOT)


def build_archive_plan(
    roots: list[str] | None = None,
    *,
    archive_root: str | Path = DEFAULT_ARCHIVE_ROOT,
) -> dict[str, object]:
    """Build a move plan for ignored local output roots."""

    selected_roots = roots or list(DEFAULT_LOCAL_ROOTS)
    archive_path = _resolve_repo_path(archive_root)
    moves: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for root in selected_roots:
        source = _resolve_repo_path(root)
        if not source.exists():
            skipped.append({"source": str(_safe_relative(source)), "reason": "missing"})
            continue
        if source == archive_path or archive_path in source.parents:
            skipped.append({"source": str(_safe_relative(source)), "reason": "already_inside_archive"})
            continue
        if source.parts[len(REPO_ROOT.parts)] == ".git":
            raise ValueError("refusing to archive .git")
        destination = archive_path / _safe_relative(source)
        moves.append({"source": str(_safe_relative(source)), "destination": str(_safe_relative(destination))})
    return {
        "archive_root": str(_safe_relative(archive_path)),
        "moves": moves,
        "skipped": skipped,
    }


def execute_archive_plan(plan: dict[str, object], *, dry_run: bool = True) -> dict[str, object]:
    """Execute a previously built archive plan."""

    executed: list[dict[str, str]] = []
    for item in list(plan.get("moves", [])):
        source = _resolve_repo_path(str(item["source"]))
        destination = _resolve_repo_path(str(item["destination"]))
        if dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            destination = destination.with_name(f"{destination.name}_{stamp}")
        shutil.move(str(source), str(destination))
        executed.append({"source": str(_safe_relative(source)), "destination": str(_safe_relative(destination))})
    return {
        **plan,
        "dry_run": dry_run,
        "executed": executed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", dest="roots", help="Local root to archive. Can be repeated.")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    parser.add_argument("--execute", action="store_true", help="Move files. Omit for dry-run.")
    args = parser.parse_args()

    plan = build_archive_plan(args.roots, archive_root=args.archive_root)
    result = execute_archive_plan(plan, dry_run=not args.execute)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
