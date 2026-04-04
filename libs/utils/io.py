"""I/O helpers."""

from __future__ import annotations

from pathlib import Path

import yaml


def read_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
