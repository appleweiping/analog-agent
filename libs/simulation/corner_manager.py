"""Corner management helpers."""

from __future__ import annotations


DEFAULT_CORNERS = ["tt", "ss", "ff"]


def select_corners(include_extremes: bool = True) -> list[str]:
    """Return baseline PVT corners."""
    return DEFAULT_CORNERS if include_extremes else ["tt"]
