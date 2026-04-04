"""Budget accounting helpers."""

from __future__ import annotations


def remaining_budget(total: int, used: int) -> int:
    """Return the unused portion of a discrete budget."""
    return max(total - used, 0)
