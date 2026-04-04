"""Design case store placeholder."""

from __future__ import annotations


class DesignCaseStore:
    """Persist benchmark cases and design outcomes."""

    def write(self, case: dict) -> dict:
        return {"store": "design_case", "status": "stub", "case": case}
