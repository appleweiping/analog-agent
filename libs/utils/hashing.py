"""Hashing helpers."""

from __future__ import annotations

import hashlib


def stable_hash(text: str) -> str:
    """Return a deterministic SHA-256 hash for text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
