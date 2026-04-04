"""Reproducibility helpers."""

from __future__ import annotations

import random


def seed_everything(seed: int) -> None:
    """Seed the standard library RNG as a minimal reproducibility hook."""
    random.seed(seed)
