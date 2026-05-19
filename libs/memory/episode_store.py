"""Episode store: persistent storage for optimization episodes backed by JSON."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class EpisodeRecord:
    """A single optimization episode record."""

    episode_id: str
    circuit_family: str
    timestamp: str
    parameters: dict[str, float]
    metrics: dict[str, float]
    feasible: bool
    constraints_violated: list[str] = field(default_factory=list)
    notes: str = ""


class EpisodeStore:
    """Persistent episode store backed by a JSON file.

    Stores optimization episodes (parameter -> result history) and provides
    query methods for retrieving history, best designs, and feasible regions.
    """

    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)
        self._episodes: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load episodes from disk if the file exists."""
        if self.store_path.exists():
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._episodes = data.get("episodes", [])
        else:
            self._episodes = []

    def _save(self) -> None:
        """Persist episodes to disk."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump({"episodes": self._episodes, "version": "1.0"}, f, indent=2)

    def record_episode(
        self,
        params: dict[str, float],
        metrics: dict[str, float],
        feasible: bool,
        circuit_family: str = "two_stage_ota",
        constraints_violated: list[str] | None = None,
        notes: str = "",
    ) -> str:
        """Record a new optimization episode.

        Args:
            params: Design parameter values (e.g., {"w_in": 5e-6, "l_in": 0.5e-6}).
            metrics: Measured/simulated metrics (e.g., {"dc_gain_db": 65.2, "gbw_hz": 100e6}).
            feasible: Whether all constraints were satisfied.
            circuit_family: Circuit family identifier.
            constraints_violated: List of constraint names that were violated.
            notes: Optional notes about this episode.

        Returns:
            The episode_id of the recorded episode.
        """
        episode_id = f"ep_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{len(self._episodes):04d}"
        record = {
            "episode_id": episode_id,
            "circuit_family": circuit_family,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parameters": params,
            "metrics": metrics,
            "feasible": feasible,
            "constraints_violated": constraints_violated or [],
            "notes": notes,
        }
        self._episodes.append(record)
        self._save()
        return episode_id

    def get_history(self, circuit_family: str | None = None) -> list[dict[str, Any]]:
        """Get episode history, optionally filtered by circuit family.

        Args:
            circuit_family: If provided, only return episodes for this family.

        Returns:
            List of episode records (most recent last).
        """
        if circuit_family is None:
            return list(self._episodes)
        return [ep for ep in self._episodes if ep.get("circuit_family") == circuit_family]

    def get_best_designs(self, n: int = 5, metric: str | None = None, maximize: bool = True) -> list[dict[str, Any]]:
        """Get the top N designs by a given metric (feasible only).

        Args:
            n: Number of top designs to return.
            metric: Metric name to rank by. If None, returns best feasible by first metric.
            maximize: If True, higher is better; if False, lower is better.

        Returns:
            List of top N episode records sorted by the metric.
        """
        feasible_episodes = [ep for ep in self._episodes if ep.get("feasible")]
        if not feasible_episodes:
            return []

        if metric is None:
            # Default: use first metric key from the first episode
            if feasible_episodes[0].get("metrics"):
                metric = next(iter(feasible_episodes[0]["metrics"]))
            else:
                return feasible_episodes[:n]

        # Filter to episodes that have this metric
        with_metric = [ep for ep in feasible_episodes if metric in ep.get("metrics", {})]
        with_metric.sort(key=lambda ep: ep["metrics"][metric], reverse=maximize)
        return with_metric[:n]

    def get_feasible_region_bounds(self) -> dict[str, dict[str, float]]:
        """Compute parameter bounds from all feasible episodes.

        Returns:
            Dict mapping parameter names to {"min": ..., "max": ..., "mean": ...}.
        """
        feasible_episodes = [ep for ep in self._episodes if ep.get("feasible")]
        if not feasible_episodes:
            return {}

        # Collect all parameter values across feasible episodes
        param_values: dict[str, list[float]] = {}
        for ep in feasible_episodes:
            for param_name, value in ep.get("parameters", {}).items():
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append(float(value))

        bounds: dict[str, dict[str, float]] = {}
        for param_name, values in param_values.items():
            bounds[param_name] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "count": len(values),
            }
        return bounds

    def get_feasible_count(self) -> int:
        """Return the number of feasible episodes."""
        return sum(1 for ep in self._episodes if ep.get("feasible"))

    def get_total_count(self) -> int:
        """Return the total number of episodes."""
        return len(self._episodes)

    def get_feasibility_rate(self) -> float:
        """Return the fraction of episodes that are feasible."""
        total = len(self._episodes)
        if total == 0:
            return 0.0
        return self.get_feasible_count() / total

    def clear(self) -> None:
        """Clear all episodes (use with caution)."""
        self._episodes = []
        self._save()
