"""Pattern mining: analyze stored episodes to extract design patterns."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeasibleBound:
    """Parameter bounds observed in feasible designs."""

    param_name: str
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    sample_count: int


@dataclass
class WarmStartPoint:
    """A suggested starting point for optimization."""

    parameters: dict[str, float]
    source_episode_id: str
    score: float  # quality score (higher is better)
    reason: str


@dataclass
class SensitivityEntry:
    """Sensitivity ranking entry for a single parameter."""

    param_name: str
    sensitivity_score: float  # higher means more impact on feasibility
    feasible_range_ratio: float  # fraction of total range that is feasible
    correlation_with_feasibility: float  # positive = higher value helps


class DesignPatternMiner:
    """Analyze stored episodes to extract design patterns and insights.

    Works with the EpisodeStore's raw episode data (list of dicts) to find:
    - Feasible parameter bounds
    - Good warm-start points
    - Parameter sensitivity rankings
    """

    def __init__(self, episodes: list[dict[str, Any]]) -> None:
        self.episodes = episodes
        self._feasible = [ep for ep in episodes if ep.get("feasible")]
        self._infeasible = [ep for ep in episodes if not ep.get("feasible")]

    def find_feasible_bounds(self) -> list[FeasibleBound]:
        """Compute parameter ranges that tend to produce feasible designs.

        Returns:
            List of FeasibleBound for each parameter observed in feasible episodes.
        """
        if not self._feasible:
            return []

        param_values: dict[str, list[float]] = {}
        for ep in self._feasible:
            for param_name, value in ep.get("parameters", {}).items():
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append(float(value))

        bounds: list[FeasibleBound] = []
        for param_name, values in param_values.items():
            if len(values) < 2:
                std = 0.0
            else:
                std = statistics.stdev(values)
            bounds.append(
                FeasibleBound(
                    param_name=param_name,
                    min_value=min(values),
                    max_value=max(values),
                    mean_value=statistics.mean(values),
                    std_value=std,
                    sample_count=len(values),
                )
            )

        return bounds

    def suggest_warm_start(self, n: int = 5) -> list[WarmStartPoint]:
        """Suggest N good starting points based on history.

        Strategy:
        1. Take the best feasible designs (by number of metrics met)
        2. If not enough feasible, include near-feasible (fewest violations)
        3. Diversify by picking designs that are spread across parameter space

        Args:
            n: Number of warm-start points to suggest.

        Returns:
            List of WarmStartPoint suggestions.
        """
        suggestions: list[WarmStartPoint] = []

        # Score feasible episodes by metric quality
        scored_feasible: list[tuple[float, dict]] = []
        for ep in self._feasible:
            metrics = ep.get("metrics", {})
            # Simple quality score: number of metrics with positive values
            # (heuristic - real scoring would use spec targets)
            score = sum(1.0 for v in metrics.values() if v > 0) / max(1, len(metrics))
            scored_feasible.append((score, ep))

        scored_feasible.sort(key=lambda x: x[0], reverse=True)

        for score, ep in scored_feasible[:n]:
            suggestions.append(
                WarmStartPoint(
                    parameters=dict(ep.get("parameters", {})),
                    source_episode_id=ep.get("episode_id", "unknown"),
                    score=score,
                    reason="feasible design with good metric coverage",
                )
            )

        # If we need more, add near-feasible (fewest constraint violations)
        if len(suggestions) < n and self._infeasible:
            near_feasible = sorted(
                self._infeasible,
                key=lambda ep: len(ep.get("constraints_violated", [])),
            )
            for ep in near_feasible[: n - len(suggestions)]:
                violations = len(ep.get("constraints_violated", []))
                score = max(0.0, 1.0 - violations * 0.2)
                suggestions.append(
                    WarmStartPoint(
                        parameters=dict(ep.get("parameters", {})),
                        source_episode_id=ep.get("episode_id", "unknown"),
                        score=score,
                        reason=f"near-feasible ({violations} violations)",
                    )
                )

        return suggestions[:n]

    def get_sensitivity_ranking(self) -> list[SensitivityEntry]:
        """Rank parameters by their impact on feasibility.

        Uses a simple heuristic: compare parameter distributions between
        feasible and infeasible episodes. Parameters with large distribution
        shifts are more sensitive.

        Returns:
            List of SensitivityEntry sorted by sensitivity (highest first).
        """
        if not self._feasible or not self._infeasible:
            return []

        # Collect parameter values for feasible and infeasible
        feasible_params: dict[str, list[float]] = {}
        infeasible_params: dict[str, list[float]] = {}

        for ep in self._feasible:
            for name, value in ep.get("parameters", {}).items():
                if name not in feasible_params:
                    feasible_params[name] = []
                feasible_params[name].append(float(value))

        for ep in self._infeasible:
            for name, value in ep.get("parameters", {}).items():
                if name not in infeasible_params:
                    infeasible_params[name] = []
                infeasible_params[name].append(float(value))

        # Compute sensitivity for parameters present in both sets
        entries: list[SensitivityEntry] = []
        common_params = set(feasible_params.keys()) & set(infeasible_params.keys())

        for param_name in common_params:
            f_values = feasible_params[param_name]
            i_values = infeasible_params[param_name]

            f_mean = statistics.mean(f_values)
            i_mean = statistics.mean(i_values)

            # Total range across all episodes
            all_values = f_values + i_values
            total_range = max(all_values) - min(all_values)
            if total_range == 0:
                continue

            # Sensitivity: normalized mean shift between feasible and infeasible
            mean_shift = abs(f_mean - i_mean) / total_range

            # Feasible range ratio
            f_range = max(f_values) - min(f_values)
            feasible_range_ratio = f_range / total_range if total_range > 0 else 1.0

            # Correlation direction: positive if higher values correlate with feasibility
            correlation = (f_mean - i_mean) / total_range

            entries.append(
                SensitivityEntry(
                    param_name=param_name,
                    sensitivity_score=mean_shift,
                    feasible_range_ratio=feasible_range_ratio,
                    correlation_with_feasibility=correlation,
                )
            )

        entries.sort(key=lambda e: e.sensitivity_score, reverse=True)
        return entries

    def get_common_failure_modes(self) -> dict[str, int]:
        """Count which constraints are most commonly violated.

        Returns:
            Dict mapping constraint name to violation count, sorted descending.
        """
        violation_counts: dict[str, int] = {}
        for ep in self._infeasible:
            for constraint in ep.get("constraints_violated", []):
                violation_counts[constraint] = violation_counts.get(constraint, 0) + 1

        return dict(sorted(violation_counts.items(), key=lambda x: x[1], reverse=True))
