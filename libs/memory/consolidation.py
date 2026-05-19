"""Memory consolidation: summarize, compress, and prune the episode store."""

from __future__ import annotations

import statistics
from typing import Any


class MemoryConsolidator:
    """Periodically summarize and compress the episode store.

    Provides methods to:
    - Merge similar episodes (keep representative ones)
    - Retain only Pareto-optimal designs
    - Forget low-quality episodes below a threshold
    """

    def __init__(self, episodes: list[dict[str, Any]]) -> None:
        self._episodes = list(episodes)

    @property
    def episodes(self) -> list[dict[str, Any]]:
        """Current episode list after consolidation."""
        return self._episodes

    def consolidate(
        self,
        similarity_threshold: float = 0.05,
        keep_pareto: bool = True,
    ) -> list[dict[str, Any]]:
        """Merge similar episodes and keep only Pareto-optimal feasible designs.

        Args:
            similarity_threshold: Relative parameter distance below which
                episodes are considered duplicates (0.05 = 5% of range).
            keep_pareto: If True, among feasible episodes, keep only
                Pareto-optimal ones (non-dominated in metric space).

        Returns:
            Consolidated list of episodes.
        """
        # Step 1: Deduplicate near-identical episodes
        deduplicated = self._deduplicate(self._episodes, similarity_threshold)

        # Step 2: Among feasible, keep Pareto-optimal
        if keep_pareto:
            feasible = [ep for ep in deduplicated if ep.get("feasible")]
            infeasible = [ep for ep in deduplicated if not ep.get("feasible")]

            if len(feasible) > 2:
                pareto_feasible = self._pareto_filter(feasible)
            else:
                pareto_feasible = feasible

            # Keep a bounded number of infeasible for learning
            # (most informative: those with fewest violations)
            infeasible.sort(key=lambda ep: len(ep.get("constraints_violated", [])))
            max_infeasible = max(len(pareto_feasible), 20)
            kept_infeasible = infeasible[:max_infeasible]

            self._episodes = pareto_feasible + kept_infeasible
        else:
            self._episodes = deduplicated

        return self._episodes

    def forget(self, threshold: float = 0.3) -> list[dict[str, Any]]:
        """Remove low-quality episodes below a quality threshold.

        Quality is estimated as:
        - Feasible episodes: always kept (quality = 1.0)
        - Infeasible episodes: quality = 1.0 - (violations / total_constraints)

        Args:
            threshold: Episodes with quality below this are removed.

        Returns:
            Remaining episodes after forgetting.
        """
        kept: list[dict[str, Any]] = []
        for ep in self._episodes:
            if ep.get("feasible"):
                kept.append(ep)
                continue

            violations = len(ep.get("constraints_violated", []))
            metrics_count = len(ep.get("metrics", {}))
            # Estimate quality: fewer violations relative to metrics = higher quality
            if metrics_count > 0:
                quality = 1.0 - (violations / max(metrics_count, violations + 1))
            else:
                quality = 0.0

            if quality >= threshold:
                kept.append(ep)

        self._episodes = kept
        return self._episodes

    def get_summary(self) -> dict[str, Any]:
        """Generate a summary of the current episode store state.

        Returns:
            Dict with statistics about the stored episodes.
        """
        total = len(self._episodes)
        feasible = [ep for ep in self._episodes if ep.get("feasible")]
        infeasible = [ep for ep in self._episodes if not ep.get("feasible")]

        summary: dict[str, Any] = {
            "total_episodes": total,
            "feasible_count": len(feasible),
            "infeasible_count": len(infeasible),
            "feasibility_rate": len(feasible) / total if total > 0 else 0.0,
        }

        # Metric statistics for feasible designs
        if feasible:
            metric_stats: dict[str, dict[str, float]] = {}
            for ep in feasible:
                for name, value in ep.get("metrics", {}).items():
                    if name not in metric_stats:
                        metric_stats[name] = {"values": []}
                    metric_stats[name]["values"].append(float(value))

            for name, data in metric_stats.items():
                values = data["values"]
                metric_stats[name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "count": len(values),
                }
            summary["metric_stats"] = metric_stats

        # Common failure modes
        if infeasible:
            violation_counts: dict[str, int] = {}
            for ep in infeasible:
                for c in ep.get("constraints_violated", []):
                    violation_counts[c] = violation_counts.get(c, 0) + 1
            summary["top_violations"] = dict(
                sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            )

        return summary

    def _deduplicate(
        self, episodes: list[dict[str, Any]], threshold: float
    ) -> list[dict[str, Any]]:
        """Remove near-duplicate episodes based on parameter similarity."""
        if not episodes:
            return []

        # Compute parameter ranges for normalization
        param_ranges = self._compute_param_ranges(episodes)
        if not param_ranges:
            return episodes

        kept: list[dict[str, Any]] = []
        for ep in episodes:
            is_duplicate = False
            for existing in kept:
                if self._is_similar(ep, existing, param_ranges, threshold):
                    # Keep the one with better outcome
                    if ep.get("feasible") and not existing.get("feasible"):
                        kept.remove(existing)
                        kept.append(ep)
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(ep)

        return kept

    def _is_similar(
        self,
        ep1: dict[str, Any],
        ep2: dict[str, Any],
        param_ranges: dict[str, float],
        threshold: float,
    ) -> bool:
        """Check if two episodes have similar parameters."""
        params1 = ep1.get("parameters", {})
        params2 = ep2.get("parameters", {})

        common_params = set(params1.keys()) & set(params2.keys())
        if not common_params:
            return False

        total_distance = 0.0
        for param in common_params:
            range_val = param_ranges.get(param, 1.0)
            if range_val == 0:
                continue
            diff = abs(float(params1[param]) - float(params2[param])) / range_val
            total_distance += diff

        avg_distance = total_distance / len(common_params)
        return avg_distance < threshold

    def _compute_param_ranges(self, episodes: list[dict[str, Any]]) -> dict[str, float]:
        """Compute the range of each parameter across all episodes."""
        param_values: dict[str, list[float]] = {}
        for ep in episodes:
            for name, value in ep.get("parameters", {}).items():
                if name not in param_values:
                    param_values[name] = []
                param_values[name].append(float(value))

        return {
            name: max(values) - min(values)
            for name, values in param_values.items()
            if len(values) > 1
        }

    def _pareto_filter(self, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only Pareto-optimal episodes in metric space.

        An episode is Pareto-optimal if no other episode dominates it
        (i.e., is better in all metrics simultaneously).
        """
        if len(episodes) <= 1:
            return episodes

        # Get common metrics across all episodes
        all_metrics: set[str] = set()
        for ep in episodes:
            all_metrics.update(ep.get("metrics", {}).keys())

        if not all_metrics:
            return episodes

        metric_list = sorted(all_metrics)

        def get_metric_vector(ep: dict) -> list[float]:
            metrics = ep.get("metrics", {})
            return [float(metrics.get(m, 0.0)) for m in metric_list]

        def dominates(a: list[float], b: list[float]) -> bool:
            """Check if a dominates b (a >= b in all, a > b in at least one)."""
            at_least_one_better = False
            for va, vb in zip(a, b):
                if va < vb:
                    return False
                if va > vb:
                    at_least_one_better = True
            return at_least_one_better

        vectors = [get_metric_vector(ep) for ep in episodes]
        pareto: list[dict[str, Any]] = []

        for i, ep in enumerate(episodes):
            is_dominated = False
            for j, other_vec in enumerate(vectors):
                if i == j:
                    continue
                if dominates(other_vec, vectors[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto.append(ep)

        return pareto if pareto else episodes[:1]  # Always keep at least one
