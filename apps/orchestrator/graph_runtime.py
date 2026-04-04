"""Helpers for running dependency graphs of agent tasks."""

from __future__ import annotations

from collections import defaultdict, deque


def topological_layers(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Return dependency layers for a directed acyclic graph."""
    incoming: dict[str, int] = defaultdict(int)
    outgoing: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    for src, dst in edges:
        nodes.update({src, dst})
        outgoing[src].append(dst)
        incoming[dst] += 1
        incoming.setdefault(src, 0)

    queue = deque(sorted(node for node in nodes if incoming[node] == 0))
    layers: list[list[str]] = []
    while queue:
        layer: list[str] = []
        for _ in range(len(queue)):
            node = queue.popleft()
            layer.append(node)
            for child in outgoing[node]:
                incoming[child] -= 1
                if incoming[child] == 0:
                    queue.append(child)
        layers.append(layer)
    return layers
