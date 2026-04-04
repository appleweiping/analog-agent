"""Graph neural network surrogate placeholder."""

from __future__ import annotations


class GNNSurrogate:
    """Predict outcomes from topology-aware graph inputs."""

    def predict(self, graph_features: dict) -> dict:
        return {"model": "gnn", "status": "stub", "graph_features": graph_features}
