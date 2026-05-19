"""
Source Critic: Adversarial surrogate calibration.

Trains a discriminator (Critic) to identify regions where the World Model
(XGBoost surrogate) is inaccurate. The optimizer then avoids these regions,
preventing "phantom optima" — points that look good to the surrogate but
fail in real simulation.

Architecture:
  G = World Model (XGBoost, frozen) — predicts circuit performance
  D = Source Critic (MLP) — classifies predictions as trustworthy or not

Training:
  - Positive samples: (params, prediction) where |prediction - truth| < threshold
  - Negative samples: (params, prediction) where |prediction - truth| > threshold
  - D learns to output trust_score in [0, 1]

Usage in optimization:
  - Acquisition function is penalized by (1 - trust_score)
  - Optimizer avoids regions where D says surrogate is unreliable
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CriticConfig:
    hidden_dims: list[int] = None
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 64
    trust_threshold: float = 0.15  # relative error threshold for "trustworthy"
    dropout: float = 0.1

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 64]


class CriticNetwork(nn.Module):
    """MLP discriminator that scores surrogate prediction trustworthiness."""

    def __init__(self, input_dim: int, config: CriticConfig):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SourceCritic:
    """
    Adversarial calibration for the World Model surrogate.
    
    Learns which regions of design space have trustworthy surrogate predictions
    and which regions are unreliable (high prediction error).
    """

    def __init__(self, config: CriticConfig | None = None):
        self.config = config or CriticConfig()
        self.network: CriticNetwork | None = None
        self.input_scaler: dict | None = None  # min/max for normalization
        self.is_trained = False

    def train(self, params_array: np.ndarray, predictions: np.ndarray,
              truths: np.ndarray, metric_names: list[str]) -> dict:
        """
        Train the critic on (params, surrogate_prediction, ngspice_truth) triples.
        
        Args:
            params_array: (N, D) design parameters
            predictions: (N, M) surrogate predictions for M metrics
            truths: (N, M) ngspice ground truth for M metrics
            metric_names: names of the M metrics
            
        Returns:
            Training stats dict
        """
        # Compute relative errors per metric
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_errors = np.abs(predictions - truths) / (np.abs(truths) + 1e-10)
        max_rel_error = np.max(rel_errors, axis=1)  # worst metric per sample

        # Labels: 1 = trustworthy (low error), 0 = untrustworthy (high error)
        labels = (max_rel_error < self.config.trust_threshold).astype(np.float32)

        # Input features: params + predictions (so critic sees what surrogate predicted)
        features = np.hstack([params_array, predictions])
        input_dim = features.shape[1]

        # Normalize features
        self.input_scaler = {
            'min': features.min(axis=0),
            'max': features.max(axis=0),
        }
        features_norm = self._normalize(features)

        # Convert to tensors
        X = torch.FloatTensor(features_norm)
        y = torch.FloatTensor(labels).unsqueeze(1)

        # Build network
        self.network = CriticNetwork(input_dim, self.config)
        optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss()

        # Training loop
        self.network.train()
        n_samples = len(X)
        losses = []

        for epoch in range(self.config.epochs):
            # Shuffle
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.config.batch_size):
                batch_idx = perm[i:i + self.config.batch_size]
                batch_x, batch_y = X[batch_idx], y[batch_idx]

                optimizer.zero_grad()
                pred = self.network(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            losses.append(epoch_loss / n_batches)

        self.network.eval()
        self.is_trained = True

        # Compute final accuracy
        with torch.no_grad():
            all_pred = self.network(X)
            accuracy = ((all_pred > 0.5).float() == y).float().mean().item()

        trustworthy_ratio = labels.mean()

        return {
            "accuracy": accuracy,
            "trustworthy_ratio": float(trustworthy_ratio),
            "final_loss": losses[-1],
            "epochs": self.config.epochs,
            "samples": n_samples,
        }

    def score_trust(self, params: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Score how trustworthy the surrogate predictions are for given params.
        
        Args:
            params: (N, D) or (D,) design parameters
            predictions: (N, M) or (M,) surrogate predictions
            
        Returns:
            trust_scores: (N,) values in [0, 1], higher = more trustworthy
        """
        if not self.is_trained:
            # Before training, return neutral trust
            if params.ndim == 1:
                return np.array([0.5])
            return np.full(len(params), 0.5)

        if params.ndim == 1:
            params = params.reshape(1, -1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)

        features = np.hstack([params, predictions])
        features_norm = self._normalize(features)

        with torch.no_grad():
            X = torch.FloatTensor(features_norm)
            scores = self.network(X).numpy().flatten()

        return scores

    def penalized_acquisition(self, acquisition_value: float, trust_score: float,
                               penalty_weight: float = 2.0) -> float:
        """Apply trust-based penalty to acquisition function value."""
        # Penalize untrusted regions: multiply acquisition by trust^penalty_weight
        return acquisition_value * (trust_score ** penalty_weight)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Min-max normalize using stored scaler."""
        if self.input_scaler is None:
            return features
        range_vals = self.input_scaler['max'] - self.input_scaler['min']
        range_vals[range_vals == 0] = 1.0
        return (features - self.input_scaler['min']) / range_vals

    def save(self, path: Path):
        """Save trained critic to disk."""
        path.mkdir(parents=True, exist_ok=True)
        if self.network:
            torch.save(self.network.state_dict(), path / "critic_weights.pt")
        if self.input_scaler:
            np.savez(path / "critic_scaler.npz",
                     min=self.input_scaler['min'], max=self.input_scaler['max'])

    def load(self, path: Path, input_dim: int):
        """Load trained critic from disk."""
        self.network = CriticNetwork(input_dim, self.config)
        self.network.load_state_dict(torch.load(path / "critic_weights.pt", weights_only=True))
        self.network.eval()
        scaler_data = np.load(path / "critic_scaler.npz")
        self.input_scaler = {'min': scaler_data['min'], 'max': scaler_data['max']}
        self.is_trained = True
