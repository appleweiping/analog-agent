"""
Pareto GAN: Generate designs on the Pareto frontier.

Instead of finding a single optimal point, generates an entire set of
Pareto-optimal designs spanning the tradeoff between competing objectives
(e.g., GBW vs Power).

Architecture (Conditional WGAN-GP with Pareto filtering):
  G: (tradeoff_weight, noise) -> design_parameters
  D: (design_parameters) -> "is this on the Pareto front?"

Training:
  1. Collect simulation data
  2. Compute Pareto front from data
  3. Train D to distinguish Pareto-optimal from dominated designs
  4. Train G to generate designs that D classifies as Pareto-optimal
  5. Condition G on a tradeoff weight to control position on the front

Usage:
  - Sweep tradeoff weight from 0 to 1
  - At each weight, G generates a design on the Pareto front
  - Result: complete Pareto curve in one forward pass
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ParetoGANConfig:
    noise_dim: int = 16
    hidden_dims: list[int] = None
    learning_rate: float = 2e-4
    epochs: int = 400
    batch_size: int = 32
    n_critic: int = 5
    gp_lambda: float = 10.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 256, 128]


def compute_pareto_front(objectives: np.ndarray, maximize: list[bool]) -> np.ndarray:
    """
    Compute Pareto-optimal indices from a set of objective values.
    
    Args:
        objectives: (N, K) array of K objective values
        maximize: list of bools, True if objective should be maximized
        
    Returns:
        pareto_mask: (N,) boolean array, True for Pareto-optimal points
    """
    n = len(objectives)
    # Convert to minimization (negate maximization objectives)
    obj_min = objectives.copy()
    for i, is_max in enumerate(maximize):
        if is_max:
            obj_min[:, i] = -obj_min[:, i]

    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not pareto_mask[i]:
            continue
        for j in range(n):
            if i == j or not pareto_mask[j]:
                continue
            # j dominates i if j is <= i in all objectives and < in at least one
            if np.all(obj_min[j] <= obj_min[i]) and np.any(obj_min[j] < obj_min[i]):
                pareto_mask[i] = False
                break
    return pareto_mask


class ParetoGenerator(nn.Module):
    """Generates Pareto-optimal designs conditioned on tradeoff weight."""

    def __init__(self, noise_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()
        # Input: noise + 1 tradeoff weight
        input_dim = noise_dim + 1
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
            prev = h
        layers.extend([nn.Linear(prev, output_dim), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, weight], dim=1)
        return self.net(x)


class ParetoDiscriminator(nn.Module):
    """Discriminates Pareto-optimal from dominated designs."""

    def __init__(self, param_dim: int, hidden_dims: list[int]):
        super().__init__()
        # Input: params + 1 tradeoff weight
        input_dim = param_dim + 1
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.LeakyReLU(0.2)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = torch.cat([params, weight], dim=1)
        return self.net(x)


class ParetoGAN:
    """
    Generates designs along the Pareto frontier between competing objectives.
    
    Objectives for OTA2: maximize GBW, minimize Power
    Tradeoff weight: 0 = pure GBW focus, 1 = pure Power focus
    """

    def __init__(self, param_dim: int, config: ParetoGANConfig | None = None):
        self.config = config or ParetoGANConfig()
        self.param_dim = param_dim
        self.G = ParetoGenerator(self.config.noise_dim, param_dim, self.config.hidden_dims)
        self.D = ParetoDiscriminator(param_dim, self.config.hidden_dims)
        self.param_bounds: np.ndarray | None = None
        self.is_trained = False

    def train(self, params_array: np.ndarray, objectives: np.ndarray,
              maximize: list[bool], param_bounds: np.ndarray,
              feasible_mask: np.ndarray | None = None) -> dict:
        """
        Train Pareto GAN.
        
        Args:
            params_array: (N, D) design parameters
            objectives: (N, 2) two competing objectives (e.g., GBW, Power)
            maximize: [True, False] for [GBW, Power]
            param_bounds: (D, 2) parameter bounds
            feasible_mask: (N,) boolean, only use feasible designs
        """
        self.param_bounds = param_bounds

        # Filter to feasible only
        if feasible_mask is not None:
            params_array = params_array[feasible_mask]
            objectives = objectives[feasible_mask]

        if len(params_array) < 10:
            return {"error": "Not enough feasible samples for Pareto GAN training"}

        # Compute Pareto front
        pareto_mask = compute_pareto_front(objectives, maximize)
        n_pareto = pareto_mask.sum()

        # Assign tradeoff weights to Pareto points (based on position along front)
        pareto_obj = objectives[pareto_mask]
        # Sort by first objective (GBW) to assign weights
        sort_idx = np.argsort(pareto_obj[:, 0])
        weights_pareto = np.linspace(0, 1, n_pareto)[sort_idx.argsort()]

        # For dominated points, assign weight based on nearest Pareto point
        all_weights = np.zeros(len(params_array))
        all_weights[pareto_mask] = weights_pareto
        dominated_idx = np.where(~pareto_mask)[0]
        pareto_idx = np.where(pareto_mask)[0]
        for di in dominated_idx:
            dists = np.linalg.norm(objectives[pareto_idx] - objectives[di], axis=1)
            nearest = np.argmin(dists)
            all_weights[di] = weights_pareto[nearest]

        # Normalize params
        params_norm = (params_array - param_bounds[:, 0]) / (param_bounds[:, 1] - param_bounds[:, 0])

        # Training tensors
        real_params = torch.FloatTensor(params_norm[pareto_mask])
        real_weights = torch.FloatTensor(weights_pareto).unsqueeze(1)
        n_real = len(real_params)

        if n_real < 3:
            return {"error": f"Only {n_real} Pareto points, need at least 3"}

        opt_G = optim.Adam(self.G.parameters(), lr=self.config.learning_rate, betas=(0.0, 0.9))
        opt_D = optim.Adam(self.D.parameters(), lr=self.config.learning_rate, betas=(0.0, 0.9))

        g_losses, d_losses = [], []

        for epoch in range(self.config.epochs):
            # Sample batch from Pareto front (with replacement if small)
            idx = torch.randint(0, n_real, (min(self.config.batch_size, n_real),))
            real_batch = real_params[idx]
            weight_batch = real_weights[idx]
            bs = len(real_batch)

            # Train D
            for _ in range(self.config.n_critic):
                noise = torch.randn(bs, self.config.noise_dim)
                fake = self.G(noise, weight_batch).detach()

                d_real = self.D(real_batch, weight_batch)
                d_fake = self.D(fake, weight_batch)

                # Gradient penalty
                alpha = torch.rand(bs, 1)
                interp = (alpha * real_batch + (1 - alpha) * fake).requires_grad_(True)
                d_interp = self.D(interp, weight_batch)
                grads = torch.autograd.grad(d_interp, interp,
                    grad_outputs=torch.ones_like(d_interp),
                    create_graph=True, retain_graph=True)[0]
                gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()

                d_loss = d_fake.mean() - d_real.mean() + self.config.gp_lambda * gp
                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            # Train G
            noise = torch.randn(bs, self.config.noise_dim)
            fake = self.G(noise, weight_batch)
            g_loss = -self.D(fake, weight_batch).mean()
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        self.G.eval()
        self.D.eval()
        self.is_trained = True

        return {
            "epochs": self.config.epochs,
            "n_pareto_points": int(n_pareto),
            "n_total_feasible": len(params_array),
            "final_d_loss": d_losses[-1] if d_losses else 0,
            "final_g_loss": g_losses[-1] if g_losses else 0,
        }

    def generate_pareto_front(self, n_points: int = 20, n_samples_per_point: int = 5) -> np.ndarray:
        """
        Generate designs spanning the Pareto front.
        
        Args:
            n_points: number of points along the front
            n_samples_per_point: candidates per tradeoff weight (best is selected)
            
        Returns:
            designs: (n_points, D) design parameters in original scale
        """
        if not self.is_trained:
            raise RuntimeError("Pareto GAN not trained")

        weights = np.linspace(0, 1, n_points)
        designs = []

        for w in weights:
            weight_tensor = torch.FloatTensor([[w]]).repeat(n_samples_per_point, 1)
            noise = torch.randn(n_samples_per_point, self.config.noise_dim)

            with torch.no_grad():
                candidates_norm = self.G(noise, weight_tensor).numpy()
                # Pick the one D rates highest
                scores = self.D(torch.FloatTensor(candidates_norm), weight_tensor).numpy().flatten()
                best_idx = np.argmax(scores)

            designs.append(candidates_norm[best_idx])

        designs = np.array(designs)
        # Denormalize
        return designs * (self.param_bounds[:, 1] - self.param_bounds[:, 0]) + self.param_bounds[:, 0]

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.G.state_dict(), path / "pareto_generator.pt")
        torch.save(self.D.state_dict(), path / "pareto_discriminator.pt")
        np.save(path / "pareto_bounds.npy", self.param_bounds)

    def load(self, path: Path):
        self.G.load_state_dict(torch.load(path / "pareto_generator.pt", weights_only=True))
        self.D.load_state_dict(torch.load(path / "pareto_discriminator.pt", weights_only=True))
        self.param_bounds = np.load(path / "pareto_bounds.npy")
        self.G.eval()
        self.D.eval()
        self.is_trained = True
