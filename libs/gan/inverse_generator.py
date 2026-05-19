"""
Inverse Design Generator: Given target specs, generate design parameters.

Architecture (Conditional WGAN-GP):
  G: (target_spec, noise) -> design_parameters
  D: (design_parameters, target_spec) -> real/fake score

The generator learns the inverse mapping: from desired performance to
circuit parameters that achieve it. This provides warm-start points
for the optimizer, dramatically reducing search iterations.

Training data: (params, metrics) pairs from ngspice simulations.
We train G to generate params conditioned on metrics (inverse direction).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataclasses import dataclass


@dataclass
class InverseGenConfig:
    noise_dim: int = 32
    hidden_dims_g: list[int] = None
    hidden_dims_d: list[int] = None
    learning_rate_g: float = 1e-4
    learning_rate_d: float = 1e-4
    epochs: int = 500
    batch_size: int = 64
    n_critic: int = 5  # D updates per G update (WGAN training)
    gp_lambda: float = 10.0  # gradient penalty weight
    condition_dim: int = 4  # number of spec metrics to condition on

    def __post_init__(self):
        if self.hidden_dims_g is None:
            self.hidden_dims_g = [128, 256, 128]
        if self.hidden_dims_d is None:
            self.hidden_dims_d = [128, 256, 128]


class Generator(nn.Module):
    """Generates design parameters conditioned on target specs."""

    def __init__(self, noise_dim: int, condition_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()
        input_dim = noise_dim + condition_dim
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # output in [0, 1], rescale to design bounds later
        self.net = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, condition], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    """Judges if (params, spec) pairs are realistic."""

    def __init__(self, param_dim: int, condition_dim: int, hidden_dims: list[int]):
        super().__init__()
        input_dim = param_dim + condition_dim
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.LeakyReLU(0.2)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        # No sigmoid — WGAN uses raw scores
        self.net = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.cat([params, condition], dim=1)
        return self.net(x)


class InverseDesignGenerator:
    """
    WGAN-GP for inverse circuit design.
    Given target specs, generates design parameters likely to achieve them.
    """

    def __init__(self, param_dim: int, config: InverseGenConfig | None = None):
        self.config = config or InverseGenConfig()
        self.param_dim = param_dim
        self.G = Generator(self.config.noise_dim, self.config.condition_dim,
                           param_dim, self.config.hidden_dims_g)
        self.D = Discriminator(param_dim, self.config.condition_dim, self.config.hidden_dims_d)
        self.param_bounds: np.ndarray | None = None  # (D, 2) min/max per param
        self.spec_scaler: dict | None = None
        self.is_trained = False

    def train(self, params_array: np.ndarray, metrics_array: np.ndarray,
              param_bounds: np.ndarray) -> dict:
        """
        Train the WGAN-GP on (params, metrics) pairs.
        
        Args:
            params_array: (N, D) design parameters
            metrics_array: (N, M) corresponding performance metrics
            param_bounds: (D, 2) min/max bounds per parameter
        """
        self.param_bounds = param_bounds

        # Normalize params to [0, 1]
        params_norm = (params_array - param_bounds[:, 0]) / (param_bounds[:, 1] - param_bounds[:, 0])

        # Normalize specs (metrics used as conditions)
        self.spec_scaler = {'mean': metrics_array.mean(axis=0), 'std': metrics_array.std(axis=0) + 1e-8}
        specs_norm = (metrics_array - self.spec_scaler['mean']) / self.spec_scaler['std']

        # Tensors
        real_params = torch.FloatTensor(params_norm)
        conditions = torch.FloatTensor(specs_norm)
        n_samples = len(real_params)

        # Optimizers
        opt_G = optim.Adam(self.G.parameters(), lr=self.config.learning_rate_g, betas=(0.0, 0.9))
        opt_D = optim.Adam(self.D.parameters(), lr=self.config.learning_rate_d, betas=(0.0, 0.9))

        g_losses, d_losses = [], []

        for epoch in range(self.config.epochs):
            perm = torch.randperm(n_samples)
            epoch_d_loss, epoch_g_loss = 0.0, 0.0
            n_batches = 0

            for i in range(0, n_samples - self.config.batch_size, self.config.batch_size):
                batch_idx = perm[i:i + self.config.batch_size]
                real_batch = real_params[batch_idx]
                cond_batch = conditions[batch_idx]
                bs = len(real_batch)

                # Train Discriminator
                for _ in range(self.config.n_critic):
                    noise = torch.randn(bs, self.config.noise_dim)
                    fake_params = self.G(noise, cond_batch).detach()

                    d_real = self.D(real_batch, cond_batch)
                    d_fake = self.D(fake_params, cond_batch)

                    # Gradient penalty
                    alpha = torch.rand(bs, 1)
                    interpolated = (alpha * real_batch + (1 - alpha) * fake_params).requires_grad_(True)
                    d_interp = self.D(interpolated, cond_batch)
                    gradients = torch.autograd.grad(
                        outputs=d_interp, inputs=interpolated,
                        grad_outputs=torch.ones_like(d_interp),
                        create_graph=True, retain_graph=True
                    )[0]
                    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                    d_loss = d_fake.mean() - d_real.mean() + self.config.gp_lambda * gp
                    opt_D.zero_grad()
                    d_loss.backward()
                    opt_D.step()

                # Train Generator
                noise = torch.randn(bs, self.config.noise_dim)
                fake_params = self.G(noise, cond_batch)
                g_loss = -self.D(fake_params, cond_batch).mean()
                opt_G.zero_grad()
                g_loss.backward()
                opt_G.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                n_batches += 1

            if n_batches > 0:
                d_losses.append(epoch_d_loss / n_batches)
                g_losses.append(epoch_g_loss / n_batches)

        self.G.eval()
        self.D.eval()
        self.is_trained = True

        return {
            "epochs": self.config.epochs,
            "final_d_loss": d_losses[-1] if d_losses else 0,
            "final_g_loss": g_losses[-1] if g_losses else 0,
            "samples_used": n_samples,
        }

    def generate(self, target_specs: np.ndarray, n_candidates: int = 10) -> np.ndarray:
        """
        Generate design parameters for given target specs.
        
        Args:
            target_specs: (M,) target metric values
            n_candidates: number of candidates to generate
            
        Returns:
            candidates: (n_candidates, D) design parameters in original scale
        """
        if not self.is_trained:
            raise RuntimeError("Generator not trained yet")

        # Normalize specs
        specs_norm = (target_specs - self.spec_scaler['mean']) / self.spec_scaler['std']
        cond = torch.FloatTensor(specs_norm).unsqueeze(0).repeat(n_candidates, 1)
        noise = torch.randn(n_candidates, self.config.noise_dim)

        with torch.no_grad():
            params_norm = self.G(noise, cond).numpy()

        # Denormalize to original scale
        params = params_norm * (self.param_bounds[:, 1] - self.param_bounds[:, 0]) + self.param_bounds[:, 0]
        return params

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.G.state_dict(), path / "generator.pt")
        torch.save(self.D.state_dict(), path / "discriminator.pt")
        np.savez(path / "inverse_gen_meta.npz",
                 param_bounds=self.param_bounds,
                 spec_mean=self.spec_scaler['mean'],
                 spec_std=self.spec_scaler['std'])

    def load(self, path: Path):
        self.G.load_state_dict(torch.load(path / "generator.pt", weights_only=True))
        self.D.load_state_dict(torch.load(path / "discriminator.pt", weights_only=True))
        meta = np.load(path / "inverse_gen_meta.npz")
        self.param_bounds = meta['param_bounds']
        self.spec_scaler = {'mean': meta['spec_mean'], 'std': meta['spec_std']}
        self.G.eval()
        self.D.eval()
        self.is_trained = True
