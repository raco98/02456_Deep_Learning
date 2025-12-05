"""
Hyperparameter sweep for CVAE training.
Tests different latent dimensions and dropout rates.
Reports best validation ELBO for each configuration.

Usage:
    python hyperparameter_sweep.py
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
from pathlib import Path
import math
import pandas as pd
import sys

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

# Assume imports from notebook are available
DATA_LOCATION = Path("data_numpy/density_large128.npy")

# Load and prepare data (same as notebook)
data = np.load(DATA_LOCATION)
max_val = np.max(data)
density_data = data / max_val

data_tensor = torch.tensor(density_data, dtype=torch.float32)
data_tensor = data_tensor.unsqueeze(1)  # [N, 1, H, W]

total_count = len(data_tensor)
train_size = int(0.8 * total_count)
test_size = total_count - train_size

train_dataset, test_dataset = random_split(data_tensor, [train_size, test_size])

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


class ReparameterizedDiagonalGaussian(torch.distributions.Distribution):
    def __init__(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        assert mu.shape == log_sigma.shape
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> torch.Tensor:
        return torch.empty_like(self.mu).normal_()

    def rsample(self) -> torch.Tensor:
        epsilon = self.sample_epsilon()
        return self.mu + self.sigma * epsilon

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
            math.log(2 * math.pi)
            + 2 * torch.log(self.sigma)
            + ((z - self.mu) ** 2) / (self.sigma ** 2)
        )


class ConvolutionalVariationalAutoencoder(nn.Module):
    def __init__(
        self, input_shape: torch.Size, latent_features: int, dropout_rate: float = 0.0
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.dropout_rate = dropout_rate

        # Encoder with optional dropout
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_features * 2),
        )

        # Decoder
        self.conv_decoder = nn.Sequential(
            nn.Linear(latent_features, 256 * 8 * 8),
            nn.Unflatten(dim=1, unflattened_size=(256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.register_buffer("prior_params", torch.zeros(1, 2 * latent_features))
        self.log_scale = nn.Parameter(torch.tensor([0.0]))

    def posterior(self, x: torch.Tensor):
        h_x = self.conv_encoder(x)
        mu, log_sigma = h_x.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, min=-10, max=10)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior(self, batch_size: int = 1):
        prior_params = self.prior_params.expand(
            batch_size, *self.prior_params.shape[-1:]
        )
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def observation_model(self, z: torch.Tensor):
        mu = self.conv_decoder(z)
        scale = torch.clamp(torch.exp(self.log_scale), min=1e-4, max=1.0)
        return torch.distributions.Normal(mu, scale)

    def forward(self, x):
        qz = self.posterior(x)
        pz = self.prior(batch_size=x.size(0))
        z = qz.rsample()
        px = self.observation_model(z)
        return {"px": px, "pz": pz, "qz": qz, "z": z}


def reduce(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1).sum(dim=1)


class VariationalInference(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, model, x):
        outputs = model(x)
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        log_px = reduce(px.log_prob(x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))
        kl = log_qz - log_pz
        elbo = log_px - kl
        beta_elbo = log_px - self.beta * kl
        loss = -beta_elbo.mean()
        with torch.no_grad():
            diagnostics = {"elbo": elbo, "log_px": log_px, "kl": kl}
        return loss, diagnostics


def train_config(latent_dim: int, dropout_rate: float, num_epochs: int = 50):
    """Train a model with given hyperparameters and return best val ELBO."""
    print(f"\n{'='*60}")
    print(f"Training: latent_dim={latent_dim}, dropout={dropout_rate:.2f}")
    print(f"{'='*60}")

    # Get dummy shape
    dummy = torch.tensor(data[22, :, :], dtype=torch.float32).clone()
    dummy = dummy.unsqueeze(0).unsqueeze(0)

    model = ConvolutionalVariationalAutoencoder(
        dummy.shape, latent_dim, dropout_rate
    )
    vi = VariationalInference(beta=0.75)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    model = model.to(device)

    best_val_elbo = -np.inf
    patience = 8
    epochs_without_improve = 0

    val_elbos = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        for x in train_loader:
            x = x.to(device)
            loss, _ = vi(model, x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_elbo_list = []
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device)
                _, diagnostics = vi(model, x)
                val_elbo_list.append(diagnostics["elbo"].mean().item())

        val_elbo_mean = np.mean(val_elbo_list)
        val_elbos.append(val_elbo_mean)

        if val_elbo_mean > best_val_elbo:
            best_val_elbo = val_elbo_mean
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= patience:
            print(f"  Early stop at epoch {epoch}, best val ELBO: {best_val_elbo:.4f}")
            break

    return best_val_elbo, epoch


# Hyperparameter grid
latent_dims = [8, 12, 16, 26, 32, 46, 52, 64, 72]
dropout_rates = [0.0, 0.1, 0.2, 0.3]

results = []

print("Starting hyperparameter sweep...\n")

for latent_dim in latent_dims:
    for dropout_rate in dropout_rates:
        best_val_elbo, final_epoch = train_config(latent_dim, dropout_rate)
        results.append(
            {
                "latent_dim": latent_dim,
                "dropout": dropout_rate,
                "best_val_elbo": best_val_elbo,
                "final_epoch": final_epoch,
            }
        )


print("\n\n" + "=" * 70)
print("HYPERPARAMETER SWEEP RESULTS")
print("=" * 70)

df = pd.DataFrame(results)
df = df.sort_values("best_val_elbo", ascending=False)

print(df.to_string(index=False))

print("\n" + "=" * 70)
best_row = df.iloc[0]
print(f"BEST CONFIG:")
print(f"  Latent Dim: {int(best_row['latent_dim'])}")
print(f"  Dropout: {best_row['dropout']:.2f}")
print(f"  Best Val ELBO: {best_row['best_val_elbo']:.4f}")
print(f"  Final Epoch: {int(best_row['final_epoch'])}")
print("=" * 70 + "\n")
