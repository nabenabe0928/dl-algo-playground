from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


if TYPE_CHECKING:
    from diffusion.model import GaussianDiffusion


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    diffusion: GaussianDiffusion,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    criterion = nn.MSELoss()
    model.train()
    loss_sum = 0.0

    for imgs, _ in tqdm(train_loader):
        imgs = imgs.to(device)
        batch_size = imgs.shape[0]

        # Sample random timesteps for each image in the batch
        t = torch.randint(0, diffusion.n_timesteps, (batch_size,), device=device)

        # Forward process: add noise to get x_t
        x_t, noise = diffusion.q_sample(imgs, t)

        # Predict the noise that was added
        predicted_noise = model(x_t, t)

        # Simple MSE loss between actual and predicted noise
        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    avg_loss = loss_sum / len(train_loader)
    print(f"Train Loss: {avg_loss:.6f}")
