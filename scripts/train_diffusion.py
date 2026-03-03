from __future__ import annotations

import torch
from torch.optim import AdamW

from cifar10.dataset import get_data_loaders
from diffusion.model import GaussianDiffusion
from diffusion.model import SimpleUNet
from diffusion.optim import train


device = torch.device("cpu")
model = SimpleUNet().to(device)
diffusion = GaussianDiffusion(n_timesteps=1000, device=device)
optimizer = AdamW(model.parameters(), lr=2e-4)
train_loader, _ = get_data_loaders(batch_size=64)
epochs = 10
for epoch in range(epochs):
    print(f"### Epoch {epoch + 1} ###")
    train(
        model=model,
        optimizer=optimizer,
        diffusion=diffusion,
        train_loader=train_loader,
        device=device,
    )
