from __future__ import annotations

from cifar10.dataset import get_data_loaders
from transformer.optim import evaluate
from transformer.optim import train
from transformer.model import SimpleViT

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


train_loader, val_loader = get_data_loaders(batch_size=128)
device = torch.device("cpu")
model = SimpleViT().to(device)
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
epochs = 10
for epoch in range(epochs):
    print(f"### Epoch {epoch + 1} ###")
    train(model=model, optimizer=optimizer, scheduler=scheduler, train_loader=train_loader)
    evaluate(model=model, val_loader=val_loader)
    scheduler.step()
