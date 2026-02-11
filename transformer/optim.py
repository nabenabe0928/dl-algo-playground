from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

from tqdm import tqdm


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        preds = outs.argmax(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    avg_loss = loss_sum / len(train_loader)
    accuracy = 100.0 * correct / total
    print(f"Train Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")


def evaluate(
    model: nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device
) -> None:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    acc = 100.0 * correct / total
    print(f"--> Val. Accuracy: {acc:.2f}%")
