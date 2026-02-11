from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as transforms


def get_data_loaders(
    batch_size: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ]
    )
    transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root="./data", train=False, download=False, transform=transform_val
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return train_loader, val_loader
