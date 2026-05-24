
import os
from typing import List
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from .transforms import get_transforms


def get_dataloaders(data_dir, batch_size=16):
    """Load train/val datasets."""
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "Train"),
        transform=get_transforms(is_train=True)
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "Test"),
        transform=get_transforms(is_train=False)
    )
    
    print(f"[DATA] Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"[DATA] Classes: {train_dataset.classes}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes
