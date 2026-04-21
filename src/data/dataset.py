import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from .transforms import get_transforms


def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    
    transform = get_transforms(is_train=True)

    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/Train",
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/Test",
        transform=get_transforms(is_train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader