import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import get_dataloaders
from src.models.model import SkinLesionModel
from src.train import train_engine
from src.data.transforms import get_transforms



# ---------------- CONFIG ----------------
DATA_DIR = "/kaggle/input/datasets/nodoubttome/skin-cancer9-classesisic/Skin cancer ISIC The International Skin Imaging Collaboration"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 50


def main():
    train_ds = datasets.ImageFolder(root=f"{DATA_DIR}/Train", transform=get_transforms(is_train=True))
    val_ds = datasets.ImageFolder(root=f"{DATA_DIR}/Test", transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Weights for Imbalance
    targets = np.array(train_ds.targets)
    class_counts = np.bincount(targets)
    weights = len(targets) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(weights).to(DEVICE)

    model = SkinLesionModel(num_classes=9)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    train_engine(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, "/kaggle/working/results/06_B3_final")

if __name__ == "__main__":
    main()