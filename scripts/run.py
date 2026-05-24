import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.data.dataset import get_dataloaders
from src.models.model import SkinLesionModel
from src.train import train
from src.data.transforms import get_transforms
from src.export_to_onnx import export_to_onnx, validate_onnx, compare_outputs, benchmark



# ---------------- CONFIG ----------------
# Data paths on Kaggle
KAGGLE_DATA_DIR = "/kaggle/input/datasets/nodoubttome/skin-cancer9-classesisic/Skin cancer ISIC The International Skin Imaging Collaboration"
OUTPUT_DIR = "models/"

# Training config
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 50
NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

print(f"[CONFIG] Device: {DEVICE}")
print(f"[CONFIG] Data dir: {KAGGLE_DATA_DIR}")
print(f"[CONFIG] Output dir: {OUTPUT_DIR}")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    print("\n" + "="*60)
    print("SKIN LESION CLASSIFICATION - PIPELINE")
    print("="*60)

    set_seed(SEED)

    # Data
    print("\n[STEP 1/5] Loading data...")
    train_loader, val_loader, classes = get_dataloaders(KAGGLE_DATA_DIR, BATCH_SIZE)

    # Class weights
    targets = np.array(train_loader.dataset.targets)
    class_counts = np.bincount(targets)
    weights = len(targets) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    print(f"[DATA] Class weights: {weights}")

    # Model & optim
    print("\n[STEP 2/5] Creating model...")
    model = SkinLesionModel(num_classes=NUM_CLASSES)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print(f"\n[STEP 3/5] Training ({EPOCHS} epochs)...")
    model, best_f1 = train(model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE, EPOCHS, OUTPUT_DIR)
    print(f"[TRAIN] Best Macro F1: {best_f1:.4f}")

    # Export
    print(f"\n[STEP 4/5] Exporting to ONNX...")
    model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    onnx_path = os.path.join(OUTPUT_DIR, "skin_classifier_v1.onnx")
    export_to_onnx(model, onnx_path, device=DEVICE)

    # Validate & compare
    model.to('cpu')
    model.eval()  # CRITICAL: Set to eval mode (disables dropout/batchnorm training behavior)
    if validate_onnx(onnx_path):
        compare_outputs(model, onnx_path)

    # Benchmark
    print(f"\n[STEP 5/5] Benchmarking...")
    benchmark_results = benchmark(model, onnx_path, device=DEVICE)

    # Save metadata
    metadata = {
        "best_f1": float(best_f1),
        "model_path": model_path,
        "onnx_path": onnx_path,
        "benchmark": benchmark_results,
        "classes": classes,
    }
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE")
    print(f"  - Best Model: {model_path}")
    print(f"  - ONNX Model: {onnx_path}")
    print(f"  - Best F1: {best_f1:.4f}")
    print(f"  - Speedup (PyTorch→ONNX): {benchmark_results['speedup']:.2f}x")
    print("="*60)

if __name__ == "__main__":
    main()