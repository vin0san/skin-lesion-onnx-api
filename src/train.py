import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
import os
import json


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return avg_loss, macro_f1, balanced_acc

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, output_dir):
    """Full training loop."""
    os.makedirs(output_dir, exist_ok=True)
    
    best_f1 = 0.0
    patience_counter = 0
    history = []
    
    model_path = os.path.join(output_dir, "best_model.pth")
    metrics_path = os.path.join(output_dir, "metrics.json")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_balanced_acc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
            status = f"✓ New Best F1: {val_f1:.4f}"
        else:
            patience_counter += 1
            status = f"✗ No improvement ({patience_counter}/7)"
        
        print(
            f"[{epoch+1:2d}/{num_epochs}] "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"Acc: {val_balanced_acc:.4f} | "
            f"LR: {current_lr:.1e} | "
            f"{status}"
        )
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_f1": float(val_f1),
            "val_balanced_acc": float(val_balanced_acc),
            "lr": float(current_lr)
        })
        
        with open(metrics_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        if patience_counter >= 7:
            print(f"\n[EARLY STOP] Best F1: {best_f1:.4f}")
            break
    
    return model, best_f1
