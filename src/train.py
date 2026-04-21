import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
import os
import json


def train_engine(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler, 
    device, 
    num_epochs, 
    save_dir
):
    model.to(device)
    best_f1 = 0.0
    history = []
    
    # Early Stopping & Checkpointing configuration
    patience_counter = 0
    early_stop_patience = 7 

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "best_model.pt")
    metrics_path = os.path.join(save_dir, "metrics.json")

    for epoch in range(num_epochs):
        # ------------------- TRAIN -------------------
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Calculate average train loss for the epoch
        avg_train_loss = train_loss / len(train_loader)

        # ------------------- VALIDATE -------------------
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        
        # ------------------- SCHEDULER STEP -------------------
        # Update LR based on validation loss plateau
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # ------------------- METRICS -------------------
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_acc = balanced_accuracy_score(all_labels, all_preds)

        # ------------------- SAVE BEST & EARLY STOPPING -------------------
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            patience_counter = 0 # Reset counter on improvement
            status = f"--> New Best F1: {val_f1:.4f}. Model Saved."
        else:
            patience_counter += 1
            status = f"--> No improvement ({patience_counter}/{early_stop_patience})"

        # ------------------- LOGGING -------------------
        print(
            f"Epoch [{epoch+1:02d}/{num_epochs}] | "
            f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"LR: {current_lr:.1e} | "
            f"{status}"
        )

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_f1": val_f1,
            "val_balanced_acc": val_acc,
            "lr": current_lr
        }
        history.append(epoch_data)

        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=4)

        if patience_counter >= early_stop_patience:
            print(f"\n[!] Early Stopping Triggered at Epoch {epoch+1}. Model reached peak generalization.")
            break

    return model