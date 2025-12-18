import sys
import os

# -------------------------
# Fix import path
# -------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import f1_score, roc_auc_score

from src.image_dataset import DeepfakeImageDataset
from src.model import TinyCNN

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Dataset paths (Google Drive)
# -------------------------
SAVE_DIR = "/content/drive/MyDrive/deepfake_models"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, "best_model.pth")

TRAIN_DIR = "/content/drive/MyDrive/deepfake_dataset/train"
VAL_DIR = "/content/drive/MyDrive/deepfake_dataset/valid"

# -------------------------
# Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# -------------------------
# Datasets
# -------------------------
train_dataset = DeepfakeImageDataset(
    root_dir=TRAIN_DIR,
    transform=train_transform
)

val_dataset = DeepfakeImageDataset(
    root_dir=VAL_DIR,
    transform=val_transform
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# -------------------------
# DataLoaders
# -------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=32,          # reduce to 16 if OOM
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# -------------------------
# Model, Loss, Optimizer
# -------------------------
model = TinyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# Training settings
# -------------------------
epochs = 10
best_auc = 0.0

# -------------------------
# Training loop
# -------------------------
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 35)

    # -------- TRAIN --------
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    print(f"Train Loss: {train_loss:.4f}")

    # -------- VALIDATION --------
    model.eval()
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Validation Accuracy : {acc:.2f}%")
    print(f"Validation F1 Score : {f1:.4f}")
    print(f"Validation ROC-AUC  : {auc:.4f}")

    # -------- SAVE BEST MODEL (AUC-based) --------
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Model saved to {SAVE_PATH}")


print("\nTraining complete.")
print(f"Best Validation ROC-AUC: {best_auc:.4f}")
