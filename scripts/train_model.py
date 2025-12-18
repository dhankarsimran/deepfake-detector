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

from src.image_dataset import DeepfakeImageDataset
from src.model import TinyCNN

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Paths (Google Drive)
# -------------------------
TRAIN_DIR = "/content/drive/MyDrive/deepfake_dataset/train"
VAL_DIR = "/content/drive/MyDrive/deepfake_dataset/valid"

# -------------------------
# Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
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

# -------------------------
# DataLoaders (Drive optimized)
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

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# -------------------------
# Model
# -------------------------
model = TinyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# Training loop
# -------------------------
epochs = 5
best_acc = 0.0

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 30)

    # -------- Training --------
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

    avg_loss = running_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")

    # -------- Validation --------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    # -------- Save best model --------
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_tinycnn_deepfake.pth")
        print("✅ Best model saved")

print("\nTraining complete.")
print(f"Best Validation Accuracy: {best_acc:.2f}%")
