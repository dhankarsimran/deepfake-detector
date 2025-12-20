# scripts/train_resnet_v2.py
import sys
import os

# -------------------------
# Fix import path
# -------------------------
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from src.image_dataset_v2 import DeepfakeImageDataset

# -------------------------
# Drive path (your dataset is in Drive)
# -------------------------
ROOT_DIR = "/content/drive/MyDrive"
train_dir = os.path.join(ROOT_DIR, "deepfake_detector/train")
val_dir = os.path.join(ROOT_DIR, "deepfake_detector/valid")
model_save_path = os.path.join(ROOT_DIR, "saved_models/best_model_resnet.pth")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------
# Dataset & Loader
# -------------------------
train_dataset = DeepfakeImageDataset(train_dir, transform=train_transform)
val_dataset   = DeepfakeImageDataset(val_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# -------------------------
# Model
# -------------------------
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# Training loop
# -------------------------
best_acc = 0.0
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ Model saved to {model_save_path}")

print("Training complete.")
print(f"Best Validation Accuracy: {best_acc:.2f}%")
