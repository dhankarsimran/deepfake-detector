import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.image_dataset import DeepfakeImageDataset
from src.model import TinyCNN
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "/content/drive/MyDrive/deepfake_dataset/test"
MODEL_PATH = "/content/drive/MyDrive/deepfake_models/best_model.pth"
BATCH_SIZE = 32
OUTPUT_CSV = "/content/drive/MyDrive/deepfake_models/test_predictions.csv"

# -------------------------
# Transforms (same as validation)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------
# Load test dataset
# -------------------------
test_dataset = DeepfakeImageDataset(
    root_dir=TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Test samples: {len(test_dataset)}")

# -------------------------
# Load model
# -------------------------
model = TinyCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# Evaluation
# -------------------------
all_labels = []
all_preds = []
all_probs = []
all_paths = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of FAKE
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # Save image paths if your dataset returns them
        if hasattr(test_dataset, "image_paths"):
            all_paths.extend(test_dataset.image_paths)

# -------------------------
# Compute metrics
# -------------------------
acc = 100 * sum([p==l for p,l in zip(all_preds, all_labels)]) / len(all_labels)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print("\n🔹 Test set evaluation")
print("----------------------------")
print(f"Accuracy : {acc:.2f}%")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

# -------------------------
# Save predictions
# -------------------------
df = pd.DataFrame({
    "image_path": all_paths if all_paths else [f"img_{i}" for i in range(len(all_labels))],
    "true_label": all_labels,
    "pred_label": all_preds,
    "prob_fake": all_probs
})

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Predictions saved to {OUTPUT_CSV}")
