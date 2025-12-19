import sys
import os

# 🔧 Fix import path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import torch
from torchvision import transforms
from PIL import Image
import argparse

from src.model import TinyCNN

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Model path
# -------------------------
MODEL_PATH = "/content/drive/MyDrive/deepfake_models/best_model.pth"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# -------------------------
# Image transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# -------------------------
# Argument parser
# -------------------------
parser = argparse.ArgumentParser(description="Deepfake Image Detection")
parser.add_argument("--image", type=str, required=True, help="Path to image file")
args = parser.parse_args()

if not os.path.exists(args.image):
    raise FileNotFoundError(f"Image not found at {args.image}")

# -------------------------
# Load model
# -------------------------
model = TinyCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------------------------
# Load and preprocess image
# -------------------------
image = Image.open(args.image).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# -------------------------
# Prediction
# -------------------------
with torch.no_grad():
    output = model(image)
    prob_fake = torch.softmax(output, dim=1)[0, 1].item()

label = "FAKE" if prob_fake >= 0.5 else "REAL"
confidence = prob_fake if label == "FAKE" else (1 - prob_fake)

# -------------------------
# Output
# -------------------------
print("\n🔍 Deepfake Detection Result")
print("----------------------------")
print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2f}")
