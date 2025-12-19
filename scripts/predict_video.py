import sys
import os

# Fix import path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import torch
import cv2
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image

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
# Transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------
# Argument parser
# -------------------------
parser = argparse.ArgumentParser(description="Deepfake Video Detection")
parser.add_argument("--video", type=str, required=True, help="Path to video file")
parser.add_argument("--frame_skip", type=int, default=10, help="Process every Nth frame")
args = parser.parse_args()

if not os.path.exists(args.video):
    raise FileNotFoundError(f"Video not found at {args.video}")

# -------------------------
# Load model
# -------------------------
model = TinyCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------------------------
# Video processing
# -------------------------
cap = cv2.VideoCapture(args.video)

frame_count = 0
fake_probs = []

print("\n📽️ Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames
    if frame_count % args.frame_skip != 0:
        continue

    # Convert frame to PIL image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(image)
        prob_fake = torch.softmax(output, dim=1)[0, 1].item()
        fake_probs.append(prob_fake)

cap.release()

# -------------------------
# Aggregate results
# -------------------------
if len(fake_probs) == 0:
    raise RuntimeError("No frames were processed. Try reducing frame_skip.")

avg_fake_prob = float(np.mean(fake_probs))
label = "FAKE" if avg_fake_prob >= 0.5 else "REAL"
confidence = avg_fake_prob if label == "FAKE" else (1 - avg_fake_prob)

# -------------------------
# Output
# -------------------------
print("\n🎬 Deepfake Video Detection Result")
print("----------------------------------")
print(f"Frames analyzed : {len(fake_probs)}")
print(f"Prediction      : {label}")
print(f"Confidence      : {confidence:.2f}")
