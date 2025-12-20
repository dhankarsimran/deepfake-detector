# scripts/predict_video_v2.py
import cv2
import torch
from torchvision import transforms, models
from src.face_crop_v2 import crop_face
from PIL import Image
import argparse
import sys
import os

# -------------------------
# Fix import path
# -------------------------
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# -------------------------
# Paths
# -------------------------
ROOT_DIR = "/content/drive/MyDrive"
MODEL_PATH = os.path.join(ROOT_DIR, "saved_models/best_model_resnet.pth")

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Argument parser
# -------------------------
parser = argparse.ArgumentParser(description="Deepfake Video Detection")
parser.add_argument("--video", type=str, required=True, help="Path to video file")
args = parser.parse_args()

if not os.path.exists(args.video):
    raise FileNotFoundError(f"Video not found at {args.video}")

# -------------------------
# Load model
# -------------------------
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------
# Video processing
# -------------------------
cap = cv2.VideoCapture(args.video)
frame_probs = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = crop_face(frame_pil)
    if face is None:
        continue
    img_tensor = transform(face).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prob_fake = torch.softmax(output, dim=1)[0, 1].item()
        frame_probs.append(prob_fake)

cap.release()

if len(frame_probs) == 0:
    raise ValueError("No faces detected in video.")

avg_prob = sum(frame_probs)/len(frame_probs)
label = "FAKE" if avg_prob >= 0.5 else "REAL"
confidence = avg_prob if label=="FAKE" else 1-avg_prob

print(f"\nVideo Prediction : {label}")
print(f"Confidence : {confidence:.2f}")
