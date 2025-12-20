# scripts/predict_image_v2.py
import torch
from torchvision import transforms, models
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
from src.face_crop_v2 import crop_face

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
parser = argparse.ArgumentParser(description="Deepfake Image Detection")
parser.add_argument("--image", type=str, required=True, help="Path to image file")
args = parser.parse_args()

if not os.path.exists(args.image):
    raise FileNotFoundError(f"Image not found at {args.image}")

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
# Load & crop face
# -------------------------
face = crop_face(args.image)
if face is None:
    raise ValueError("No face detected in the image.")
image = transform(face).unsqueeze(0).to(device)

# -------------------------
# Prediction
# -------------------------
with torch.no_grad():
    output = model(image)
    prob_fake = torch.softmax(output, dim=1)[0, 1].item()

label = "FAKE" if prob_fake >= 0.5 else "REAL"
confidence = prob_fake if label=="FAKE" else 1-prob_fake

print(f"\nPrediction : {label}")
print(f"Confidence : {confidence:.2f}")
