# src/image_dataset_v2.py
import sys
import os

# -------------------------
# Fix import path
# -------------------------
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from PIL import Image
from torch.utils.data import Dataset
from src.face_crop_v2 import crop_face

class DeepfakeImageDataset(Dataset):
    """
    Dataset for Deepfake Image Detection (ResNet18 pipeline)
    Expects folder structure:
        root_dir/
            real/
            fake/
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Collect image paths and labels
        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        for img_name in os.listdir(real_dir):
            img_path = os.path.join(real_dir, img_name)
            if os.path.isfile(img_path):
                self.samples.append((img_path, 1))  # 1 = real

        for img_name in os.listdir(fake_dir):
            img_path = os.path.join(fake_dir, img_name)
            if os.path.isfile(img_path):
                self.samples.append((img_path, 0))  # 0 = fake

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Open image
        image = Image.open(img_path).convert("RGB")

        # Crop face
        cropped = crop_face(image)

        if cropped is None:
            # Fallback: use original image resized to 224x224
            cropped = image.resize((224, 224))

        # Apply transforms (to Tensor, Normalize, etc.)
        if self.transform:
            cropped = self.transform(cropped)

        return cropped, label
