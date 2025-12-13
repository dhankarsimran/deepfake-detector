import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import FaceDataset
from torch.utils.data import DataLoader

dataset = FaceDataset(root_dir="data/frames/faces", label=1)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for images, labels in loader:
    print(images.shape, labels)
    break
