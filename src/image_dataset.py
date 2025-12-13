import os
from PIL import Image
from torch.utils.data import Dataset

class DeepfakeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path like data/deepfake_images/train
        Expected structure:
            root_dir/
                real/
                fake/
        """
        self.transform = transform
        self.samples = []

        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        for img_name in os.listdir(real_dir):
            self.samples.append((os.path.join(real_dir, img_name), 1))

        for img_name in os.listdir(fake_dir):
            self.samples.append((os.path.join(fake_dir, img_name), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
