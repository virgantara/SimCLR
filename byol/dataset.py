from torch.utils.data import Dataset
import os
from PIL import Image
import torch

class UnlabeledImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = [
            f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image