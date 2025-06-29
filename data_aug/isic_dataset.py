from torch.utils.data import Dataset
from PIL import Image
import os

class ISICCustomDataset(Dataset):
    def __init__(self, img_dir, label_dict, transform=None):
        self.img_dir = img_dir
        self.img_names = list(label_dict.keys())
        self.labels = [label_dict[name] for name in self.img_names]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        label = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
