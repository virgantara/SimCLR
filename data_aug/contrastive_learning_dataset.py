from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import os

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

        v1, v2 = self.transform(image)
        return [v1, v2], 0

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {
                            'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),
                            'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                            'tinyimagenet' : lambda: ImageFolder(
                                root=os.path.join(self.root_folder, 'tiny-imagenet-200', 'train'),
                                transform=ContrastiveLearningViewGenerator(
                                    self.get_simclr_pipeline_transform(64),  
                                    n_views)),
                            'isic': lambda: UnlabeledImageDataset(
                                    img_dir=os.path.join(self.root_folder, 'ISIC_2019_Training_Input'),
                                    transform=ContrastiveLearningViewGenerator(
                                        self.get_simclr_pipeline_transform(224), n_views
                                    )
                                ),
                            'isic2024': lambda: UnlabeledImageDataset(
                                    img_dir=os.path.join(self.root_folder, 'ISIC-images'),
                                    transform=ContrastiveLearningViewGenerator(
                                        self.get_simclr_pipeline_transform(224), n_views
                                    )
                                )
                        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
