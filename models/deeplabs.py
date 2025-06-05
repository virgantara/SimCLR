from torchvision.models.segmentation import DeepLabV3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models import resnet18
import torch.nn as nn

def deeplabv3_resnet18(num_classes):
    backbone = resnet18(pretrained=False)
    backbone.fc = nn.Identity()

    model = DeepLabV3(backbone=backbone, classifier=DeepLabHead(512, num_classes))
    return model