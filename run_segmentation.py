import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_pretrained_encoder(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    state_dict = checkpoint['state_dict']

    backbone = deeplabv3_resnet50(weights=False, num_classes=3)
    resnet_backbone = backbone.backbone

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone") and "projector" not in k:
            new_k = k.replace("backbone.", "")
            new_state_dict[new_k] = v

    resnet_backbone.load_state_dict(new_state_dict, strict=False)
    return backbone


def get_dataloaders():
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.PILToTensor()
    ])

    dataset = OxfordIIITPet(
        root="./oxford_pet",
        split="trainval",
        target_types="segmentation",
        transform=image_transform,
        target_transform=mask_transform,
        download=True
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4)

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader):
        imgs = imgs.to(device)
        masks = masks.squeeze(1).long().to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.squeeze(1).long().to(device)
            outputs = model(imgs)['out']
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == masks).sum().item()
            total += masks.numel()
    acc = correct / total * 100
    print(f"Pixel Accuracy: {acc:.2f}%")
    return acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders()
    model = load_pretrained_encoder(args.pretrain_path).to(device)

    if args.eval:
        model.load_state_dict(torch.load(args.seg_model_name, map_location=device))
        evaluate(model, test_loader, device)
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        evaluate(model, test_loader, device)

    torch.save(model.state_dict(), args.seg_model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path', type=str, default='pretrain/checkpoint_0200.pth.tar')
    parser.add_argument('--seg_model_name', type=str, default='segmentation_model.pth')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    main(args)
