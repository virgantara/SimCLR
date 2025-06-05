import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from models.linear_model import LinearClassifier
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def load_pretrained_encoder(filepath):
    checkpoint = torch.load(filepath, map_location='cpu',weights_only=False)
    state_dict = checkpoint['state_dict']

    backbone = models.resnet18()
    num_ftrs = backbone.fc.in_features
    
    backbone.fc = nn.Identity()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone") and "projector" not in k:
            new_k = k.replace("backbone.", "")
            new_state_dict[new_k] = v
    backbone.load_state_dict(new_state_dict, strict=False)
    return backbone, num_ftrs

def main(args):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filepath = args.pretrain_path
    backbone, feat_dim = load_pretrained_encoder(filepath)
    model = LinearClassifier(backbone, feature_dim=feat_dim).to(device)

    if args.eval:
        model.load_state_dict(torch.load(args.linear_model_name, map_location=device, weights_only=True))
        evaluate(model, test_loader, device)
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.2f}%")


    torch.save(model.state_dict(), args.linear_model_name)

    evaluate(model, test_loader, device)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc='Evaluating'):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    labels = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - CIFAR-10")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path', type=str, default='pretrain/checkpoint_0200.pth.tar', help='Model Name')
    parser.add_argument('--linear_model_name', type=str, default='linear_classifier.pth', help='Model Name')
    parser.add_argument('--epochs', type=int, default=10, help='Num of epoch')
    parser.add_argument('--eval', action='store_true', help='Only evaluate without training')
    
    args = parser.parse_args()
    main(args)
    
