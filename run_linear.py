import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from models.linear_model import LinearClassifier

def load_pretrained_encoder(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
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

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

filepath = 'pretrain/checkpoint_0200.pth.tar'
backbone, feat_dim = load_pretrained_encoder(filepath)
model = LinearClassifier(backbone, feature_dim=feat_dim)
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
