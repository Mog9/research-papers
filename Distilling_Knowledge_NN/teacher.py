import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.446],
                         std=[0.247, 0.243, 0.261])
])

test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.446],
                         std=[0.247, 0.243, 0.261])
])

train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tfms)
test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=test_tfms)

train_loader =  DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

#techer model
teacher = models.resnet18(weights=None)
teacher.conv1 = nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)
teacher.maxpool = nn.Identity()
teacher.fc = nn.Linear(teacher.fc.in_features, 10)
teacher = teacher.to(device)


#loss/optim
crit = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    teacher.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001
)


def train_teacher(model, loader):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = crit(logits, labels) #CLE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
    return correct/total


epochs = 10
for epoch in range(epochs):
    train_loss = train_teacher(teacher, train_loader)
    val_acc = evaluate(teacher, val_loader)
    print(f"epoch {epoch}: loss: {train_loss:.4f}, val_acc: {val_acc:.4f}")

for p in teacher.parameters():
    p.requires_grad = False
teacher.eval()
