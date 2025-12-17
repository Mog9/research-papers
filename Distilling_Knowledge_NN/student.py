import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.49, 0.48, 0.44),
                         (0.24, 0.24, 0.26))
])

test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49, 0.48, 0.44),
                         (0.24, 0.24, 0.26))
])


train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tfms)
test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=test_tfms)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)


#load teacher



