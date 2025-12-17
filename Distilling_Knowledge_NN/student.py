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


train_ds = datasets.CIFAR10(root="data", train=True, download=False, transform=train_tfms)
test_ds = datasets.CIFAR10(root="data", train=False, download=False, transform=test_tfms)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)


#load teacher
teacher = models.resnet18(weights=None)
teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
teacher.maxpool = nn.Identity()
teacher.fc = nn.Linear(teacher.fc.in_features, 10)
teacher.load_state_dict(torch.load("teacher_cifar10.pth"))
teacher = teacher.to(device)
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False


#student model
student = models.resnet18(weights=None)
student.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
student.maxpool = nn.Identity()
student.fc = nn.Linear(student.fc.in_features, 10)
student = student.to(device)


#paper equation (temperature softmax)
def softmax_with_temp(logits, T):
    return F.softmax(logits / T, dim=1)


def distill_loss(student_logits, teacher_logits, T):
    student_prob = F.log_softmax(student_logits / T, dim=1)
    teacher_prob = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(student_prob, teacher_prob, reduction="batchmean") * (T * T)


optimizer = optim.SGD(
    student.parameters(),
    lr = 0.01,
    momentum=0.9,
    weight_decay=0.0001
)
T = 4.0


def train_student(student, teacher, loader):
    student.train()
    total_loss = 0

    for images, _ in loader:
        images = images.to(device)

        with torch.no_grad():
            teacher_logits = teacher(images)

        student_logits = student(images)
        loss = distill_loss(student_logits, teacher_logits, T)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
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
    return correct / total

epochs = 10

for epoch in range(epochs):
    loss = train_student(student, teacher, train_loader)
    acc = evaluate(student, val_loader)
    print(f"epoch {epoch}: distill_loss: {loss:.4f}, val_acc: {acc:.4f}")
