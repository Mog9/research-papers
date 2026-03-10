import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
from main import model

dataset = load_dataset("glue", "sst2")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
val_loader = DataLoader(dataset["validation"], batch_size=32)

class BertWithLoRAClassifier(nn.Module):
    def __init__(self, bert, num_labels=2):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :] #cls
        return self.classifier(cls_token)

classifier = BertWithLoRAClassifier(model).cuda()

optimizer = AdamW(filter(lambda p: p.requires_grad, classifier.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 3
for epoch in range(epochs):
    classifier.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["label"].cuda()

        optimizer.zero_grad()
        logits = classifier(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["label"].cuda()
            logits = classifier(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"epoch {epoch+1} - loss: {total_loss/len(train_loader):.4f} - val acc: {100*correct/total:.2f}%")







