import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear, r, alpha):
        super().__init__()
        d_in, d_out = linear.weight.shape[1], linear.weight.shape[0]
        self.W = linear
        self.W.weight.requires_grad = False #freeze
        if self.W.bias is not None:
            self.W.bias.requires_grad = False
        self.A = nn.Parameter(torch.randn(d_in, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, d_out)) #both a and b are trainable, Wo frozen
        self.scaling = alpha / r

    def  forward(self, x):
        return self.W(x) + self.scaling * (x @ self.A @ self.B)
    
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-uncased")
r = 4
alpha = 8
for param in model.parameters():
    param.requires_grad = False

for i in range(model.config.num_hidden_layers):
    model.encoder.layer[i].attention.self.query = LoRALinear(model.encoder.layer[i].attention.self.query, r, alpha)
    model.encoder.layer[i].attention.self.key = LoRALinear(model.encoder.layer[i].attention.self.key, r, alpha)
print(model.encoder.layer[0].attention.self.query)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape) #only A and B

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total: {total_params:,}")
print(f"trainable: {trainable_params:,}")
print(f"trainable %: {100 * trainable_params / total_params:.2f}%")
