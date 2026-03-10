# LoRA Fine-tuning BERT on SST-2

A from-scratch implementation of Low-Rank Adaptation (LoRA) applied to BERT for sentiment classification. The LoRA layer is implemented manually without using any PEFT libraries. Only 0.13% of the model parameters are trained, achieving 90.48% validation accuracy on SST-2.

---

## What is LoRA

Full fine-tuning updates every weight in a pretrained model. For a 109M parameter model like BERT, that means storing weights, gradients, and optimizer states simultaneously, which requires significant GPU memory and compute.

LoRA solves this by keeping the original weights frozen and learning a low-rank decomposition of the weight update instead. For a weight matrix W, instead of computing the full update delta W directly, LoRA decomposes it as:

delta W = A x B

where A is [d_in, r] and B is [r, d_out] and r is much smaller than d_in or d_out. The original W is never modified. Only A and B are trained.

The key insight from the paper is that task-specific weight updates have low intrinsic rank. You do not need to update the full matrix to adapt a pretrained model to a new task. A low-rank approximation captures almost all of the meaningful update with a fraction of the parameters.

At initialization, A is set to small random gaussian values and B is set to zero, so the LoRA contribution is exactly zero at the start of training. The forward pass becomes:

h = W(x) + (alpha / r) * (x @ A @ B)

where alpha / r is a scaling factor that controls the magnitude of the LoRA update relative to the frozen weights.

---

## Implementation

LoRA is implemented as a wrapper around any existing nn.Linear layer. The wrapper freezes the original linear layer and adds two trainable parameter matrices A and B.

```python
class LoRALinear(nn.Module):
    def __init__(self, linear, r, alpha):
        super().__init__()
        d_in = linear.weight.shape[1]
        d_out = linear.weight.shape[0]

        self.W = linear
        self.W.weight.requires_grad = False
        if self.W.bias is not None:
            self.W.bias.requires_grad = False

        self.A = nn.Parameter(torch.randn(d_in, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, d_out))
        self.scaling = alpha / r

    def forward(self, x):
        return self.W(x) + self.scaling * (x @ self.A @ self.B)
```

The entire BERT model is frozen first. LoRA is then applied to the query and key projection layers in all 12 transformer blocks, replacing the original nn.Linear layers with LoRALinear wrappers.

---

## Training

Model: BERT base uncased (109.6M parameters)
Dataset: SST-2 (Stanford Sentiment Treebank, binary sentiment classification)
Task: Positive / negative sentiment classification on movie reviews
Trainable parameters: 147,456 (0.13% of total)
Rank: r = 4
Alpha: 8
Learning rate: 1e-3
Batch size: 32
Hardware: RTX 3050 4GB

---

## Results
```
Total parameters      109.6M
Trainable parameters  147K (0.13%)
Validation accuracy   90.48%
```
<img width="1224" height="839" alt="Screenshot From 2026-03-10 20-43-34" src="https://github.com/user-attachments/assets/f5f219ba-5789-4502-8852-a2596505e5b0" />

Full fine-tuning of BERT base on SST-2 typically achieves 92-93% accuracy. This implementation reaches 90.48% while training only 0.13% of the parameters. The model converged in a single epoch, with validation accuracy dropping slightly on epoch 2, indicating fast convergence and the onset of overfitting on the small adapter parameter space.

---

## Project Structure

```
lora-bert/
├── main.py        # model setup, LoRA application, parameter count
├── train.py       # dataset loading, tokenization, training loop
├── plot.py        # results visualization
└── README.md
```

---

The intrinsic dimension argument is what makes LoRA more than just a memory saving trick. The fact that 0.13% of parameters is sufficient to adapt a general language model to a specific task tells you something real about how pretrained representations work. The useful information for a new task already exists in the frozen weights. The adapters are just learning how to surface it.

The gap between 90.48% and the 92-93% from full fine-tuning is the cost of the low-rank constraint. Closing that gap means increasing rank, applying LoRA to more layers including value and feed-forward projections, or tuning the learning rate and scaling factor more carefully.