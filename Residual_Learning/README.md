# Implementation of *Deep Residual Learning for Image Recognition*

## Paper Overview

This repository contains a from-scratch implementation of the paper  
**Deep Residual Learning for Image Recognition**.

The paper shows that simply stacking more layers in a CNN can make optimization worse, even when vanishing/exploding gradients are under control. The core idea of ResNet is to reformulate learning so that stacked layers learn a **residual mapping** instead of a full transformation.

Instead of forcing layers to learn `H(x)`, the network learns:
`F(x) = H(x) - x`

and reconstructs the original mapping via:
`H(x) = F(x) + x`


This formulation makes identity mappings easy to represent and significantly improves optimization in deep networks.

This repository focuses on implementing this idea manually, training the model from scratch, and validating the effect of residual connections through controlled ablations.

---

## `basic_block.py` : Residual Block from Scratch

The core unit of the implementation is a manually written **ResNet basic block**.

### Block Structure
Conv → BatchNorm → ReLU → Conv → BatchNorm → + shortcut(x) → ReLU
<img width="511" height="272" alt="image" src="https://github.com/user-attachments/assets/b0b3a170-b3a2-498f-88f7-0a2fcc0a3d8c" />


- The residual branch learns the transformation `F(x)`
- The shortcut path carries the input unchanged
- A 1×1 convolution + BatchNorm is used only when spatial size or channel count changes

If the residual branch learns zero, the block reduces to an identity mapping. This single block captures the entire conceptual contribution of the ResNet paper.

---

## `resnet.py` : ResNet-18 (CIFAR-10)

ResNet-18 is constructed by stacking the basic block into four stages.

### Overall Network Flow
Conv → BatchNorm → ReLU <br>
→ Layer1 → Layer2 → Layer3 → Layer4<br>
→ Global Average Pool → Linear


### Architectural Details

- CIFAR-style stem using a 3×3 convolution with stride 1
- No max-pooling
- Four residual stages
- Channel progression: `16 → 32 → 64 → 128`
- Downsampling occurs only in the first block of each stage
- No pretrained weights
- No torchvision abstractions

The network structure directly follows the design logic described in the paper.

---

## Training on CIFAR-10

The ResNet-18 model is trained from scratch on the CIFAR-10 dataset.

### Training Setup

- Optimizer: SGD with momentum
- Loss: Cross-entropy with label smoothing
- Regularization: Weight decay and data augmentation
- Learning rate schedule: Multi-step decay

### Results

- Training loss decreases from approximately `1.89` to `0.6`
- Test accuracy reaches approximately `91%`
- Proper learning rate scheduling and sufficient epochs are required for convergence
<img width="705" height="303" alt="image" src="https://github.com/user-attachments/assets/85e86c69-5a09-4f15-b3f8-0abf5a5d165b" />


This confirms that optimization dynamics and scheduling are critical once residual connections are in place.

---

### Summary
This repository demonstrates:

- Residual blocks implemented manually to understand how identity shortcuts and residual mappings actually work
- ResNet-18 built from basic blocks without relying on high-level libraries or pretrained components
- Shows that training dynamics and optimization matter more than simply increasing network depth

  