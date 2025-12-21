# 1. Implementation of *Distilling the Knowledge in a Neural Network*

## Paper Overview

This project is a hands-on implementation of the paper **“Distilling the Knowledge in a Neural Network” (Hinton et al.)**, which introduces the idea of transferring knowledge from a trained neural network to another model using its output behavior instead of hard labels.

The key insight of the paper is that a model’s logits contain much richer information than one-hot labels. By training a student model to match these softened outputs (via temperature-scaled softmax), the student can learn how the teacher generalizes, even without access to ground-truth labels.

---

## `teacher.py` : Supervised Teacher Model

The teacher model is trained in a standard supervised setting.

- **Dataset:** CIFAR-10  
- **Model:** ResNet-18 (with CIFAR-style stem)  
- **Training:** Images + labels using cross-entropy loss  

The teacher reaches ~**87% validation accuracy in 10 epochs** with stable convergence.  
After training, the model is **frozen** and treated as a fixed function that maps images → logits. These logits act as the source of knowledge for distillation.

---

## `student.py` : Knowledge Distillation

The student model is trained purely using **knowledge distillation**.

- **Model:** ResNet-18  
- **Supervision:** Teacher logits only (no image labels)  
- **Method:** Softmax with temperature + KL divergence loss  

During training, the student never sees CIFAR-10 labels. Instead, it learns to match the **full output distribution** of the teacher, inheriting its uncertainty and class relationships.

After just 10 epochs:
- Distillation loss drops smoothly from ~5.4 → ~0.2  
- Validation accuracy reaches ~**85–86%**, despite zero label supervision  
<img width="598" height="433" alt="Screenshot From 2025-12-17 21-02-42" src="https://github.com/user-attachments/assets/815c6fe4-8716-42f6-93a2-d0f6195b5283" />

This demonstrates that a large portion of the learning signal is encoded in the teacher’s output behavior rather than the labels themselves.
---

## Why Knowledge Distillation Matters

In real-world systems, large models are often too slow or expensive to deploy. Knowledge distillation allows a large, high-capacity model to be trained once, then its behavior to be transferred into smaller, faster models suitable for deployment on web services, mobile devices, and edge hardware, while retaining most of the performance.

---

#### Note on Model Size
Both the teacher and student models in this project use **ResNet-18** due to limited compute resources. Ideally, distillation is performed from a larger model (e.g. ResNet-34) to a smaller one, but the same distillation logic applies and works identically in that setting.
