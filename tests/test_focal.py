import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_timm_models import PyTorchFocalLoss

# Test PyTorch Focal Loss
weight = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float32)
loss_fn = PyTorchFocalLoss(alpha=weight, gamma=2.0)

# Batch of 3 samples with 3 classes
logits = torch.tensor([
    [3.0, 0.1, -1.0],
    [0.2, 4.0, 0.5],
    [-0.5, 0.2, 2.8]
], dtype=torch.float32)
targets = torch.tensor([0, 1, 2], dtype=torch.long)

loss_val = loss_fn(logits, targets)
print(f"PyTorch Focal Loss computation OK - loss: {loss_val.item():.4f}")
assert loss_val.item() > 0.0, "Loss value should be strictly positive"

print("All tests passed!")

