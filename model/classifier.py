from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

try:
    import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int = config.EMBED_DIM, hidden_dim: int = config.CLASSIFIER_HIDDEN_DIM,
                 num_classes: int = config.NUM_CLASSES, dropout_rate: float = config.CLASSIFIER_DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(-1) != self.fc1.in_features:
            raise ValueError(f"Expected (B, {self.fc1.in_features}), got {tuple(x.shape)}.")
        x = self.dropout(self.activation(self.fc1(x)))
        return self.fc2(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    D, C = config.EMBED_DIM, config.NUM_CLASSES

    x = torch.randn(4, D, requires_grad=True)
    model = ClassificationHead(embed_dim=D, num_classes=C)
    logits = model(x)

    assert logits.shape == (4, C)
    nn.CrossEntropyLoss()(logits, torch.randint(0, C, (4,))).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print("ClassificationHead: VERIFIED")
