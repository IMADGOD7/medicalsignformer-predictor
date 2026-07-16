"""
Classification head for MedicalSignFormer Stage 3.

This module preserves the TensorFlow reference layers and dropout behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# Handle both relative and absolute imports
try:
    from ..config import NUM_CLASSES, EMBED_DIM, DROPOUT_RATE
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import NUM_CLASSES, EMBED_DIM, DROPOUT_RATE


class ClassificationHead(nn.Module):
    """Classification head with two dense layers and dropout."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = DROPOUT_RATE):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(EMBED_DIM, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(-1) != EMBED_DIM:
            raise ValueError(
                f"Expected input shape (B, {EMBED_DIM}), got {x.shape}."
            )

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


if __name__ == "__main__":
    module = ClassificationHead(num_classes=4, dropout_rate=0.3)
    dummy_input = torch.randn(2, 256)
    output = module(dummy_input)
    print("output", output.shape)
