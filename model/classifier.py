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

    def __init__(
        self,
        embed_dim: int = config.EMBED_DIM,
        hidden_dim: int = config.CLASSIFIER_HIDDEN_DIM,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.CLASSIFIER_DROPOUT,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected input shape (B, {self.embed_dim}), got {tuple(x.shape)}."
            )

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.fc2(x)


if __name__ == "__main__":
    # --- Verification: shapes + "logits only" check + gradient flow ---
    torch.manual_seed(0)
    batch_size = 4
    embed_dim = config.EMBED_DIM
    num_classes = config.NUM_CLASSES

    x = torch.randn(batch_size, embed_dim, requires_grad=True)
    model = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes)
    logits = model(x)

    print("logits:", logits.shape)
    assert logits.shape == (batch_size, num_classes)
    print("\nShape verification: PASSED")

    row_sums = logits.sum(dim=-1)
    print(f"\nLogit row sums (should NOT be ~1.0, confirming no softmax baked in): {row_sums.tolist()}")
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2), (
        "Logits sum to ~1 per row - softmax may have been accidentally applied inside the model!"
    )
    print("Logits-only (no internal softmax) check: PASSED")

    # Confirm CrossEntropyLoss (which expects raw logits) works directly.
    dummy_labels = torch.randint(0, num_classes, (batch_size,))
    loss = nn.CrossEntropyLoss()(logits, dummy_labels)
    print(f"\nCrossEntropyLoss on raw logits: {loss.item():.4f} (finite, no shape error)")

    loss.backward()
    missing_grad = [
        name for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    if missing_grad:
        print(f"\nWARNING: {len(missing_grad)} parameter(s) received NO gradient:")
        for name in missing_grad:
            print(f"  - {name}")
    else:
        print("\nGradient flow verification: PASSED (all parameters received gradients)")

    upstream_ok = x.grad is not None and torch.any(x.grad != 0)
    print(f"Upstream gradient reach-through (input x): {'PASSED' if upstream_ok else 'FAILED'}")