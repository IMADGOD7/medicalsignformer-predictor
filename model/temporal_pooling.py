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


class TemporalAttentionPooling(nn.Module):


    def __init__(
        self,
        embed_dim: int = config.EMBED_DIM,
        hidden_dim: int = config.TEMPORAL_POOLING_HIDDEN_DIM,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.score = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.score[0].weight)
        nn.init.zeros_(self.score[0].bias)
        nn.init.xavier_uniform_(self.score[2].weight)
        nn.init.zeros_(self.score[2].bias)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected input shape (B, T, {self.embed_dim}), got {tuple(x.shape)}."
            )

        scores = self.score(x)  # (B, T, 1)

        if key_padding_mask is not None:
            if key_padding_mask.shape != x.shape[:2]:
                raise ValueError(
                    f"key_padding_mask shape {tuple(key_padding_mask.shape)} "
                    f"must match (B, T) = {tuple(x.shape[:2])}."
                )
            mask = key_padding_mask.unsqueeze(-1)  # (B, T, 1)
            scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        pooled = (x * attention_weights).sum(dim=1)  # (B, embed_dim)

        return pooled, attention_weights


if __name__ == "__main__":
    # --- Verification: shapes + masking behavior + gradient flow ---
    torch.manual_seed(0)
    batch_size = 2
    seq_len = config.SEQUENCE_LENGTH
    embed_dim = config.EMBED_DIM

    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    lengths = torch.tensor([seq_len, seq_len - 30])
    position_ids = torch.arange(seq_len).unsqueeze(0)
    padding_mask = position_ids >= lengths.unsqueeze(1)

    model = TemporalAttentionPooling(embed_dim=embed_dim)
    pooled, attention_weights = model(x, key_padding_mask=padding_mask)

    print("pooled:            ", pooled.shape)
    print("attention_weights: ", attention_weights.shape)

    assert pooled.shape == (batch_size, embed_dim)
    assert attention_weights.shape == (batch_size, seq_len, 1)
    print("\nShape verification: PASSED")

    # Attention weights must sum to (approximately) 1 over the time axis.
    sums = attention_weights.squeeze(-1).sum(dim=1)
    print(f"\nAttention weight sums per sample (should be ~1.0): {sums.tolist()}")
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    print("Softmax normalization check: PASSED")

    # Padding-safety: padded frames should receive ~0 attention weight.
    padded_weight_sum = attention_weights[1, seq_len - 30:, 0].sum().item()
    print(f"\nSample 1 (last 30 frames padded) - attention mass on padded frames: {padded_weight_sum:.8f}")
    assert padded_weight_sum < 1e-4, "Padded frames received non-trivial attention weight!"
    print("Padding-safety check: PASSED")

    # Gradient flow check.
    loss = pooled.sum()
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