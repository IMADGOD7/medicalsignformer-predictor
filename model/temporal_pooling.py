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
    def __init__(self, embed_dim: int = config.EMBED_DIM,
                 hidden_dim: int = config.TEMPORAL_POOLING_HIDDEN_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.score = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.size(-1) != self.embed_dim:
            raise ValueError(f"Expected (B, T, {self.embed_dim}), got {tuple(x.shape)}.")

        scores = self.score(x)
        if key_padding_mask is not None:
            if key_padding_mask.shape != x.shape[:2]:
                raise ValueError(f"key_padding_mask {tuple(key_padding_mask.shape)} must match (B, T) = {tuple(x.shape[:2])}.")
            scores = scores.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))

        attention_weights = torch.softmax(scores, dim=1)
        pooled = (x * attention_weights).sum(dim=1)
        return pooled, attention_weights


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 2, config.SEQUENCE_LENGTH
    D = config.EMBED_DIM

    x = torch.randn(B, T, D, requires_grad=True)
    lengths = torch.tensor([T, T - 30])
    pad = torch.arange(T).unsqueeze(0) >= lengths.unsqueeze(1)

    model = TemporalAttentionPooling(embed_dim=D)
    pooled, attn = model(x, key_padding_mask=pad)

    assert pooled.shape == (B, D) and attn.shape == (B, T, 1)
    assert attn.squeeze(-1).sum(dim=1).allclose(torch.ones(B), atol=1e-4)
    assert attn[1, T - 30:, 0].sum() < 1e-4
    pooled.sum().backward()
    print("TemporalAttentionPooling: VERIFIED")
