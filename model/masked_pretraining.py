
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

try:
    import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config

try:
    from graph_attention import GraphAwareEncoder
except ImportError:
    from .graph_attention import GraphAwareEncoder


def temporal_pool(x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    if padding_mask is not None:
        mask = (~padding_mask).float().unsqueeze(-1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return x.mean(dim=1)


def time_warp(x: torch.Tensor, max_shift: float = 0.08) -> torch.Tensor:
    B, T, D = x.shape
    warp = torch.cumsum(torch.randn(B, T, device=x.device) * max_shift, dim=1)
    indices = (torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(0) + warp).clamp(0, T - 1).round().long()
    out = torch.zeros_like(x)
    for b in range(B):
        out[b] = x[b, indices[b]]
    return out


def temporal_dropout(x: torch.Tensor, drop_ratio: float = 0.04) -> torch.Tensor:
    B, T, D = x.shape
    drop_len = max(1, int(T * drop_ratio))
    mask = torch.ones(B, T, 1, device=x.device)
    for b in range(B):
        start = torch.randint(0, max(1, T - drop_len), (1,)).item()
        mask[b, start:start + drop_len] = 0.0
    return x * mask


def augment_landmarks(x: torch.Tensor) -> torch.Tensor:
    x = time_warp(x)
    x = x + torch.randn_like(x) * 0.003
    if torch.rand(1).item() < 0.5:
        x = temporal_dropout(x)
    return x


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.shape[0]
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
        sim = torch.matmul(z, z.T) / self.temperature
        labels = torch.cat([
            torch.arange(B, device=z.device) + B,
            torch.arange(B, device=z.device),
        ])
        sim.fill_diagonal_(-9e15)
        return self.ce(sim, labels)


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int = config.EMBED_DIM, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastivePretraining(nn.Module):
    def __init__(
        self,
        embed_dim: int = config.EMBED_DIM,
        proj_dim: int = 128,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.encoder = GraphAwareEncoder(embed_dim=embed_dim)
        self.projection_head = ProjectionHead(embed_dim=embed_dim, proj_dim=proj_dim)
        self.contrastive_loss = NTXentLoss(temperature=temperature)

    def forward(
        self,
        raw_landmarks: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if raw_landmarks.ndim != 3 or raw_landmarks.size(-1) != config.INPUT_DIM:
            raise ValueError(f"Expected (B, T, {config.INPUT_DIM}), got {tuple(raw_landmarks.shape)}.")

        v1 = augment_landmarks(raw_landmarks)
        v2 = augment_landmarks(raw_landmarks)

        f1 = self.encoder(v1, key_padding_mask=padding_mask)
        f2 = self.encoder(v2, key_padding_mask=padding_mask)

        p1 = self.projection_head(temporal_pool(f1, padding_mask))
        p2 = self.projection_head(temporal_pool(f2, padding_mask))

        return {"loss": self.contrastive_loss(p1, p2)}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 4, config.SEQUENCE_LENGTH, config.EMBED_DIM

    x = torch.randn(B, T, config.INPUT_DIM)
    lengths = torch.tensor([T, T - 20, T, T - 15])
    pad_mask = torch.arange(T).unsqueeze(0) >= lengths.unsqueeze(1)

    model = ContrastivePretraining(embed_dim=D)
    out = model(x, padding_mask=pad_mask)

    assert out["loss"].item() > 0
    out["loss"].backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print("ContrastivePretraining: VERIFIED")