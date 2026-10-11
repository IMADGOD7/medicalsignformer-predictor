
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


def compute_velocity(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected (B, T, feature_dim), got {tuple(x.shape)}.")
    velocity = torch.zeros_like(x)
    velocity[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    return velocity


class VelocityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = config.EMBED_DIM, dropout: float = config.GATED_FUSION_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(nn.Module):
    def __init__(self, embed_dim: int = config.EMBED_DIM, dropout: float = config.GATED_FUSION_DROPOUT):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, position_feature: torch.Tensor, velocity_feature: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(torch.cat([position_feature, velocity_feature], dim=-1)))
        fused = gate * position_feature + (1.0 - gate) * velocity_feature
        fused = self.dropout(fused)
        return self.norm(fused)


class MotionFeatureFusion(nn.Module):
    def __init__(self, embed_dim: int = config.EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        self._pose_slice = slice(0, config.POSE_DIM)
        self._face_slice = slice(config.POSE_DIM, config.POSE_DIM + config.FACE_DIM)
        self._left_hand_slice = slice(
            config.POSE_DIM + config.FACE_DIM,
            config.POSE_DIM + config.FACE_DIM + config.LEFT_HAND_DIM,
        )
        self._right_hand_slice = slice(
            config.POSE_DIM + config.FACE_DIM + config.LEFT_HAND_DIM,
            config.POSE_DIM + config.FACE_DIM + config.LEFT_HAND_DIM + config.RIGHT_HAND_DIM,
        )

        self.pose_velocity_encoder = VelocityEncoder(input_dim=config.POSE_DIM)
        self.face_velocity_encoder = VelocityEncoder(input_dim=config.FACE_DIM)
        self.left_hand_velocity_encoder = VelocityEncoder(input_dim=config.LEFT_HAND_DIM)
        self.right_hand_velocity_encoder = VelocityEncoder(input_dim=config.RIGHT_HAND_DIM)

        self.hand_velocity_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.hand_gated_fusion = GatedFusion(embed_dim=embed_dim)
        self.pose_gated_fusion = GatedFusion(embed_dim=embed_dim)
        self.face_gated_fusion = GatedFusion(embed_dim=embed_dim)

    def forward(
        self,
        raw_landmarks: torch.Tensor,
        hand_features: torch.Tensor,
        pose_features: torch.Tensor,
        face_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if raw_landmarks.ndim != 3 or raw_landmarks.size(-1) != config.INPUT_DIM:
            raise ValueError(f"Expected (B, T, {config.INPUT_DIM}), got {tuple(raw_landmarks.shape)}.")
        for name, feat in (("hand_features", hand_features), ("pose_features", pose_features), ("face_features", face_features)):
            if feat.shape[:2] != raw_landmarks.shape[:2] or feat.size(-1) != self.embed_dim:
                raise ValueError(f"Expected {name} (B, T, {self.embed_dim}), got {tuple(feat.shape)}.")

        pose_raw = raw_landmarks[:, :, self._pose_slice]
        face_raw = raw_landmarks[:, :, self._face_slice]
        left_hand_raw = raw_landmarks[:, :, self._left_hand_slice]
        right_hand_raw = raw_landmarks[:, :, self._right_hand_slice]

        pose_velocity = self.pose_velocity_encoder(compute_velocity(pose_raw))
        face_velocity = self.face_velocity_encoder(compute_velocity(face_raw))
        left_hand_velocity = self.left_hand_velocity_encoder(compute_velocity(left_hand_raw))
        right_hand_velocity = self.right_hand_velocity_encoder(compute_velocity(right_hand_raw))
        hand_velocity = self.hand_velocity_fusion(
            torch.cat([left_hand_velocity, right_hand_velocity], dim=-1)
        )

        hand_fused = self.hand_gated_fusion(hand_features, hand_velocity)
        pose_fused = self.pose_gated_fusion(pose_features, pose_velocity)
        face_fused = self.face_gated_fusion(face_features, face_velocity)

        return hand_fused, pose_fused, face_fused


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 2, config.SEQUENCE_LENGTH
    D = config.EMBED_DIM

    raw = torch.randn(B, T, config.INPUT_DIM)
    h_f = torch.randn(B, T, D, requires_grad=True)
    p_f = torch.randn(B, T, D, requires_grad=True)
    f_f = torch.randn(B, T, D, requires_grad=True)

    model = MotionFeatureFusion(embed_dim=D)
    h, p, f = model(raw, h_f, p_f, f_f)

    assert h.shape == p.shape == f.shape == (B, T, D)
    assert compute_velocity(raw)[:, 0, :].abs().sum() == 0

    (h + p + f).sum().backward()
    assert all(g.grad is not None for g in (h_f, p_f, f_f))
    print("MotionFeatureFusion: VERIFIED")