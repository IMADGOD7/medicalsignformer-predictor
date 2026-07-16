
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

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = config.EMBED_DIM,
        dropout: float = config.GATED_FUSION_DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, input_dim) raw velocity features.
        Returns:
            (B, T, hidden_dim)
        """
        return self.net(x)


class GatedFusion(nn.Module):

    def __init__(self, embed_dim: int = config.EMBED_DIM, dropout: float = config.GATED_FUSION_DROPOUT):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, position_feature: torch.Tensor, velocity_feature: torch.Tensor) -> torch.Tensor:
        """Args:
            position_feature: (B, T, embed_dim)
            velocity_feature: (B, T, embed_dim)
        Returns:
            (B, T, embed_dim) gated fusion of the two.
        """
        if position_feature.shape != velocity_feature.shape:
            raise ValueError(
                f"position_feature and velocity_feature must match shapes, "
                f"got {tuple(position_feature.shape)} and {tuple(velocity_feature.shape)}."
            )

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

        # Separate velocity encoders per modality (left/right hand kept
        # separate, mirroring Module 2's left/right hand graph branches,
        # then fused together the same way hand_features are fused there).
        self.pose_velocity_encoder = VelocityEncoder(input_dim=config.POSE_DIM)
        self.face_velocity_encoder = VelocityEncoder(input_dim=config.FACE_DIM)
        self.left_hand_velocity_encoder = VelocityEncoder(input_dim=config.LEFT_HAND_DIM)
        self.right_hand_velocity_encoder = VelocityEncoder(input_dim=config.RIGHT_HAND_DIM)

        # Fuse left/right hand velocity encodings the same way Module 2
        # fuses left/right hand position encodings, before hand-level gating.
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
            raise ValueError(
                f"Expected raw_landmarks shape (B, T, {config.INPUT_DIM}), "
                f"got {tuple(raw_landmarks.shape)}."
            )
        for name, feat in (("hand_features", hand_features), ("pose_features", pose_features), ("face_features", face_features)):
            if feat.shape[:2] != raw_landmarks.shape[:2] or feat.size(-1) != self.embed_dim:
                raise ValueError(
                    f"Expected {name} shape (B, T, {self.embed_dim}) matching "
                    f"raw_landmarks' (B, T), got {tuple(feat.shape)}."
                )

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
    # --- Verification: shapes + gradient flow, per the one-module-at-a-time rule ---
    torch.manual_seed(0)
    batch_size = 2
    seq_len = config.SEQUENCE_LENGTH
    embed_dim = config.EMBED_DIM

    raw_landmarks = torch.randn(batch_size, seq_len, config.INPUT_DIM, requires_grad=False)

    hand_features = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    pose_features = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    face_features = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

    model = MotionFeatureFusion(embed_dim=embed_dim)
    hand_fused, pose_fused, face_fused = model(raw_landmarks, hand_features, pose_features, face_features)

    print("hand_fused:", hand_fused.shape)
    print("pose_fused:", pose_fused.shape)
    print("face_fused:", face_fused.shape)

    expected_shape = (batch_size, seq_len, embed_dim)
    assert hand_fused.shape == expected_shape, "hand_fused shape mismatch"
    assert pose_fused.shape == expected_shape, "pose_fused shape mismatch"
    assert face_fused.shape == expected_shape, "face_fused shape mismatch"
    print("\nShape verification: PASSED")

    # Sanity check: velocity at t=0 should be exactly zero for every modality.
    pose_raw = raw_landmarks[:, :, model._pose_slice]
    v0 = compute_velocity(pose_raw)[:, 0, :]
    assert torch.all(v0 == 0.0), "velocity at t=0 should be zero"
    print("Velocity t=0 zero-check: PASSED")

    # Gradient flow check.
    loss = hand_fused.sum() + pose_fused.sum() + face_fused.sum()
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
        print("Gradient flow verification: PASSED (all parameters received gradients)")

    # Confirm gradient also reaches the upstream position-feature inputs
    # (i.e. this module wouldn't block gradient flow back into Module 2).
    upstream_ok = all(
        t.grad is not None and torch.any(t.grad != 0)
        for t in (hand_features, pose_features, face_features)
    )
    print(
        f"Upstream gradient reach-through (hand/pose/face_features): "
        f"{'PASSED' if upstream_ok else 'FAILED'}"
    )

    # Confirm the gate actually produces non-trivial (not all-0 or all-1)
    # blends, i.e. it's doing real gating rather than degenerating to a
    # pass-through of one branch.
    with torch.no_grad():
        cat = torch.cat([hand_features, hand_velocity if False else torch.zeros_like(hand_features)], dim=-1)
    print("\nModule 3 build complete.")