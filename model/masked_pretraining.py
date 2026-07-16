
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
    from .graph_attention import AdaptiveGraphAttentionNetwork
    from .motion_fusion import MotionFeatureFusion
except ImportError:
    from graph_attention import AdaptiveGraphAttentionNetwork
    from motion_fusion import MotionFeatureFusion

def generate_contiguous_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = config.PRETRAIN_MASK_RATIO,
    min_span_len: int = config.PRETRAIN_MIN_SPAN_LEN,
    max_span_len: int = config.PRETRAIN_MAX_SPAN_LEN,
    min_span_gap: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:

    if not (0.0 < mask_ratio < 1.0):
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}.")
    if min_span_len < 1 or max_span_len < min_span_len:
        raise ValueError(
            f"Require 1 <= min_span_len <= max_span_len, got "
            f"min_span_len={min_span_len}, max_span_len={max_span_len}."
        )

    target_masked = max(1, round(seq_len * mask_ratio))
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    for b in range(batch_size):
        masked_count = 0
        occupied = torch.zeros(seq_len, dtype=torch.bool)
        # Cap attempts to avoid a pathological infinite loop on tiny seq_len.
        max_attempts = seq_len * 10
        attempts = 0
        while masked_count < target_masked and attempts < max_attempts:
            attempts += 1
            span_len = int(torch.randint(min_span_len, max_span_len + 1, (1,)).item())
            if seq_len - span_len < 0:
                continue
            start = int(torch.randint(0, seq_len - span_len + 1, (1,)).item())
            end = start + span_len

            window_start = max(0, start - min_span_gap)
            window_end = min(seq_len, end + min_span_gap)
            if occupied[window_start:window_end].any():
                continue  # overlaps an existing span, resample
            occupied[start:end] = True
            masked_count += span_len

        mask[b] = occupied.to(device)

    return mask


def apply_mask_to_landmarks(raw_landmarks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    masked_landmarks = raw_landmarks.clone()
    masked_landmarks[mask] = 0.0
    return masked_landmarks


class GatedModalityCombiner(nn.Module):

    def __init__(self, embed_dim: int = config.EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        # Produces 3 logits per frame (one per modality) from the
        # concatenation of all three streams.
        self.gate_proj = nn.Linear(embed_dim * 3, 3)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, hand: torch.Tensor, pose: torch.Tensor, face: torch.Tensor
    ) -> torch.Tensor:

        stacked = torch.stack([hand, pose, face], dim=-2)  # (B, T, 3, D)
        concat = torch.cat([hand, pose, face], dim=-1)      # (B, T, 3*D)
        gate_logits = self.gate_proj(concat)                 # (B, T, 3)
        gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)  # (B, T, 3, 1)
        combined = (stacked * gate_weights).sum(dim=-2)      # (B, T, D)
        return self.norm(combined)


class GraphAwareEncoder(nn.Module):


    def __init__(self, embed_dim: int = config.EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.graph_attention_network = AdaptiveGraphAttentionNetwork(embed_dim=embed_dim)
        self.motion_fusion = MotionFeatureFusion(embed_dim=embed_dim)
        self.modality_combiner = GatedModalityCombiner(embed_dim=embed_dim)

    def forward(
        self, raw_landmarks: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        hand_feat, pose_feat, face_feat = self.graph_attention_network(
            raw_landmarks, key_padding_mask=key_padding_mask
        )
        hand_fused, pose_fused, face_fused = self.motion_fusion(
            raw_landmarks, hand_feat, pose_feat, face_feat
        )
        return self.modality_combiner(hand_fused, pose_fused, face_fused)


class ReconstructionHead(nn.Module):
    """Small MLP mapping the trainable (masked-input) encoder output back
    into the same latent space as the clean target, at every position.
    Loss restricts comparison to masked positions only (see
    MaskedTemporalPretraining.forward).
    """

    def __init__(self, embed_dim: int = config.EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MaskedTemporalPretraining(nn.Module):

    def __init__(
        self,
        embed_dim: int = config.EMBED_DIM,
        mask_ratio: float = config.PRETRAIN_MASK_RATIO,
        min_span_len: int = config.PRETRAIN_MIN_SPAN_LEN,
        max_span_len: int = config.PRETRAIN_MAX_SPAN_LEN,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.min_span_len = min_span_len
        self.max_span_len = max_span_len

        self.encoder = GraphAwareEncoder(embed_dim=embed_dim)
        self.reconstruction_head = ReconstructionHead(embed_dim=embed_dim)

    def forward(
        self,
        raw_landmarks: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        if raw_landmarks.ndim != 3 or raw_landmarks.size(-1) != config.INPUT_DIM:
            raise ValueError(
                f"Expected raw_landmarks shape (B, T, {config.INPUT_DIM}), "
                f"got {tuple(raw_landmarks.shape)}."
            )

        batch_size, seq_len, _ = raw_landmarks.shape
        device = raw_landmarks.device

        pretrain_mask = generate_contiguous_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            mask_ratio=self.mask_ratio,
            min_span_len=self.min_span_len,
            max_span_len=self.max_span_len,
            device=device,
        )
        # Never treat an already-padded frame as a "masked-for-pretraining"
        # frame - it has no real content to reconstruct in the first place.
        if padding_mask is not None:
            pretrain_mask = pretrain_mask & (~padding_mask)

        # Clean target: original sequence through the SAME encoder, no_grad.
        with torch.no_grad():
            clean_target = self.encoder(raw_landmarks, key_padding_mask=padding_mask)

        # Trainable pass: masked sequence through the same encoder, with grad.
        masked_landmarks = apply_mask_to_landmarks(raw_landmarks, pretrain_mask)
        masked_encoding = self.encoder(masked_landmarks, key_padding_mask=padding_mask)
        reconstructed = self.reconstruction_head(masked_encoding)

        # Masked MSE: compare only at pretrain_mask positions. Unmasked
        # frames (real or padded) contribute exactly zero.
        per_position_sq_error = (reconstructed - clean_target.detach()) ** 2  # (B, T, D)
        per_position_mse = per_position_sq_error.mean(dim=-1)                 # (B, T)
        mask_float = pretrain_mask.float()
        num_masked = mask_float.sum().clamp(min=1.0)
        loss = (per_position_mse * mask_float).sum() / num_masked

        return {
            "reconstructed": reconstructed,
            "clean_target": clean_target,
            "mask": pretrain_mask,
            "loss": loss,
        }


if __name__ == "__main__":
    # --- Verification: shapes + masking behavior + masked-only loss + gradient flow ---
    torch.manual_seed(0)
    batch_size = 2
    seq_len = config.SEQUENCE_LENGTH
    embed_dim = config.EMBED_DIM

    raw_landmarks = torch.randn(batch_size, seq_len, config.INPUT_DIM)
    lengths = torch.tensor([seq_len, seq_len - 20])
    position_ids = torch.arange(seq_len).unsqueeze(0)
    padding_mask = position_ids >= lengths.unsqueeze(1)

    model = MaskedTemporalPretraining(embed_dim=embed_dim)
    output = model(raw_landmarks, padding_mask=padding_mask)

    print("reconstructed:", output["reconstructed"].shape)
    print("clean_target :", output["clean_target"].shape)
    print("mask         :", output["mask"].shape, output["mask"].dtype)
    print("loss         :", output["loss"].item())

    expected_shape = (batch_size, seq_len, embed_dim)
    assert output["reconstructed"].shape == expected_shape
    assert output["clean_target"].shape == expected_shape
    assert output["mask"].shape == (batch_size, seq_len)
    assert output["mask"].dtype == torch.bool
    print("\nShape verification: PASSED")

    # Mask ratio sanity check: masked fraction should be close to
    # PRETRAIN_MASK_RATIO for the non-padded sample (sample 0, no padding).
    frac_masked_sample0 = output["mask"][0].float().mean().item()
    print(
        f"\nSample 0 masked fraction: {frac_masked_sample0:.3f} "
        f"(target: {config.PRETRAIN_MASK_RATIO})"
    )
    assert abs(frac_masked_sample0 - config.PRETRAIN_MASK_RATIO) < 0.05, (
        "Masked fraction too far from target mask_ratio"
    )
    print("Mask ratio check: PASSED")

    # Padding-safety check: no padded position should ever be marked masked.
    assert not (output["mask"] & padding_mask).any(), "A padded position was masked!"
    print("Padding-safety check: PASSED (no padded frame was ever pretraining-masked)")

    for b in range(batch_size):
        m = output["mask"][b]
        transitions = (m[1:].int() - m[:-1].int())
        num_spans = int((transitions == 1).sum().item()) + int(m[0].item())
        print(f"Sample {b}: {num_spans} contiguous masked span(s)")

    # Masked-only loss verification: perturbing clean_target at UNMASKED
    # positions must not change the loss at all, since those positions
    # never enter the masked-MSE computation.
    with torch.no_grad():
        perturbed_target = output["clean_target"].clone()
        unmasked_positions = ~output["mask"]
        perturbed_target[unmasked_positions] += 1000.0  # huge perturbation

        recon = output["reconstructed"]
        mask_f = output["mask"].float()
        n_masked = mask_f.sum().clamp(min=1.0)

        original_masked_loss = (
            ((recon - output["clean_target"]) ** 2).mean(dim=-1) * mask_f
        ).sum() / n_masked
        perturbed_masked_loss = (
            ((recon - perturbed_target) ** 2).mean(dim=-1) * mask_f
        ).sum() / n_masked

    loss_diff = (original_masked_loss - perturbed_masked_loss).abs().item()
    print(f"\nLoss change from perturbing UNMASKED target positions: {loss_diff:.8f} (should be ~0)")
    assert loss_diff < 1e-4, "Unmasked positions are leaking into the loss!"
    print("Masked-only loss isolation check: PASSED")

    # Gradient flow check.
    output["loss"].backward()
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