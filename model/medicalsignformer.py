"""
MedicalSignFormer model assembly for Stage 3.

This module composes the Stage 3 PyTorch implementation using the exact
architecture from the TensorFlow reference notebook.

Padding-mask support added: forward() now accepts an optional `lengths`
tensor. When provided, a boolean key_padding_mask (True = padded position) is
built and threaded through the cross-attention encoder and MSAM, and global
average pooling becomes length-aware (averages only real frames, not padded
zeros). `lengths` is optional and defaults to None for backward compatibility
with existing smoke tests / dummy-input calls that don't have real sequence
lengths - in that case no masking is applied, matching the previous
behavior exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# Handle both relative and absolute imports
try:
    from ..config import NUM_CLASSES, INPUT_DIM, EMBED_DIM, NUM_HEADS, FF_DIM, SEQUENCE_LENGTH, DROPOUT_RATE
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import NUM_CLASSES, INPUT_DIM, EMBED_DIM, NUM_HEADS, FF_DIM, SEQUENCE_LENGTH, DROPOUT_RATE

from .embedding import MultiModalEmbedding
from .positional_encoding import PositionalEncoding
from .transformer_encoder import FullCrossAttentionEncoder
from .medical_semantic_attention import MedicalSemanticAttention
from .classifier import ClassificationHead


class MedicalSignFormer(nn.Module):
    """Full MedicalSignFormer model matching the TensorFlow prototype."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        sequence_length: int = SEQUENCE_LENGTH,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        self.embedding = MultiModalEmbedding(embed_dim=embed_dim, input_dim=INPUT_DIM)
        self.positional_encoding = PositionalEncoding(
            sequence_length=sequence_length,
            embed_dim=embed_dim,
        )
        self.encoder = FullCrossAttentionEncoder(
            embed_dim=embed_dim,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
        )
        self.msam = MedicalSemanticAttention(embed_dim=embed_dim)
        self.classifier = ClassificationHead(num_classes=num_classes, dropout_rate=DROPOUT_RATE)

    def _build_key_padding_mask(
        self, x: torch.Tensor, lengths: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Build a boolean key_padding_mask of shape (B, T) from lengths.

        True marks a padded (non-real) position, following the convention
        expected by nn.MultiheadAttention's key_padding_mask argument.

        Args:
            x: Input tensor of shape (B, T, INPUT_DIM), used only for shape.
            lengths: Tensor of shape (B,) giving the real (pre-padding)
                sequence length of each sample, or None.

        Returns:
            Boolean mask of shape (B, T), or None if lengths was None.
        """
        if lengths is None:
            return None

        batch_size, seq_len, _ = x.shape
        # position_ids: (1, T) broadcast against lengths: (B, 1) -> (B, T)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        lengths = lengths.to(x.device).unsqueeze(1)
        key_padding_mask = position_ids >= lengths  # True where padded
        return key_padding_mask

    def _masked_mean_pool(
        self, features: torch.Tensor, key_padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Average-pool over the time axis, ignoring padded positions.

        Args:
            features: Tensor of shape (B, T, embed_dim).
            key_padding_mask: Boolean mask of shape (B, T), True = padded,
                or None to fall back to plain mean over all positions
                (matches the original, pre-masking behavior).

        Returns:
            Pooled tensor of shape (B, embed_dim).
        """
        if key_padding_mask is None:
            return features.mean(dim=1)

        real_mask = (~key_padding_mask).float().unsqueeze(-1)  # (B, T, 1), 1=real, 0=pad
        summed = (features * real_mask).sum(dim=1)  # (B, embed_dim)
        counts = real_mask.sum(dim=1).clamp(min=1.0)  # (B, 1), avoid div-by-zero
        return summed / counts

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, {INPUT_DIM}), got {x.shape}.")

        if x.size(1) != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {x.size(1)}."
            )

        if x.size(2) != INPUT_DIM:
            raise ValueError(
                f"Expected feature dimension {INPUT_DIM}, got {x.size(2)}."
            )

        key_padding_mask = self._build_key_padding_mask(x, lengths)

        hand, pose, face = self.embedding(x)
        hand = self.positional_encoding(hand)
        pose = self.positional_encoding(pose)
        face = self.positional_encoding(face)

        hand, pose, face = self.encoder(hand, pose, face, key_padding_mask=key_padding_mask)
        features, attention = self.msam(hand, pose, face, key_padding_mask=key_padding_mask)

        pooled = self._masked_mean_pool(features, key_padding_mask)
        output = self.classifier(pooled)

        return output, attention


if __name__ == "__main__":
    batch_size = 2
    sequence_length = SEQUENCE_LENGTH
    dummy_input = torch.randn(batch_size, sequence_length, 1629)
    # Exercise the masking path with varied dummy lengths (not all full-length).
    dummy_lengths = torch.tensor([sequence_length, max(1, sequence_length - 15)])

    model = MedicalSignFormer(num_classes=NUM_CLASSES, sequence_length=sequence_length, embed_dim=EMBED_DIM)

    logits, attention = model(dummy_input, lengths=dummy_lengths)
    print("logits (with lengths)", logits.shape)
    print("attention (with lengths)", attention.shape)

    # Confirm backward compatibility: still works with no lengths passed.
    logits_no_mask, attention_no_mask = model(dummy_input)
    print("logits (no lengths, no masking)", logits_no_mask.shape)