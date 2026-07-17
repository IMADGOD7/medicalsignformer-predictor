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

try:
    from .masked_pretraining import GraphAwareEncoder
    from .mamba_encoder import MambaTemporalEncoder
    from .temporal_pooling import TemporalAttentionPooling
    from .classifier import ClassificationHead
except ImportError:
    from masked_pretraining import GraphAwareEncoder
    from mamba_encoder import MambaTemporalEncoder
    from temporal_pooling import TemporalAttentionPooling
    from classifier import ClassificationHead


class MedicalSignFormerV2(nn.Module):

    def __init__(
        self,
        embed_dim: int = config.EMBED_DIM,
        num_classes: int = config.NUM_CLASSES,
        sequence_length: int = config.SEQUENCE_LENGTH,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        self.encoder = GraphAwareEncoder(embed_dim=embed_dim)
        self.mamba = MambaTemporalEncoder(d_model=embed_dim)
        self.temporal_pooling = TemporalAttentionPooling(embed_dim=embed_dim)
        self.classifier = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes)

    def _build_key_padding_mask(
        self, x: torch.Tensor, lengths: torch.Tensor | None
    ) -> torch.Tensor | None:

        if lengths is None:
            return None

        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        lengths = lengths.to(x.device).unsqueeze(1)
        return position_ids >= lengths  # True where padded

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if x.ndim != 3 or x.size(-1) != config.INPUT_DIM:
            raise ValueError(
                f"Expected input shape (B, T, {config.INPUT_DIM}), got {tuple(x.shape)}."
            )
        if x.size(1) != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {x.size(1)}."
            )

        key_padding_mask = self._build_key_padding_mask(x, lengths)

        latent = self.encoder(x, key_padding_mask=key_padding_mask)  # (B, T, D)
        temporal_features = self.mamba(latent)  # (B, T, D) - no mask arg, see module docstring
        pooled, attention_weights = self.temporal_pooling(
            temporal_features, key_padding_mask=key_padding_mask
        )  # pooled: (B, D), attention_weights: (B, T, 1)
        logits = self.classifier(pooled)  # (B, num_classes)

        return logits, attention_weights

    def load_pretrained_encoder(self, checkpoint_path: str | Path, strict: bool = True) -> None:
        """Load Stage 1 pretraining weights into self.encoder only - the
        Mamba encoder, temporal pooling, and classifier remain freshly
        initialized for Stage 2 fine-tuning, per the spec's two-stage
        training design ("Load pretrained encoder. Attach classification
        head.")."""
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(state_dict, strict=strict)


if __name__ == "__main__":
    # --- Verification: shapes, gradient flow, and the padding-safety chain ---
    torch.manual_seed(0)
    batch_size = 2
    seq_len = config.SEQUENCE_LENGTH
    num_classes = config.NUM_CLASSES

    x = torch.randn(batch_size, seq_len, config.INPUT_DIM, requires_grad=True)
    lengths = torch.tensor([seq_len, seq_len - 25])
    position_ids = torch.arange(seq_len).unsqueeze(0)
    padding_mask = position_ids >= lengths.unsqueeze(1)

    model = MedicalSignFormerV2()
    logits, attention_weights = model(x, lengths=lengths)

    print("logits:            ", logits.shape)
    print("attention_weights: ", attention_weights.shape)

    assert logits.shape == (batch_size, num_classes)
    assert attention_weights.shape == (batch_size, seq_len, 1)
    print("\nShape verification: PASSED")

    # Attention weights should sum to ~1 and assign ~0 to padded frames
    # (re-verifying the full chain, not just Module 6 in isolation).
    sums = attention_weights.squeeze(-1).sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    padded_mass = attention_weights[1, seq_len - 25:, 0].sum().item()
    print(f"\nAttention mass on sample 1's padded frames (should be ~0): {padded_mass:.8f}")
    assert padded_mass < 1e-4
    print("End-to-end padding-safety (attention) check: PASSED")

    # Logits-only check (re-verified at the full-model level).
    row_sums = logits.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2), (
        "Logits sum to ~1 - softmax may be leaking in somewhere in the chain!"
    )
    print("Logits-only (no internal softmax anywhere in the chain) check: PASSED")

    # Gradient flow across the WHOLE assembled model.
    loss = nn.CrossEntropyLoss()(logits, torch.randint(0, num_classes, (batch_size,)))
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
        print(f"\nGradient flow verification: PASSED (all {len(list(model.named_parameters()))} parameters received gradients)")

    # End-to-end padding-safety stress test: perturbing ONLY the padded
    # region of sample 1's input should not change that sample's logits at
    # all - not because of any explicit masking of the input, but because
    # (a) Module 5's causal scan can't let trailing padding affect earlier
    # real frames, and (b) Module 6 excludes padded frames from pooling
    # regardless. This tests the property end-to-end rather than trusting
    # each module's own isolated guarantee.
    with torch.no_grad():
        model.eval()
        x_clean = torch.randn(1, seq_len, config.INPUT_DIM)
        sample_length = torch.tensor([seq_len - 25])

        logits_clean, _ = model(x_clean, lengths=sample_length)

        x_perturbed = x_clean.clone()
        x_perturbed[:, seq_len - 25:, :] += torch.randn_like(x_perturbed[:, seq_len - 25:, :]) * 50.0
        logits_perturbed, _ = model(x_perturbed, lengths=sample_length)

        logit_diff = (logits_clean - logits_perturbed).abs().max().item()

    print(f"\nEnd-to-end padding-safety (logits) check: max logit diff from perturbing only padded frames: {logit_diff:.8f} (should be ~0)")
    assert logit_diff < 1e-3, "Perturbing only the padded region changed the final logits - padding is leaking into the prediction!"
    print("End-to-end padding-safety (logits) check: PASSED")

    print("\nFull MedicalSignFormerV2 assembly: VERIFIED")