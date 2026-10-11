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
    from .graph_attention import GraphAwareEncoder
    from .mamba_encoder import MambaTemporalEncoder
    from .temporal_pooling import TemporalAttentionPooling
    from .classifier import ClassificationHead
except ImportError:
    from graph_attention import GraphAwareEncoder
    from mamba_encoder import MambaTemporalEncoder
    from temporal_pooling import TemporalAttentionPooling
    from classifier import ClassificationHead


class MedicalSignFormerV2(nn.Module):
    def __init__(self, embed_dim: int = config.EMBED_DIM, num_classes: int = config.NUM_CLASSES,
                 sequence_length: int = config.SEQUENCE_LENGTH):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        self.encoder = GraphAwareEncoder(embed_dim=embed_dim)
        self.mamba = MambaTemporalEncoder(d_model=embed_dim)
        self.temporal_pooling = TemporalAttentionPooling(embed_dim=embed_dim)
        self.classifier = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes)

    def _build_key_padding_mask(self, x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor | None:
        if lengths is None:
            return None
        position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        lengths = lengths.to(x.device).unsqueeze(1)
        return position_ids >= lengths

    def load_pretrained_encoder(self, checkpoint_path, strict: bool = True) -> None:
        """Load pretrained encoder weights saved by ContrastivePretraining.

        The pretrained checkpoint stores ``model.encoder.state_dict()`` directly,
        so we use ``strict=False`` on the encoder sub-module and report any
        mismatched keys.
        """
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = self.encoder.load_state_dict(state, strict=strict)
        if missing or unexpected:
            print(f"  [load_pretrained_encoder] missing={missing} unexpected={unexpected}")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.size(-1) != config.INPUT_DIM:
            raise ValueError(f"Expected (B, T, {config.INPUT_DIM}), got {tuple(x.shape)}.")
        if x.size(1) != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {x.size(1)}.")

        key_padding_mask = self._build_key_padding_mask(x, lengths)
        latent = self.encoder(x, key_padding_mask=key_padding_mask)
        temporal_features = self.mamba(latent)
        pooled, attention_weights = self.temporal_pooling(temporal_features, key_padding_mask=key_padding_mask)
        logits = self.classifier(pooled)
        return logits, attention_weights


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 2, config.SEQUENCE_LENGTH
    C = config.NUM_CLASSES

    x = torch.randn(B, T, config.INPUT_DIM, requires_grad=True)
    lengths = torch.tensor([T, T - 25])

    model = MedicalSignFormerV2()
    logits, attn = model(x, lengths=lengths)

    assert logits.shape == (B, C) and attn.shape == (B, T, 1)
    assert attn.squeeze(-1).sum(dim=1).allclose(torch.ones(B), atol=1e-4)
    assert attn[1, T - 25:, 0].sum() < 1e-4

    nn.CrossEntropyLoss()(logits, torch.randint(0, C, (B,))).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        model.eval()
        clean = torch.randn(1, T, config.INPUT_DIM)
        l1, _ = model(clean, lengths=torch.tensor([T - 25]))
        perturbed = clean.clone()
        perturbed[:, T - 25:, :] += torch.randn_like(perturbed[:, T - 25:, :]) * 50.0
        l2, _ = model(perturbed, lengths=torch.tensor([T - 25]))
    assert (l1 - l2).abs().max() < 1e-3

    print("MedicalSignFormerV2: VERIFIED")
