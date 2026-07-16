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


def enable_mc_dropout(model: nn.Module) -> None:

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    *forward_args,
    num_samples: int = config.MC_DROPOUT_NUM_SAMPLES,
    **forward_kwargs,
) -> dict[str, torch.Tensor]:

    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")

    all_probs = []
    with torch.no_grad():
        for _ in range(num_samples):
            output = model(*forward_args, **forward_kwargs)
            logits = _extract_logits(output)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

    stacked = torch.stack(all_probs, dim=0)  # (num_samples, B, num_classes)

    mean_probs = stacked.mean(dim=0)  # (B, num_classes)
    variance = stacked.var(dim=0, unbiased=False)  # (B, num_classes)

    prediction = mean_probs.argmax(dim=-1)  # (B,)
    confidence = mean_probs.gather(1, prediction.unsqueeze(-1)).squeeze(-1)  # (B,)
    predicted_class_variance = variance.gather(1, prediction.unsqueeze(-1)).squeeze(-1)  # (B,)

    eps = 1e-12  # avoid log(0) for classes with ~zero mean probability
    entropy = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=-1)  # (B,)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "mean_probs": mean_probs,
        "variance": variance,
        "predicted_class_variance": predicted_class_variance,
        "entropy": entropy,
    }


def _extract_logits(output: torch.Tensor | tuple | dict) -> torch.Tensor:

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Cannot extract logits from output of type {type(output)}.")


if __name__ == "__main__":
    # --- Verification using a small dummy classifier with dropout ---
    torch.manual_seed(0)

    class _DummyClassifier(nn.Module):
        """Minimal stand-in for a trained model - just enough dropout and
        nonlinearity to produce genuine sample-to-sample variation."""

        def __init__(self, in_dim: int = 32, hidden_dim: int = 64, num_classes: int = config.NUM_CLASSES, p: float = 0.5):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.dropout1 = nn.Dropout(p)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.dropout2 = nn.Dropout(p)
            self.fc3 = nn.Linear(hidden_dim, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            return self.fc3(x)

    batch_size = 4
    in_dim = 32
    num_classes = config.NUM_CLASSES
    x = torch.randn(batch_size, in_dim)

    dummy_model = _DummyClassifier(in_dim=in_dim, num_classes=num_classes, p=0.5)
    enable_mc_dropout(dummy_model)

    dropout_modes = [m.training for m in dummy_model.modules() if isinstance(m, nn.Dropout)]
    print(f"Dropout submodules in train mode (should all be True): {dropout_modes}")
    assert all(dropout_modes), "enable_mc_dropout failed to activate dropout stochasticity"
    assert not dummy_model.fc1.training if hasattr(dummy_model.fc1, "training") else True  # Linear has no meaningful train/eval distinction, sanity only

    result = mc_dropout_predict(dummy_model, x, num_samples=config.MC_DROPOUT_NUM_SAMPLES)

    print("\nprediction:               ", result["prediction"].shape, result["prediction"].dtype)
    print("confidence:                ", result["confidence"].shape)
    print("mean_probs:                ", result["mean_probs"].shape)
    print("variance:                  ", result["variance"].shape)
    print("predicted_class_variance:  ", result["predicted_class_variance"].shape)
    print("entropy:                   ", result["entropy"].shape)

    assert result["prediction"].shape == (batch_size,)
    assert result["confidence"].shape == (batch_size,)
    assert result["mean_probs"].shape == (batch_size, num_classes)
    assert result["variance"].shape == (batch_size, num_classes)
    assert result["predicted_class_variance"].shape == (batch_size,)
    assert result["entropy"].shape == (batch_size,)
    print("\nShape verification: PASSED")

    # mean_probs must be valid probability distributions (sum to 1, all >= 0).
    prob_sums = result["mean_probs"].sum(dim=-1)
    print(f"\nmean_probs row sums (should be ~1.0): {prob_sums.tolist()}")
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)
    assert (result["mean_probs"] >= 0).all()
    print("Probability validity check: PASSED")

    mean_variance = result["variance"].mean().item()
    print(f"\nMean per-class variance across {config.MC_DROPOUT_NUM_SAMPLES} samples: {mean_variance:.6f} (should be > 0)")
    assert mean_variance > 1e-6, "Variance is ~0 - dropout may not actually be active during sampling!"
    print("Stochasticity check: PASSED (dropout is genuinely sampling)")

    deterministic_model = _DummyClassifier(in_dim=in_dim, num_classes=num_classes, p=0.0)
    enable_mc_dropout(deterministic_model)  # no-op for stochasticity here since p=0
    det_result = mc_dropout_predict(deterministic_model, x, num_samples=config.MC_DROPOUT_NUM_SAMPLES)
    det_variance = det_result["variance"].mean().item()
    print(f"\nMean per-class variance with dropout p=0 (negative control): {det_variance:.10f} (should be ~0)")
    assert det_variance < 1e-8, "Variance is non-zero even with dropout disabled - unexpected randomness source!"
    print("Negative control check: PASSED (variance traces back to dropout specifically)")

    # Entropy sanity: a near-uniform mean_probs distribution should have
    # HIGH entropy; a near-one-hot distribution should have LOW entropy.
    uniform_probs = torch.full((1, num_classes), 1.0 / num_classes)
    onehot_probs = torch.zeros((1, num_classes))
    onehot_probs[0, 0] = 1.0 - 1e-6
    onehot_probs[0, 1:] = 1e-6 / (num_classes - 1)

    uniform_entropy = -(uniform_probs * torch.log(uniform_probs + 1e-12)).sum(dim=-1).item()
    onehot_entropy = -(onehot_probs * torch.log(onehot_probs + 1e-12)).sum(dim=-1).item()
    print(f"\nEntropy of a uniform distribution over {num_classes} classes: {uniform_entropy:.4f} (should be high, ~log({num_classes})={torch.log(torch.tensor(float(num_classes))).item():.4f})")
    print(f"Entropy of a near-one-hot distribution: {onehot_entropy:.6f} (should be near 0)")
    assert uniform_entropy > onehot_entropy
    print("Entropy sanity check: PASSED")

    # This module has no trainable parameters of its own - confirm no_grad
    # was respected (no gradient tracking occurred during MC sampling).
    print(f"\nresult['mean_probs'].requires_grad (should be False): {result['mean_probs'].requires_grad}")
    assert not result["mean_probs"].requires_grad
    print("No-grad (inference-only) check: PASSED")