from __future__ import annotations

import sys
from pathlib import Path

import torch
from thop import profile

ROOT = Path(__file__).parent


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {"total": total, "trainable": trainable, "non_trainable": non_trainable}


def estimate_model_size_mb(param_count: int, bytes_per_param: int = 4) -> float:
    return (param_count * bytes_per_param) / (1024 ** 2)


def profile_model(model: torch.nn.Module, dummy_input: torch.Tensor) -> dict:
    model.eval()
    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2
        return {"macs": macs, "flops": flops, "profiled_params": params}
    except Exception as e:
        print(f"WARNING: thop profiling failed: {e}")
        print("  FLOPs count will be omitted; parameter count is still valid.")
        return {"macs": None, "flops": None, "profiled_params": None}


def format_number(n: float | None) -> str:
    if n is None:
        return "N/A"
    if n >= 1e9:
        return f"{n / 1e9:.3f} G"
    if n >= 1e6:
        return f"{n / 1e6:.3f} M"
    if n >= 1e3:
        return f"{n / 1e3:.3f} K"
    return f"{n:.0f}"


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import config as v2_config
    from model.medicalsignformer import MedicalSignFormerV2

    model = MedicalSignFormerV2(
        embed_dim=v2_config.EMBED_DIM,
        num_classes=v2_config.NUM_CLASSES,
        sequence_length=v2_config.SEQUENCE_LENGTH,
    )
    dummy_input = torch.randn(1, v2_config.SEQUENCE_LENGTH, v2_config.INPUT_DIM)

    params = count_parameters(model)
    flops_result = profile_model(model, dummy_input)
    size_mb = estimate_model_size_mb(params["total"])

    print("=" * 60)
    print("MEDICALSIGNFORMERV2 — COMPUTATIONAL PROFILE")
    print("=" * 60)
    print(f"  Total parameters      : {format_number(params['total'])}")
    print(f"  Trainable parameters  : {format_number(params['trainable'])}")
    print(f"  Non-trainable params  : {format_number(params['non_trainable'])}")
    print(f"  Estimated size (fp32) : {size_mb:.2f} MB")
    print(f"  FLOPs per forward pass: {format_number(flops_result['flops'])}")
    print(f"  MACs per forward pass : {format_number(flops_result['macs'])}")

    print(
        "\nNOTE: thop traces standard ops (Linear, Conv1d, MultiheadAttention) "
        "but NOT custom tensor-math ops (manual GAT attention, sequential "
        "Mamba fallback). FLOPs are a lower bound for those components."
    )


if __name__ == "__main__":
    main()
