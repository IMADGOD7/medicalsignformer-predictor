
from __future__ import annotations

import sys
from pathlib import Path

import torch
from thop import profile

V1_ROOT = Path("C:\\Users\\manab\\OneDrive\\Desktop\\internproject")  # adjust to your actual v1 project path
V2_ROOT = Path(__file__).parent


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {"total": total, "trainable": trainable, "non_trainable": non_trainable}


def estimate_model_size_mb(param_count: int, bytes_per_param: int = 4) -> float:

    return (param_count * bytes_per_param) / (1024 ** 2)


def profile_model(model: torch.nn.Module, dummy_input: tuple, model_name: str) -> dict:
    model.eval()
    try:
        macs, params = profile(model, inputs=dummy_input, verbose=False)

        flops = macs * 2
        return {"macs": macs, "flops": flops, "profiled_params": params}
    except Exception as e:
        print(f"WARNING: thop profiling failed for {model_name}: {e}")
        print(f"  (FLOPs count will be omitted for {model_name}; parameter count above is still valid.)")
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
    sys.path.insert(0, str(V1_ROOT))

    print("Loading v1 (MedicalSignFormer)...")
    try:
        import config as v1_config  # v1's config.py
        from model.medicalsignformer import MedicalSignFormer
        v1_model = MedicalSignFormer(
            num_classes=v1_config.NUM_CLASSES,
            sequence_length=v1_config.SEQUENCE_LENGTH,
            embed_dim=v1_config.EMBED_DIM,
        )
        v1_seq_len = v1_config.SEQUENCE_LENGTH
        v1_input_dim = v1_config.INPUT_DIM
    except Exception as e:
        print(f"ERROR loading v1: {e}")
        print(f"Check V1_ROOT at the top of this file points to your actual v1 project directory.")
        return

    for mod_name in list(sys.modules):
        if mod_name in ("config",) or mod_name.startswith("models") or mod_name.startswith("model"):
            del sys.modules[mod_name]
    sys.path.remove(str(V1_ROOT))
    sys.path.insert(0, str(V2_ROOT))

    print("Loading v2 (MedicalSignFormerV2)...")
    try:
        import config as v2_config  # v2's config.py
        from model.medicalsignformer import MedicalSignFormerV2
        v2_model = MedicalSignFormerV2(
            embed_dim=v2_config.EMBED_DIM,
            num_classes=v2_config.NUM_CLASSES,
            sequence_length=v2_config.SEQUENCE_LENGTH,
        )
        v2_seq_len = v2_config.SEQUENCE_LENGTH
        v2_input_dim = v2_config.INPUT_DIM
    except Exception as e:
        print(f"ERROR loading v2: {e}")
        return

    v1_dummy = (torch.randn(1, v1_seq_len, v1_input_dim),)
    v2_dummy = (torch.randn(1, v2_seq_len, v2_input_dim),)

    v1_params = count_parameters(v1_model)
    v2_params = count_parameters(v2_model)

    v1_flops_result = profile_model(v1_model, v1_dummy, "v1")
    v2_flops_result = profile_model(v2_model, v2_dummy, "v2")

    v1_size_mb = estimate_model_size_mb(v1_params["total"])
    v2_size_mb = estimate_model_size_mb(v2_params["total"])

    print("\n" + "=" * 70)
    print("COMPUTATIONAL COMPARISON: v1 (MedicalSignFormer) vs v2 (MedicalSignFormer)")
    print("=" * 70)

    print(f"\n{'Metric':<30}{'v1':<20}{'v2':<20}")
    print("-" * 70)
    print(f"{'Total parameters':<30}{format_number(v1_params['total']):<20}{format_number(v2_params['total']):<20}")
    print(f"{'Trainable parameters':<30}{format_number(v1_params['trainable']):<20}{format_number(v2_params['trainable']):<20}")
    print(f"{'Estimated size (MB, fp32)':<30}{v1_size_mb:<20.2f}{v2_size_mb:<20.2f}")
    print(f"{'FLOPs (1 forward pass)':<30}{format_number(v1_flops_result['flops']):<20}{format_number(v2_flops_result['flops']):<20}")
    print(f"{'MACs (1 forward pass)':<30}{format_number(v1_flops_result['macs']):<20}{format_number(v2_flops_result['macs']):<20}")

    param_ratio = v2_params["total"] / v1_params["total"]
    print(f"\nv2 has {param_ratio:.2f}x the parameters of v1 "
          f"({'more' if param_ratio > 1 else 'fewer'} parameters).")

    if v1_flops_result["flops"] and v2_flops_result["flops"]:
        flops_ratio = v2_flops_result["flops"] / v1_flops_result["flops"]
        print(f"v2 requires {flops_ratio:.2f}x the FLOPs of v1 per forward pass "
              f"({'more' if flops_ratio > 1 else 'fewer'} computation).")

    print(
        "\nNOTE: thop's FLOP counter traces standard nn.Module operations "
        "(Linear, Conv1d, MultiheadAttention, etc.) via hooks. Custom ops "
        "implemented as raw tensor math (e.g. this project's manual GAT "
        "attention computation in graph_attention.py, or the sequential "
        "scan in the pure-PyTorch Mamba fallback) are NOT automatically "
        "traced by thop and may be undercounted in v2's FLOPs figure above "
        "- treat the FLOPs number as a lower bound for v2, not an exact "
        "count. Parameter counts above are exact regardless, since they "
        "don't depend on which ops thop recognizes."
    )


if __name__ == "__main__":
    main()