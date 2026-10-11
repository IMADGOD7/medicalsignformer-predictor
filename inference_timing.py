from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent


def time_forward_passes(
    model: torch.nn.Module, dummy_input: torch.Tensor, device: torch.device,
) -> dict:
    model = model.to(device)
    model.eval()
    dummy_input = dummy_input.to(device)

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings = []
        for _ in range(TIMED_RUNS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

    timings = np.array(timings)
    return {
        "device": str(device),
        "mean_ms": float(timings.mean() * 1000),
        "std_ms": float(timings.std() * 1000),
        "min_ms": float(timings.min() * 1000),
        "max_ms": float(timings.max() * 1000),
    }


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

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print("=" * 60)
    print(f"INFERENCE TIMING — MedicalSignFormerV2")
    print(f"({WARMUP_RUNS} warmup + {TIMED_RUNS} timed passes)")
    print("=" * 60)

    for device in devices:
        result = time_forward_passes(model, dummy_input, device)
        print(f"\n  Device : {result['device']}")
        print(f"  Mean   : {result['mean_ms']:.2f} ms  (+/- {result['std_ms']:.2f} ms)")
        print(f"  Min    : {result['min_ms']:.2f} ms")
        print(f"  Max    : {result['max_ms']:.2f} ms")
        meets = "YES" if result["mean_ms"] < INFERENCE_TARGET_MS else "NO"
        print(f"  < {INFERENCE_TARGET_MS}ms target: {meets}")

    if torch.cuda.is_available():
        try:
            from model.mamba_encoder import USING_OFFICIAL_MAMBA
            note = (
                "Mamba: OFFICIAL CUDA-fused mamba_ssm package."
                if USING_OFFICIAL_MAMBA
                else (
                    "Mamba: pure-PyTorch fallback (_SelectiveSSM) — sequential scan "
                    "loop is slower than the official fused kernel."
                )
            )
            print(f"\n  NOTE: {note}")
        except ImportError:
            pass


WARMUP_RUNS = 10
TIMED_RUNS = 100
INFERENCE_TARGET_MS = 50.0

if __name__ == "__main__":
    main()
