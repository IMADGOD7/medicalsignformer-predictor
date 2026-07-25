from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

V1_ROOT = Path("C:\\Users\\manab\\OneDrive\\Desktop\\internproject")  # adjust to your actual v1 project path
V2_ROOT = Path(__file__).parent

WARMUP_RUNS = 10
TIMED_RUNS = 100
INFERENCE_TIME_TARGET_SECONDS = 15.0


def time_forward_passes(
    model: torch.nn.Module, dummy_input: torch.Tensor, device: torch.device, model_name: str,
) -> dict:
    model = model.to(device)
    model.eval()
    dummy_input = dummy_input.to(device)

    is_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(dummy_input)
        if is_cuda:
            torch.cuda.synchronize()

        timings = []
        for _ in range(TIMED_RUNS):
            if is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if is_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

    timings = np.array(timings)
    return {
        "device": str(device),
        "mean_seconds": float(timings.mean()),
        "std_seconds": float(timings.std()),
        "min_seconds": float(timings.min()),
        "max_seconds": float(timings.max()),
        "mean_ms": float(timings.mean() * 1000),
    }


def print_timing_result(model_name: str, result: dict) -> None:
    print(f"\n{model_name} on {result['device']}:")
    print(f"  Mean latency : {result['mean_ms']:.2f} ms  (± {result['std_seconds']*1000:.2f} ms std)")
    print(f"  Min / Max    : {result['min_seconds']*1000:.2f} ms / {result['max_seconds']*1000:.2f} ms")
    meets_target = result["mean_seconds"] < INFERENCE_TIME_TARGET_SECONDS
    print(
        f"  Meets <{INFERENCE_TIME_TARGET_SECONDS:.0f}s target: "
        f"{'YES' if meets_target else 'NO'} "
        f"({result['mean_seconds']:.6f}s per sample)"
    )


def main() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    else:
        print("CUDA not available in this environment - timing CPU only.\n")

    sys.path.insert(0, str(V1_ROOT))
    import config as v1_config
    from model.medicalsignformer import MedicalSignFormer
    v1_model = MedicalSignFormer(
        num_classes=v1_config.NUM_CLASSES, sequence_length=v1_config.SEQUENCE_LENGTH, embed_dim=v1_config.EMBED_DIM,
    )
    v1_dummy = torch.randn(1, v1_config.SEQUENCE_LENGTH, v1_config.INPUT_DIM)

    for mod_name in list(sys.modules):
        if mod_name in ("config",) or mod_name.startswith("models") or mod_name.startswith("model"):
            del sys.modules[mod_name]
    sys.path.remove(str(V1_ROOT))
    sys.path.insert(0, str(V2_ROOT))

    import config as v2_config
    from model.medicalsignformer import MedicalSignFormerV2
    v2_model = MedicalSignFormerV2(
        embed_dim=v2_config.EMBED_DIM, num_classes=v2_config.NUM_CLASSES, sequence_length=v2_config.SEQUENCE_LENGTH,
    )
    v2_dummy = torch.randn(1, v2_config.SEQUENCE_LENGTH, v2_config.INPUT_DIM)

    print("=" * 70)
    print(f"INFERENCE TIMING ({WARMUP_RUNS} warmup + {TIMED_RUNS} timed single-sample passes each)")
    print("=" * 70)

    results = {}
    for device in devices:
        results[("v1", device.type)] = time_forward_passes(v1_model, v1_dummy, device, "v1")
        results[("v2", device.type)] = time_forward_passes(v2_model, v2_dummy, device, "v2")

    for device in devices:
        print(f"\n--- {device.type.upper()} ---")
        print_timing_result("v1 (MedicalSignFormer)", results[("v1", device.type)])
        print_timing_result("v2 (MedicalSignFormerV2)", results[("v2", device.type)])

        v1_ms = results[("v1", device.type)]["mean_ms"]
        v2_ms = results[("v2", device.type)]["mean_ms"]
        ratio = v2_ms / v1_ms
        print(f"\n  v2 is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'} than v1 on {device.type.upper()}.")

    if torch.cuda.is_available():
        try:
            from model.mamba_encoder import USING_OFFICIAL_MAMBA
            print(
                f"\nNOTE: v2's Mamba encoder is using "
                f"{'the OFFICIAL CUDA-fused mamba_ssm package' if USING_OFFICIAL_MAMBA else 'the pure-PyTorch fallback (_SelectiveSSM)'}. "
                + ("" if USING_OFFICIAL_MAMBA else
                   "The fallback's sequential Python scan loop is meaningfully slower than the official "
                   "fused kernel would be - if mamba_ssm becomes installable on your setup, v2's GPU "
                   "timing above would likely improve.")
            )
        except ImportError:
            pass


if __name__ == "__main__":
    main()