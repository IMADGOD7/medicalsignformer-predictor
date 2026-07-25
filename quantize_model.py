from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import config
from model.medicalsignformer import MedicalSignFormerV2
from dataset.dataloader import get_dataloaders

ROOT = Path(__file__).parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_finetuned_model.pth"
QUANTIZED_CHECKPOINT_PATH = ROOT / "checkpoints" / "best_finetuned_model_quantized.pth"

WARMUP_RUNS = 10
TIMED_RUNS = 50


def get_model_size_mb(model: nn.Module) -> float:
    temp_path = ROOT / "_temp_size_check.pth"
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 ** 2)
    os.remove(temp_path)
    return size_mb


def quantize_excluding_mamba(model_fp32: nn.Module) -> nn.Module:
    import copy
    model_for_quantization = copy.deepcopy(model_fp32)

    excluded_count = 0
    for name, module in model_for_quantization.named_modules():
        if isinstance(module, nn.Linear) and name.startswith("mamba."):
            module.qconfig = None  # type: ignore[assignment]
            excluded_count += 1

    total_linear = sum(1 for m in model_for_quantization.modules() if isinstance(m, nn.Linear))
    print(
        f"Selective quantization: {total_linear - excluded_count} Linear layers quantized, "
        f"{excluded_count} Mamba-internal Linear layers excluded (qconfig=None, left in fp32)."
    )

    return torch.quantization.quantize_dynamic(
        model_for_quantization, {nn.Linear}, dtype=torch.qint8,
    )


def time_cpu_inference(model: nn.Module, dummy_input: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(dummy_input)

        timings = []
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            timings.append(end - start)

    timings = np.array(timings)
    return {"mean_ms": float(timings.mean() * 1000), "std_ms": float(timings.std() * 1000)}


def evaluate_accuracy(model: nn.Module, test_loader, device: torch.device) -> dict:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels, lengths in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits, _attention_weights = model(features, lengths=lengths)
            predictions = torch.argmax(logits, dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return {"accuracy": correct / total, "total_samples": total}


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: {CHECKPOINT_PATH} not found - run train_finetune.py first.")
        return

    device = torch.device("cpu")  # dynamic quantization's int8 kernels are CPU-only

    print("Loading trained model...")
    model_fp32 = MedicalSignFormerV2(
        embed_dim=config.EMBED_DIM, num_classes=config.NUM_CLASSES, sequence_length=config.SEQUENCE_LENGTH,
    )
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model_fp32.load_state_dict(state_dict, strict=True)
    model_fp32.to(device)

    print("Applying dynamic quantization (nn.Linear layers -> int8)...")
    model_int8_full = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8,
    )
    torch.save(model_int8_full.state_dict(), QUANTIZED_CHECKPOINT_PATH)
    print(f"Fully quantized model saved to: {QUANTIZED_CHECKPOINT_PATH}")

    print("\nApplying SELECTIVE dynamic quantization (excluding Mamba's Linear layers)...")
    model_int8_selective = quantize_excluding_mamba(model_fp32)
    selective_checkpoint_path = ROOT / "checkpoints" / "best_finetuned_model_quantized_selective.pth"
    torch.save(model_int8_selective.state_dict(), selective_checkpoint_path)
    print(f"Selectively quantized model saved to: {selective_checkpoint_path}")

    configs = {
        "fp32 (original)": model_fp32,
        "int8 (full quantization)": model_int8_full,
        "int8 (selective, Mamba excluded)": model_int8_selective,
    }

    print("\n" + "=" * 70)
    print("SIZE COMPARISON")
    print("=" * 70)
    sizes = {}
    for name, model in configs.items():
        sizes[name] = get_model_size_mb(model)
        print(f"{name:<40}{sizes[name]:>10.2f} MB")
    print(f"\nFull quantization size reduction     : {(1 - sizes['int8 (full quantization)'] / sizes['fp32 (original)']) * 100:.1f}%")
    print(f"Selective quantization size reduction: {(1 - sizes['int8 (selective, Mamba excluded)'] / sizes['fp32 (original)']) * 100:.1f}%")

    print("\n" + "=" * 70)
    print(f"INFERENCE LATENCY COMPARISON (CPU, {TIMED_RUNS} timed passes)")
    print("=" * 70)
    dummy_input = torch.randn(1, config.SEQUENCE_LENGTH, config.INPUT_DIM)
    timings = {}
    for name, model in configs.items():
        timings[name] = time_cpu_inference(model, dummy_input)
        print(f"{name:<40}{timings[name]['mean_ms']:>8.2f} ms  (± {timings[name]['std_ms']:.2f} ms)")

    print("\n" + "=" * 70)
    print("ACCURACY COMPARISON (test set)")
    print("=" * 70)
    dataloaders = get_dataloaders(ROOT / "data" / "processed")
    test_loader = dataloaders.get("test")
    if test_loader is None:
        print("test.csv not found - skipping accuracy comparison.")
    else:
        accuracies = {}
        for name, model in configs.items():
            result = evaluate_accuracy(model, test_loader, device)
            accuracies[name] = result["accuracy"]
            print(f"{name:<40}{result['accuracy']:>8.4f}  ({result['total_samples']} samples)")

        baseline = accuracies["fp32 (original)"]
        print(f"\nFull quantization accuracy change     : {accuracies['int8 (full quantization)'] - baseline:+.4f}")
        print(f"Selective quantization accuracy change: {accuracies['int8 (selective, Mamba excluded)'] - baseline:+.4f}")

    print(
        f"\nNOTE: dynamic quantization only converts nn.Linear layers to "
        f"int8 in the first place. The 'full quantization' row still "
        f"leaves nn.Conv1d (Mamba's causal conv) and nn.MultiheadAttention "
        f"(Face Encoder) in fp32 regardless, same as before - only the "
        f"SELECTIVE row makes an additional, deliberate exclusion (Mamba's "
        f"4 Linear layers: in_proj, x_proj, dt_proj, out_proj) on top of that."
    )


if __name__ == "__main__":
    main()