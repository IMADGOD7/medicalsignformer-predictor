from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from model.medicalsignformer import MedicalSignFormerV2
from dataset.dataloader import get_dataloaders

ROOT = Path(__file__).parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_finetuned_model.pth"
EVAL_DIR = ROOT / "evaluation"


def load_label_names(label_map_path: Path, num_classes: int) -> list[str]:
    if not label_map_path.exists():
        return [str(i) for i in range(num_classes)]
    with label_map_path.open("r", encoding="utf-8") as f:
        name_to_idx = json.load(f)
    idx_to_name = [None] * num_classes
    for name, idx in name_to_idx.items():
        if 0 <= idx < num_classes:
            idx_to_name[idx] = name
    return [name if name is not None else str(i) for i, name in enumerate(idx_to_name)]


def collect_per_class_attention(
    model: torch.nn.Module, test_loader, device: torch.device, num_classes: int, seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()

    attention_sums = np.zeros((num_classes, seq_len), dtype=np.float64)
    sample_counts = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for features, labels, lengths in test_loader:
            features = features.to(device)
            lengths_dev = lengths.to(device)

            _logits, attention_weights = model(features, lengths=lengths_dev)  # (B, T, 1)
            attention_weights = attention_weights.squeeze(-1).cpu().numpy()  # (B, T)

            for i in range(features.size(0)):
                label = labels[i].item()
                real_len = lengths[i].item()

                attention_sums[label, :real_len] += attention_weights[i, :real_len]
                sample_counts[label] += 1

    return attention_sums, sample_counts


def plot_heatmap(mean_attention: np.ndarray, label_names: list[str], output_path: Path) -> None:
    num_classes, seq_len = mean_attention.shape

    fig_height = max(10, num_classes * 0.3)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    im = ax.imshow(mean_attention, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(label_names, fontsize=6)
    ax.set_xlabel("Frame position (0 = start of sign)")
    ax.set_ylabel("True class")
    ax.set_title("Mean Temporal Attention Weight by Class (test set)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Mean attention weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Attention heat map saved to: {output_path}")


def print_confusable_pair_comparison(
    mean_attention: np.ndarray, label_names: list[str], class_a: str, class_b: str,
) -> None:

    if class_a not in label_names or class_b not in label_names:
        print(f"\n(Skipping {class_a}/{class_b} comparison - one or both class names not found in label_map.)")
        return

    idx_a = label_names.index(class_a)
    idx_b = label_names.index(class_b)
    curve_a = mean_attention[idx_a]
    curve_b = mean_attention[idx_b]

    correlation = np.corrcoef(curve_a, curve_b)[0, 1]
    peak_a = int(curve_a.argmax())
    peak_b = int(curve_b.argmax())

    print(f"\n--- {class_a} vs {class_b} attention pattern comparison ---")
    print(f"Peak attention frame - {class_a}: {peak_a}, {class_b}: {peak_b}")
    print(f"Pearson correlation between the two mean attention curves: {correlation:.4f}")
    print(
        "(A high correlation would suggest the model attends to similarly-timed "
        "moments for both classes - one possible contributor to the confusion "
        "between them, alongside the landmark motion itself being visually similar.)"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: {CHECKPOINT_PATH} not found - run train_finetune.py first.")
        return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    model = MedicalSignFormerV2(
        embed_dim=config.EMBED_DIM, num_classes=config.NUM_CLASSES, sequence_length=config.SEQUENCE_LENGTH,
    ).to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    print("Loaded fine-tuned model.")

    dataloaders = get_dataloaders(ROOT / "data" / "processed")
    test_loader = dataloaders.get("test")
    if test_loader is None:
        print("ERROR: test.csv not found.")
        return

    label_names = load_label_names(ROOT / "data" / "processed" / "label_map.json", config.NUM_CLASSES)

    print(f"Collecting per-class attention over {len(test_loader.dataset)} test samples...")
    attention_sums, sample_counts = collect_per_class_attention(
        model, test_loader, device, config.NUM_CLASSES, config.SEQUENCE_LENGTH,
    )

    empty_classes = [label_names[i] for i in range(config.NUM_CLASSES) if sample_counts[i] == 0]
    if empty_classes:
        print(f"\nWARNING: {len(empty_classes)} class(es) have NO test samples, will show as all-zero rows: {empty_classes}")

    safe_counts = np.maximum(sample_counts, 1)[:, None]  # avoid divide-by-zero for empty classes
    mean_attention = attention_sums / safe_counts

    plot_heatmap(mean_attention, label_names, EVAL_DIR / "attention_heatmap.png")

    np.save(EVAL_DIR / "mean_attention_per_class.npy", mean_attention)
    print(f"Raw per-class mean attention array saved to: {EVAL_DIR / 'mean_attention_per_class.npy'}")


    print_confusable_pair_comparison(mean_attention, label_names, "TB", "DIARRHOEA")


if __name__ == "__main__":
    main()