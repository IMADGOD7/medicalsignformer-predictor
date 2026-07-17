from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

import config
from model.medicalsignformer import MedicalSignFormerV2
from model.monte_carlo import enable_mc_dropout, mc_dropout_predict
from dataset.dataloader import get_dataloaders

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints"
EVAL_DIR = ROOT / "evaluation"


def load_label_names(label_map_path: Path, num_classes: int) -> list[str]:
    """Index -> class name, falling back to numeric indices if the label
    map is unavailable (same pattern used by tools/visualize_attention.py
    in the v1 project)."""
    if not label_map_path.exists():
        return [str(i) for i in range(num_classes)]

    with label_map_path.open("r", encoding="utf-8") as f:
        name_to_idx = json.load(f)

    idx_to_name = [None] * num_classes
    for name, idx in name_to_idx.items():
        if 0 <= idx < num_classes:
            idx_to_name[idx] = name
    return [name if name is not None else str(i) for i, name in enumerate(idx_to_name)]


def run_deterministic_evaluation(
    model: torch.nn.Module, test_loader, device: torch.device, label_names: list[str]
) -> tuple[list[int], list[int]]:
    """Standard eval-mode inference over the full test set. Returns
    (all_labels, all_predictions) for downstream metric computation."""
    model.eval()

    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for features, labels, lengths in test_loader:
            features = features.to(device)
            lengths = lengths.to(device)

            logits, _attention_weights = model(features, lengths=lengths)
            predictions = torch.argmax(logits, dim=-1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return all_labels, all_predictions


def save_classification_report(
    all_labels: list[int], all_predictions: list[int], label_names: list[str], output_path: Path
) -> None:
    report_dict = classification_report(
        all_labels, all_predictions, labels=list(range(len(label_names))),
        target_names=label_names, output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_path)
    print(f"Classification report saved to: {output_path}")

    overall_accuracy = report_dict["accuracy"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    print(f"\nOverall accuracy : {overall_accuracy:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Weighted F1      : {weighted_f1:.4f}")


def save_confusion_matrix(
    all_labels: list[int], all_predictions: list[int], label_names: list[str], output_path: Path
) -> None:
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(len(label_names))))

    fig_size = max(10, len(label_names) * 0.3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90, fontsize=6)
    ax.set_yticklabels(label_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - MedicalSignFormerV2 (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {output_path}")


def run_mc_dropout_evaluation(
    model: torch.nn.Module, test_loader, device: torch.device, label_names: list[str], output_path: Path
) -> None:
    """Runs MC Dropout inference sample-by-batch over the test set and
    writes a per-sample CSV of prediction/confidence/entropy/variance/
    correctness, plus prints summary statistics."""
    enable_mc_dropout(model)

    rows = []
    with torch.no_grad():
        for features, labels, lengths in test_loader:
            features = features.to(device)
            lengths = lengths.to(device)

            result = mc_dropout_predict(model, features, lengths=lengths, num_samples=config.MC_DROPOUT_NUM_SAMPLES)

            predictions = result["prediction"].cpu().tolist()
            confidences = result["confidence"].cpu().tolist()
            entropies = result["entropy"].cpu().tolist()
            pred_variances = result["predicted_class_variance"].cpu().tolist()
            true_labels = labels.tolist()

            for true_label, pred, conf, ent, var in zip(true_labels, predictions, confidences, entropies, pred_variances):
                rows.append({
                    "true_label": label_names[true_label],
                    "predicted_label": label_names[pred],
                    "correct": true_label == pred,
                    "confidence": conf,
                    "entropy": ent,
                    "predicted_class_variance": var,
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nMC Dropout per-sample predictions saved to: {output_path}")

    print(f"\nMC Dropout summary ({config.MC_DROPOUT_NUM_SAMPLES} stochastic passes per sample):")
    print(f"  Mean confidence           : {df['confidence'].mean():.4f}")
    print(f"  Mean predictive entropy   : {df['entropy'].mean():.4f}")
    print(f"  Mean predicted-class var. : {df['predicted_class_variance'].mean():.6f}")

    wrong_and_confident = df[(~df["correct"]) & (df["confidence"] > df["confidence"].median())]
    print(
        f"  Wrong predictions with above-median confidence: {len(wrong_and_confident)} "
        f"/ {(~df['correct']).sum()} wrong total"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    dataloaders = get_dataloaders(ROOT / "data" / "processed")
    test_loader = dataloaders.get("test")
    if test_loader is None:
        raise RuntimeError("test_loader is required for evaluation - data/processed/test.csv not found.")
    print(f"Test samples: {len(test_loader.dataset)}")

    label_names = load_label_names(ROOT / "data" / "processed" / "label_map.json", config.NUM_CLASSES)

    model = MedicalSignFormerV2(
        embed_dim=config.EMBED_DIM, num_classes=config.NUM_CLASSES, sequence_length=config.SEQUENCE_LENGTH,
    ).to(device)

    checkpoint_path = CHECKPOINT_DIR / "best_finetuned_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found - run train_finetune.py first to produce a trained model."
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded fine-tuned model from {checkpoint_path}")

    print("\n--- Standard (deterministic) evaluation ---")
    all_labels, all_predictions = run_deterministic_evaluation(model, test_loader, device, label_names)
    save_classification_report(all_labels, all_predictions, label_names, EVAL_DIR / "classification_report.csv")
    save_confusion_matrix(all_labels, all_predictions, label_names, EVAL_DIR / "confusion_matrix.png")

    print("\n--- Monte Carlo Dropout evaluation ---")
    run_mc_dropout_evaluation(model, test_loader, device, label_names, EVAL_DIR / "mc_dropout_predictions.csv")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()