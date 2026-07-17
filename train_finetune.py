
from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import config
from model.medicalsignformer_v2 import MedicalSignFormerV2
from dataset.dataloader import get_dataloaders

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints"
PRETRAINED_ENCODER_PATH = CHECKPOINT_DIR / "pretrained_encoder.pth"
FINETUNE_LOG_PATH = ROOT / "finetune_log.csv"


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels, lengths in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits, _attention_weights = model(features, lengths=lengths)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        )

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels, lengths in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits, _attention_weights = model(features, lengths=lengths)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(
        all_labels, all_predictions, labels=list(range(num_classes)), average="macro", zero_division=0,
    )
    return average_loss, accuracy, macro_f1


def log_epoch_to_csv(
    log_path: Path, epoch: int, train_loss: float, train_acc: float,
    val_loss: float, val_acc: float, val_macro_f1: float, gap: float, checkpoint_saved: bool,
) -> None:
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                "val_macro_f1", "gap", "checkpoint_saved",
            ])
        writer.writerow([
            epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}",
            f"{val_acc:.6f}", f"{val_macro_f1:.6f}", f"{gap:+.6f}", checkpoint_saved,
        ])


def save_best_model(model: nn.Module, checkpoint_dir: Path = CHECKPOINT_DIR) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_finetuned_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def build_class_weights(train_csv_path: Path, num_classes: int, device: torch.device) -> torch.Tensor:
    """1/sqrt(class_count), normalized to mean 1 - identical formula to the
    original v1 train.py, for direct comparability."""
    train_df = pd.read_csv(train_csv_path)
    counts = train_df["label"].value_counts().reindex(range(num_classes), fill_value=0)
    # Avoid divide-by-zero for any class absent from this particular split.
    counts = counts.clip(lower=1)
    weights = 1.0 / counts.values.astype(float) ** 0.5
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
    num_classes: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
) -> None:
    best_val_loss = float("inf")
    best_macro_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_macro_f1 = validate_one_epoch(model, val_loader, criterion, device, num_classes)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        train_val_gap = train_loss - val_loss
        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f} | "
            f"gap={train_val_gap:+.4f}"
        )

        scheduler.step(val_macro_f1)

        improved = (val_macro_f1 - best_macro_f1) > min_delta
        checkpoint_path = None
        if improved:
            best_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            checkpoint_path = save_best_model(model)
            print(f"  -> val_macro_f1 improved. Saved checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  -> no improvement for {epochs_without_improvement}/{patience} epoch(s)")

        log_epoch_to_csv(
            FINETUNE_LOG_PATH, epoch, train_loss, train_acc, val_loss, val_acc,
            val_macro_f1, train_val_gap, checkpoint_saved=improved,
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after epoch {epoch}. Best val_macro_f1: {best_macro_f1:.4f}")
            break
    else:
        print(f"\nReached max_epochs={max_epochs} without early stopping. Best val_macro_f1: {best_macro_f1:.4f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders = get_dataloaders(ROOT / "data" / "processed")
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")
    print(f"Loaders ready -> train: {'yes' if train_loader else 'missing'}, val: {'yes' if val_loader else 'missing'}")
    if train_loader is None or val_loader is None:
        raise RuntimeError("Both train_loader and val_loader are required for fine-tuning.")

    model = MedicalSignFormerV2(
        embed_dim=config.EMBED_DIM,
        num_classes=config.NUM_CLASSES,
        sequence_length=config.SEQUENCE_LENGTH,
    ).to(device)
    print(f"Model initialized with num_classes={config.NUM_CLASSES}")

    if PRETRAINED_ENCODER_PATH.exists():
        model.load_pretrained_encoder(PRETRAINED_ENCODER_PATH, strict=True)
        print(f"Loaded pretrained encoder from {PRETRAINED_ENCODER_PATH}")
    else:
        print(
            f"WARNING: {PRETRAINED_ENCODER_PATH} not found - proceeding with a "
            f"randomly-initialized encoder. Run train_pretrain.py first for the "
            f"intended two-stage training pipeline."
        )

    class_weights = build_class_weights(ROOT / "data" / "processed" / "train.csv", config.NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.FINETUNE_LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.FINETUNE_LEARNING_RATE, weight_decay=config.FINETUNE_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE,
        threshold=config.LR_SCHEDULER_THRESHOLD, threshold_mode="abs",
        cooldown=config.LR_SCHEDULER_COOLDOWN, min_lr=config.LR_SCHEDULER_MIN_LR,
    )

    print(
        f"\nStarting Stage 2 fine-tuning:"
        f"\n  Max Epochs         : {config.FINETUNE_MAX_EPOCHS}"
        f"\n  Early Stopping     : {config.FINETUNE_EARLY_STOPPING_PATIENCE}"
        f"\n  Training Samples   : {len(train_loader.dataset)}"
        f"\n  Validation Samples : {len(val_loader.dataset)}"
        f"\n  Number of Classes  : {config.NUM_CLASSES}"
        f"\n"
    )
    print(f"Per-epoch metrics will also be logged to: {FINETUNE_LOG_PATH}\n")

    run_training_loop(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device,
        num_classes=config.NUM_CLASSES, max_epochs=config.FINETUNE_MAX_EPOCHS,
        patience=config.FINETUNE_EARLY_STOPPING_PATIENCE, min_delta=config.FINETUNE_EARLY_STOPPING_MIN_DELTA,
    )

    print("\nStage 2 fine-tuning complete.")


if __name__ == "__main__":
    main()