from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import config
from model.medicalsignformer import MedicalSignFormerV2
from dataset.dataloader import get_dataloaders

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints"
PRETRAINED_ENCODER_PATH = CHECKPOINT_DIR / "pretrained_encoder.pth"
LOG_PATH = ROOT / "finetune_log.csv"


def mixup_batch(
    features: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = features.size(0)
    index = torch.randperm(batch_size, device=features.device)

    mixed_features = lam * features + (1.0 - lam) * features[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_features, labels_a, labels_b, lam


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, max_epochs: int,
                 min_lr: float = 1e-6, last_epoch: int = -1) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if epoch < self.warmup_epochs:
                lr = base_lr * (epoch + 1) / max(self.warmup_epochs, 1)
            else:
                progress = (epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine
            lrs.append(max(lr, self.min_lr))
        return lrs


def train_one_epoch(
    model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, device: torch.device,
    use_mixup: bool = True, mixup_alpha: float = 0.2,
    scaler: GradScaler | None = None,
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

        with autocast("cuda"):
            if use_mixup and model.training:
                mixed_features, labels_a, labels_b, lam = mixup_batch(
                    features, labels, alpha=mixup_alpha
                )
                logits, _ = model(mixed_features, lengths=lengths)
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            else:
                logits, _ = model(features, lengths=lengths)
                loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def validate_one_epoch(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
    device: torch.device, num_classes: int,
) -> Tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for features, labels, lengths in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits, _ = model(features, lengths=lengths)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    accuracy = correct / total
    macro_f1 = f1_score(
        all_labels, all_predictions,
        labels=list(range(num_classes)), average="macro", zero_division=0,
    )
    return total_loss / total, accuracy, macro_f1


def log_epoch_to_csv(
    log_path: Path, epoch: int, phase: str, train_loss: float, train_acc: float,
    val_loss: float, val_acc: float, val_macro_f1: float, gap: float,
    current_lr: float, checkpoint_saved: bool,
) -> None:
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "epoch", "phase", "train_loss", "train_acc", "val_loss",
                "val_acc", "val_macro_f1", "gap", "lr", "checkpoint_saved",
            ])
        writer.writerow([
            epoch, phase, f"{train_loss:.6f}", f"{train_acc:.6f}",
            f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_macro_f1:.6f}",
            f"{gap:+.6f}", f"{current_lr:.2e}", checkpoint_saved,
        ])


def save_best_model(model: nn.Module, checkpoint_dir: Path = CHECKPOINT_DIR) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def build_class_weights(
    train_csv_path: Path, num_classes: int, device: torch.device,
) -> torch.Tensor:
    train_df = pd.read_csv(train_csv_path)
    counts = train_df["label"].value_counts().reindex(range(num_classes), fill_value=0)
    counts = counts.clip(lower=1)
    weights = 1.0 / counts.values.astype(float)
    weights = weights / weights.mean()
    weights = np.clip(weights, a_min=None, a_max=5.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        with torch.no_grad():
            target_smooth = torch.full_like(logits, self.label_smoothing / max(num_classes - 1, 1))
            target_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        focal_factor = (1.0 - probs) ** self.gamma
        focal_log_probs = focal_factor * log_probs
        loss = -torch.sum(target_smooth * focal_log_probs, dim=-1)

        if self.weight is not None:
            loss = loss * self.weight[targets]

        return loss.mean()


def freeze_encoder(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_encoder(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders = get_dataloaders(ROOT / "data" / "processed")
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")
    if train_loader is None or val_loader is None:
        raise RuntimeError("Both train_loader and val_loader are required.")

    model_obj = MedicalSignFormerV2(
        embed_dim=config.EMBED_DIM, num_classes=config.NUM_CLASSES,
        sequence_length=config.SEQUENCE_LENGTH,
    ).to(device)

    if PRETRAINED_ENCODER_PATH.exists():
        model_obj.load_pretrained_encoder(PRETRAINED_ENCODER_PATH, strict=True)
        print(f"Loaded pretrained encoder from {PRETRAINED_ENCODER_PATH}")
    else:
        print(f"WARNING: {PRETRAINED_ENCODER_PATH} not found. Run train_pretrain.py first.")

    # Differential learning rates: encoder gets lower LR to preserve pretrained features
    encoder_params = []
    head_params = []

    for name, param in model_obj.named_parameters():
        if "classifier" in name or "mamba" in name or "temporal_pooling" in name:
            head_params.append(param)
        else:
            encoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": config.ENCODER_LR},
        {"params": head_params, "lr": config.FINETUNE_LEARNING_RATE},
    ], weight_decay=config.FINETUNE_WEIGHT_DECAY)
    scaler = GradScaler()

    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=config.FINETUNE_WARMUP_EPOCHS,
        max_epochs=config.FINETUNE_MAX_EPOCHS, min_lr=config.LR_SCHEDULER_MIN_LR,
    )

    class_weights = build_class_weights(
        ROOT / "data" / "processed" / "train.csv", config.NUM_CLASSES, device,
    )
    criterion = FocalLoss(
        weight=class_weights, gamma=config.FOCAL_GAMMA,
        label_smoothing=config.FINETUNE_LABEL_SMOOTHING,
    )

    print(f"\nModel: {sum(p.numel() for p in model_obj.parameters() if p.requires_grad):,} params")
    print(f"Classes: {config.NUM_CLASSES} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    print(f"\nStage 2 fine-tuning:")
    print(f"  Epochs: {config.FINETUNE_MAX_EPOCHS} | Warmup: {config.FINETUNE_WARMUP_EPOCHS}")
    print(f"  Encoder LR: {config.ENCODER_LR:.1e} | Head LR: {config.FINETUNE_LEARNING_RATE:.1e}")
    print(f"  Freeze epochs: {config.ENCODER_FREEZE_EPOCHS} | MixUp: {config.USE_MIXUP}")
    print(f"  Label smoothing: {config.FINETUNE_LABEL_SMOOTHING} | Early stop patience: {config.FINETUNE_EARLY_STOPPING_PATIENCE}")
    print()

    if LOG_PATH.exists():
        LOG_PATH.unlink()

    best_macro_f1 = 0.0
    epochs_without_improvement = 0

    if config.ENCODER_FREEZE_EPOCHS > 0:
        freeze_encoder(model_obj)
        print(f"[Phase 1] Freezing encoder for {config.ENCODER_FREEZE_EPOCHS} epochs\n")

    for epoch in range(1, config.FINETUNE_MAX_EPOCHS + 1):
        if epoch == config.ENCODER_FREEZE_EPOCHS + 1 and config.ENCODER_FREEZE_EPOCHS > 0:
            unfreeze_encoder(model_obj)
            # Unfreeze needs smaller batch — encoder gradients nearly double VRAM usage
            if train_loader.batch_size != config.BATCH_SIZE_UNFREEZE:
                train_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=config.BATCH_SIZE_UNFREEZE,
                    shuffle=True,
                    num_workers=train_loader.num_workers,
                    collate_fn=train_loader.collate_fn,
                    pin_memory=train_loader.pin_memory,
                    persistent_workers=False,
                )
                print(f"  [Phase 2] Batch size reduced {config.BATCH_SIZE} → {config.BATCH_SIZE_UNFREEZE} for unfrozen encoder\n")
            # Rebuild optimizer with correct param groups after unfreezing
            encoder_params = []
            head_params = []
            for name, param in model_obj.named_parameters():
                if "classifier" in name or "mamba" in name or "temporal_pooling" in name:
                    head_params.append(param)
                else:
                    encoder_params.append(param)
            optimizer = torch.optim.AdamW([
                {"params": encoder_params, "lr": config.ENCODER_LR},
                {"params": head_params, "lr": config.FINETUNE_LEARNING_RATE},
            ], weight_decay=config.FINETUNE_WEIGHT_DECAY)
            scaler = GradScaler()
            scheduler = CosineWarmupScheduler(
                optimizer, warmup_epochs=0,
                max_epochs=config.FINETUNE_MAX_EPOCHS - config.ENCODER_FREEZE_EPOCHS,
                min_lr=config.LR_SCHEDULER_MIN_LR,
            )
            print(f"[Phase 2] Unfrozen encoder at epoch {epoch}. LR: encoder={config.ENCODER_LR:.1e}, head={config.FINETUNE_LEARNING_RATE:.1e}\n")

        phase = "frozen" if epoch <= config.ENCODER_FREEZE_EPOCHS else "full"
        current_lr = optimizer.param_groups[-1]["lr"]

        train_loss, train_acc = train_one_epoch(
            model_obj, train_loader, criterion, optimizer, device,
            use_mixup=(config.USE_MIXUP and phase == "full" and epoch >= config.MIXUP_START_EPOCH),
            mixup_alpha=config.MIXUP_ALPHA,
            scaler=scaler,
        )
        val_loss, val_acc, val_macro_f1 = validate_one_epoch(
            model_obj, val_loader, criterion, device, config.NUM_CLASSES,
        )
        scheduler.step()

        gap = val_loss - train_loss
        print(
            f"Epoch {epoch:3d}/{config.FINETUNE_MAX_EPOCHS} [{phase:6s}] | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_macro_f1:.4f} | "
            f"gap={gap:+.4f} lr={current_lr:.1e}"
        )

        improved = (val_macro_f1 - best_macro_f1) > config.FINETUNE_EARLY_STOPPING_MIN_DELTA
        if improved:
            best_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            path = save_best_model(model_obj)
            print(f"  -> improved to {best_macro_f1:.4f}. Saved -> {path}")
        else:
            epochs_without_improvement += 1
            print(f"  -> no improvement for {epochs_without_improvement}/{config.FINETUNE_EARLY_STOPPING_PATIENCE}")

        log_epoch_to_csv(
            LOG_PATH, epoch, phase, train_loss, train_acc,
            val_loss, val_acc, val_macro_f1, gap, current_lr, checkpoint_saved=improved,
        )

        if epochs_without_improvement >= config.FINETUNE_EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best val_macro_f1: {best_macro_f1:.4f}")
            break
    else:
        print(f"\nReached max_epochs={config.FINETUNE_MAX_EPOCHS}. Best val_macro_f1: {best_macro_f1:.4f}")

    print(f"\nStage 2 complete. Best model: {CHECKPOINT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()
