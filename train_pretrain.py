from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import config
from model.masked_pretraining import ContrastivePretraining
from dataset.dataloader import get_dataloaders

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints"
PRETRAIN_LOG_PATH = ROOT / "pretrain_log.csv"


def _build_padding_mask(lengths: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    lengths = lengths.to(device).unsqueeze(1)
    return position_ids >= lengths


def run_pretrain_epoch(
    model: ContrastivePretraining,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    scaler: GradScaler | None = None,
) -> float:
    model.train(mode=train)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for features, _labels, lengths in loader:
            features = features.to(device)
            batch_size, seq_len, _ = features.shape
            padding_mask = _build_padding_mask(lengths, seq_len, device)

            if train:
                optimizer.zero_grad()

            with autocast("cuda"):
                output = model(features, padding_mask=padding_mask)
                loss = output["loss"]

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def log_epoch_to_csv(log_path: Path, epoch: int, train_loss: float, val_loss: float, checkpoint_saved: bool) -> None:
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_contrastive_loss", "val_contrastive_loss", "checkpoint_saved"])
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", checkpoint_saved])


def save_pretrained_encoder(model: ContrastivePretraining, checkpoint_dir: Path = CHECKPOINT_DIR) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "pretrained_encoder.pth"
    torch.save(model.encoder.state_dict(), checkpoint_path)
    return checkpoint_path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders = get_dataloaders(ROOT / "data" / "processed")
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")
    if train_loader is None or val_loader is None:
        raise RuntimeError("Both train_loader and val_loader are required for pretraining.")

    model = ContrastivePretraining(
        embed_dim=config.EMBED_DIM,
        proj_dim=config.PRETRAIN_PROJ_DIM,
        temperature=config.PRETRAIN_TEMPERATURE,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.PRETRAIN_LEARNING_RATE,
        weight_decay=config.PRETRAIN_WEIGHT_DECAY,
    )
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.PRETRAIN_MAX_EPOCHS,
        eta_min=config.LR_SCHEDULER_MIN_LR,
    )

    print(f"\nStarting Stage 1 contrastive pretraining:")
    print(f"  Max Epochs: {config.PRETRAIN_MAX_EPOCHS} | LR: {config.PRETRAIN_LEARNING_RATE:.1e}")
    print(f"  Temperature: {config.PRETRAIN_TEMPERATURE} | Proj Dim: {config.PRETRAIN_PROJ_DIM}")
    print(f"  Augmentations: time_warp({config.PRETRAIN_TIME_WARP}), noise({config.PRETRAIN_NOISE_STD}), dropout({config.PRETRAIN_DROP_RATIO})")
    print(f"  Training Samples: {len(train_loader.dataset)} | Validation Samples: {len(val_loader.dataset)}\n")

    if PRETRAIN_LOG_PATH.exists():
        PRETRAIN_LOG_PATH.unlink()

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.PRETRAIN_MAX_EPOCHS + 1):
        train_loss = run_pretrain_epoch(model, train_loader, optimizer, device, train=True, scaler=scaler)
        val_loss = run_pretrain_epoch(model, val_loader, optimizer=None, device=device, train=False)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        improved = (best_val_loss - val_loss) > config.PRETRAIN_EARLY_STOPPING_MIN_DELTA
        checkpoint_saved = False
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint_path = save_pretrained_encoder(model)
            checkpoint_saved = True
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:3d}/{config.PRETRAIN_MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr:.2e}"
            + (f"  -> saved" if checkpoint_saved else
               f"  -> no improvement ({epochs_without_improvement}/{config.PRETRAIN_EARLY_STOPPING_PATIENCE})")
        )

        log_epoch_to_csv(PRETRAIN_LOG_PATH, epoch, train_loss, val_loss, checkpoint_saved)

        if epochs_without_improvement >= config.PRETRAIN_EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
            break
    else:
        print(f"\nReached max_epochs={config.PRETRAIN_MAX_EPOCHS}.")

    print(f"\nStage 1 pretraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Pretrained encoder saved to: {CHECKPOINT_DIR / 'pretrained_encoder.pth'}")


if __name__ == "__main__":
    main()
