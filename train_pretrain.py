
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from model.masked_pretraining import MaskedTemporalPretraining
from dataset.dataloader import get_dataloaders

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints"
PRETRAIN_LOG_PATH = ROOT / "pretrain_log.csv"


def _build_padding_mask(lengths: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    """Boolean (B, T) mask, True = padded position, from real frame counts."""
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    lengths = lengths.to(device).unsqueeze(1)
    return position_ids >= lengths


def run_pretrain_epoch(
    model: MaskedTemporalPretraining,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> float:
    """One epoch of Stage 1. If `optimizer` is None, runs in eval mode
    (used for the validation pass) without any gradient updates."""
    model.train(mode=train)

    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for features, _labels, lengths in loader:
            # _labels intentionally unused - pretraining is label-independent.
            features = features.to(device)
            batch_size, seq_len, _ = features.shape
            padding_mask = _build_padding_mask(lengths, seq_len, device)

            if train:
                optimizer.zero_grad()

            output = model(features, padding_mask=padding_mask)
            loss = output["loss"]

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def log_epoch_to_csv(
    log_path: Path, epoch: int, train_loss: float, val_loss: float, checkpoint_saved: bool
) -> None:
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_masked_mse", "val_masked_mse", "checkpoint_saved"])
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", checkpoint_saved])


def save_pretrained_encoder(model: MaskedTemporalPretraining, checkpoint_dir: Path = CHECKPOINT_DIR) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "pretrained_encoder.pth"
    # Save ONLY the encoder - the reconstruction head is pretraining-only
    # scaffolding, not reused at fine-tuning time.
    torch.save(model.encoder.state_dict(), checkpoint_path)
    return checkpoint_path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders = get_dataloaders(ROOT / "data" / "processed")
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")
    print(
        f"Loaders ready -> "
        f"train: {'yes' if train_loader else 'missing'}, "
        f"val: {'yes' if val_loader else 'missing'}"
    )
    if train_loader is None or val_loader is None:
        raise RuntimeError("Both train_loader and val_loader are required for pretraining.")

    model = MaskedTemporalPretraining(
        embed_dim=config.EMBED_DIM,
        mask_ratio=config.PRETRAIN_MASK_RATIO,
        min_span_len=config.PRETRAIN_MIN_SPAN_LEN,
        max_span_len=config.PRETRAIN_MAX_SPAN_LEN,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.PRETRAIN_LEARNING_RATE,
        weight_decay=config.PRETRAIN_WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.PRETRAIN_MAX_EPOCHS,
    eta_min=1e-6,
    )

    print(
        f"\nStarting Stage 1 pretraining:"
        f"\n  Max Epochs        : {config.PRETRAIN_MAX_EPOCHS}"
        f"\n  Mask Ratio         : {config.PRETRAIN_MASK_RATIO}"
        f"\n  Span Length        : [{config.PRETRAIN_MIN_SPAN_LEN}, {config.PRETRAIN_MAX_SPAN_LEN}]"
        f"\n  Training Samples   : {len(train_loader.dataset)}"
        f"\n  Validation Samples : {len(val_loader.dataset)}"
        f"\n"
    )
    print(f"Per-epoch metrics will also be logged to: {PRETRAIN_LOG_PATH}\n")

    best_val_loss = float("inf")

    for epoch in range(1, config.PRETRAIN_MAX_EPOCHS + 1):
        train_loss = run_pretrain_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_pretrain_epoch(model, val_loader, optimizer=None, device=device, train=False)

        improved = val_loss < best_val_loss
        checkpoint_saved = False
        if improved:
            best_val_loss = val_loss
            checkpoint_path = save_pretrained_encoder(model)
            checkpoint_saved = True

        scheduler.step()
        
        print(
            f"Epoch {epoch:3d}/{config.PRETRAIN_MAX_EPOCHS} | "
            f"train_masked_mse={train_loss:.4f} | val_masked_mse={val_loss:.4f}"
            + (f"  -> improved, saved to {checkpoint_path}" if checkpoint_saved else "")
        )

        log_epoch_to_csv(PRETRAIN_LOG_PATH, epoch, train_loss, val_loss, checkpoint_saved)

    print(f"\nStage 1 pretraining complete. Best val_masked_mse: {best_val_loss:.4f}")
    print(f"Pretrained encoder saved to: {CHECKPOINT_DIR / 'pretrained_encoder.pth'}")


if __name__ == "__main__":
    main()