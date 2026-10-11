"""run_kaggle_cross_val.py
========================
Cross-dataset evaluation for MSL-Medical (Kaggle signers).

Reads MSL root CSV files (msl_medical_train_kaggle_fixed.csv,
msl_medical_test_kaggle_fixed.csv) directly, translates video paths
to .npy landmark paths and MSL label IDs to model label IDs, then
runs evaluation — no intermediate CSV generation needed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from model.medicalsignformer import MedicalSignFormerV2
from model.monte_carlo import enable_mc_dropout, mc_dropout_predict
from dataset.sign_dataset import MedicalSignDataset

# Paths
LANDMARKS_DIR  = ROOT / "landmarks_kaggle"
EVAL_DIR       = ROOT / "evaluation_kaggle"
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pth"
MSL_TRAIN_CSV  = ROOT / "msl_medical_train_kaggle_fixed.csv"
MSL_TEST_CSV   = ROOT / "msl_medical_test_kaggle_fixed.csv"

# MSL Medical label ID -> MedicalSignFormer label ID mapping
# (from msl_naming.py)
MSL_TO_MODEL_LABEL: dict[int, int] = {
    # Exact matches
    22: 0,    # Asthma -> ASTHMA
    23: 13,   # Cough -> Cough
    24: 43,   # Swollen -> Swell
    25: 29,   # Itchy -> Itch
    26: 46,   # Vomit -> Vomit
    27: 20,   # Flu -> FLU
    28: 22,   # Fever -> Fever
    29: 23,   # Headache -> HEAD ACHE
    32: 37,   # Sore Throat -> SORE THROAT
    # Approximate matches
    1:  36,   # Shoulder -> SHOULDER PAIN
    3:  41,   # Fingers -> Sore Finger
    5:  34,   # Skin -> Rash
    9:  45,   # Teeth -> Tooth ache
    15: 3,    # Bones -> BROKEN BONES
    18: 6,    # Blood -> Bleed
    19: 24,   # Heart -> HEART ATTACK
    20: 28,   # Pain -> Injury
    30: 39,   # Stomach -> STOMACH ACHE
    31: 5,    # Waist Pain -> Back Ache
}


def _load_msl_csv(csv_path: Path, landmarks_dir: Path) -> pd.DataFrame:
    """Load MSL root CSV (video paths + MSL label IDs) and translate.

    Input:  msl-medical/dataset_2/Ernest_34_c.mp4,33
    Output: filepath, signer, label (model label IDs)
    """
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["video_path", "msl_label"]

    # Extract signer from filename: msl-medical/dataset_2/Ernest_34_c.mp4 -> Ernest
    df["signer"] = df["video_path"].apply(
        lambda p: Path(p).stem.rsplit("_", 1)[0].rsplit("_", 1)[0]
    )

    # Translate .mp4 video path -> relative .npy path (MedicalSignDataset prepends root_dir)
    df["filepath"] = df["video_path"].apply(
        lambda p: f"landmarks_kaggle/{Path(p).stem}.npy"
    )

    # Map MSL label IDs -> model label IDs, drop unmapped classes
    df["label"] = df["msl_label"].map(MSL_TO_MODEL_LABEL)
    dropped = df["label"].isna().sum()
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    if dropped > 0:
        print(f"  Dropped {dropped} samples with unmapped MSL label IDs")

    df["disease"] = ""  # not needed for evaluation
    return df[["filepath", "disease", "signer", "label"]]


def collate_fn(batch):
    features, labels, lengths = zip(*batch)
    lengths = torch.stack(lengths)
    features_tensor = torch.stack(features)
    labels_tensor = torch.stack(labels)
    return features_tensor, labels_tensor, lengths


def evaluate(split_name: str, split_df: pd.DataFrame) -> None:
    """Run deterministic + MC-Dropout evaluation on an MSL split dataframe."""
    print("=" * 60)
    print(f"Evaluating: {split_name}")
    print("=" * 60)

    # Verify .npy files exist
    missing = [p for p in split_df["filepath"].head(5) if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Some NPY files missing. Example: {missing[0]}"
        )

    # Build label map for this split
    idx_to_disease = {int(row["label"]): f"class_{int(row['label'])}"
                      for _, row in split_df.iterrows()}
    num_classes = config.NUM_CLASSES
    label_names = [idx_to_disease.get(i, str(i)) for i in range(num_classes)]

    # Save split CSV for MedicalSignDataset
    csv_path = EVAL_DIR / f"{split_name}.csv"
    split_df.to_csv(csv_path, index=False)

    dataset = MedicalSignDataset(csv_file=csv_path, root_dir=ROOT, is_train=False)
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    n_classes = split_df["label"].nunique()
    n_signers = split_df["signer"].nunique()
    print(f"Loaded {len(dataset)} samples ({n_classes} classes, {n_signers} signers)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    net = MedicalSignFormerV2(
        embed_dim=config.EMBED_DIM,
        num_classes=config.NUM_CLASSES,
        sequence_length=config.SEQUENCE_LENGTH,
    ).to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    net.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint: {CHECKPOINT_PATH.name}\n")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Deterministic evaluation ---
    net.eval()
    all_labels: list[int] = []
    all_preds:  list[int] = []

    with torch.no_grad():
        for features, labels, lengths in loader:
            features = features.to(device)
            lengths  = lengths.to(device)
            logits, _ = net(features, lengths=lengths)
            preds = torch.argmax(logits, dim=-1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())

    present_labels = sorted(set(all_labels))
    present_names  = [label_names[i] for i in present_labels]

    accuracy  = float(np.mean(np.array(all_labels) == np.array(all_preds)))
    report_dict = classification_report(
        all_labels, all_preds,
        labels=present_labels, target_names=present_names,
        output_dict=True, zero_division=0,
    )
    if "accuracy" not in report_dict:
        report_dict["accuracy"] = accuracy

    report_df   = pd.DataFrame(report_dict).transpose()
    report_path = EVAL_DIR / f"{split_name}_classification_report.csv"
    report_df.to_csv(report_path)

    macro_f1    = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]

    print("-" * 40)
    print(f"RESULTS  ({split_name.upper()})")
    print("-" * 40)
    print(f"  Overall Accuracy  : {accuracy * 100:.2f}%")
    print(f"  Macro F1          : {macro_f1:.4f}")
    print(f"  Weighted F1       : {weighted_f1:.4f}")
    print(f"  Report saved  ->  {report_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    fig_size = max(10, len(present_labels) * 0.35)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(present_labels)))
    ax.set_yticks(range(len(present_labels)))
    ax.set_xticklabels(present_names, rotation=90, fontsize=6)
    ax.set_yticklabels(present_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {split_name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_path = EVAL_DIR / f"{split_name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix  ->  {cm_path}")

    # --- 2. Monte Carlo Dropout evaluation ---
    print("\nRunning MC-Dropout evaluation ...")
    enable_mc_dropout(net)

    mc_rows: list[dict] = []
    filepaths  = split_df["filepath"].tolist()
    signers    = split_df["signer"].tolist()
    sample_idx = 0

    with torch.no_grad():
        for features, labels, lengths in loader:
            features = features.to(device)
            lengths  = lengths.to(device)
            result   = mc_dropout_predict(
                net, features, lengths=lengths,
                num_samples=config.MC_DROPOUT_NUM_SAMPLES,
            )
            for true_label, pred, conf, ent, var in zip(
                labels.tolist(),
                result["prediction"].cpu().tolist(),
                result["confidence"].cpu().tolist(),
                result["entropy"].cpu().tolist(),
                result["predicted_class_variance"].cpu().tolist(),
            ):
                mc_rows.append({
                    "true_label":               label_names[true_label],
                    "predicted_label":          label_names[pred],
                    "correct":                  true_label == pred,
                    "confidence":               conf,
                    "entropy":                  ent,
                    "predicted_class_variance": var,
                    "filepath": filepaths[sample_idx] if sample_idx < len(filepaths) else "",
                    "signer":   signers[sample_idx] if sample_idx < len(signers) else "",
                })
                sample_idx += 1

    mc_df   = pd.DataFrame(mc_rows)
    mc_path = EVAL_DIR / f"{split_name}_mc_dropout_predictions.csv"
    mc_df.to_csv(mc_path, index=False)

    print(f"  MC predictions saved  ->  {mc_path}")
    print(f"  Mean Confidence        : {mc_df['confidence'].mean():.4f}")
    print(f"  Mean Predictive Entropy: {mc_df['entropy'].mean():.4f}")
    print(f"  Mean Class Variance    : {mc_df['predicted_class_variance'].mean():.6f}")

    # Per-signer breakdown
    print("\nPer-signer accuracy:")
    for signer in sorted(mc_df["signer"].unique()):
        sub = mc_df[mc_df["signer"] == signer]
        print(f"  {signer:15s}: {sub['correct'].mean()*100:5.1f}%  ({len(sub)} samples)")
    print("=" * 60 + "\n")


def main() -> None:
    print("MSL-Medical Kaggle Cross-Validation")
    print("=" * 60)
    print(f"Reading MSL root CSVs directly:")
    print(f"  Train: {MSL_TRAIN_CSV.name}")
    print(f"  Test:  {MSL_TEST_CSV.name}")
    print(f"Label mapping: {len(MSL_TO_MODEL_LABEL)} MSL classes -> model classes")

    # Load MSL CSVs and translate paths + labels
    train_df = _load_msl_csv(MSL_TRAIN_CSV, LANDMARKS_DIR)
    test_df  = _load_msl_csv(MSL_TEST_CSV,  LANDMARKS_DIR)

    print(f"\nTrain: {len(train_df)} samples, {train_df['label'].nunique()} mapped classes, "
          f"{train_df['signer'].nunique()} signers")
    print(f"Test:  {len(test_df)} samples, {test_df['label'].nunique()} mapped classes, "
          f"{test_df['signer'].nunique()} signers")

    # Check signer overlap
    train_signers = set(train_df["signer"])
    test_signers  = set(test_df["signer"])
    overlap = train_signers & test_signers
    print(f"Signer overlap train/test: {len(overlap)} "
          f"({', '.join(sorted(overlap)) if overlap else 'none'})")

    # Verify .npy files exist
    npy_count = len(list(LANDMARKS_DIR.glob("*.npy")))
    print(f"\nFound {npy_count} .npy files in {LANDMARKS_DIR}")

    missing_train = [p for p in train_df["filepath"] if not (ROOT / p).exists()]
    missing_test  = [p for p in test_df["filepath"] if not (ROOT / p).exists()]
    if missing_train:
        print(f"WARNING: {len(missing_train)} train .npy files missing")
    if missing_test:
        print(f"WARNING: {len(missing_test)} test .npy files missing")
    print()

    evaluate("msl_train", train_df)
    evaluate("msl_test",  test_df)

    print("Cross-validation complete.")


if __name__ == "__main__":
    main()
