"""
Signer-stratified train/val/test split — global signer assignment.

Since each signer signs multiple classes, we assign signers globally
(not per-class) so each signer appears in exactly one partition.

Strategy:
  1. Sort all signers by total sample count.
  2. Greedily assign to train until ~TRAIN_RATIO, then val, then test.
  3. Every signer's samples go entirely to their assigned split.

This guarantees zero signer overlap across splits.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print("Missing dependency:", e)
    print("pip install pandas numpy")
    sys.exit(1)

ROOT = (
    Path(__file__).parents[1]
    if Path(__file__).parent.name in ("tools", "preprocessing")
    else Path(__file__).parent
)
INDEX  = ROOT / "dataset_index.csv"
OUTDIR = ROOT / "data" / "processed"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20
RANDOM_SEED = 42

# ── Load ──────────────────────────────────────────────────────────────────────
if not INDEX.exists():
    print(f"dataset_index.csv not found at {INDEX}")
    sys.exit(1)

df = pd.read_csv(INDEX)
required = {"filepath", "label", "signer"}
if not required.issubset(df.columns):
    print(f"Missing columns: {required - set(df.columns)}")
    sys.exit(1)

df["signer"] = df["signer"].str.strip().str.lower()

print(f"Total samples : {len(df)}")
print(f"Total classes : {df['label'].nunique()}")
print(f"Total signers : {df['signer'].nunique()}")
print(f"Target split  : {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}  (signer-disjoint globally)\n")

# ── Global signer assignment ──────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)

# Count samples per signer, shuffle within same-size groups for fairness
signer_counts = (
    df.groupby("signer").size()
    .reset_index(name="n")
    .sort_values("n", ascending=False)
)
signers_all = signer_counts["signer"].tolist()
rng.shuffle(signers_all)   # break ties randomly

total = len(df)
n_train_target = int(total * TRAIN_RATIO)
n_val_target   = int(total * VAL_RATIO)

train_signers, val_signers, test_signers = [], [], []
n_train, n_val = 0, 0

for s in signers_all:
    s_count = int(signer_counts.loc[signer_counts["signer"] == s, "n"].values[0])
    if n_train < n_train_target:
        train_signers.append(s)
        n_train += s_count
    elif n_val < n_val_target:
        val_signers.append(s)
        n_val += s_count
    else:
        test_signers.append(s)

# Edge case: ensure val and test are non-empty
if not val_signers and len(train_signers) > 1:
    val_signers.append(train_signers.pop())
if not test_signers and len(train_signers) > 1:
    test_signers.append(train_signers.pop())

print(f"Train signers ({len(train_signers)}): {sorted(train_signers)}")
print(f"Val   signers ({len(val_signers)}):   {sorted(val_signers)}")
print(f"Test  signers ({len(test_signers)}):  {sorted(test_signers)}")
print()

# Build split dataframes
train_df = df[df["signer"].isin(train_signers)].reset_index(drop=True)
val_df   = df[df["signer"].isin(val_signers)].reset_index(drop=True)
test_df  = df[df["signer"].isin(test_signers)].reset_index(drop=True)

# ── Verify ────────────────────────────────────────────────────────────────────
ts = set(train_signers); vs = set(val_signers); es = set(test_signers)
assert not (ts & vs), f"Train/val overlap: {ts & vs}"
assert not (ts & es), f"Train/test overlap: {ts & es}"
assert not (vs & es), f"Val/test overlap: {vs & es}"

missing_train = set(df["label"].unique()) - set(train_df["label"].unique())
missing_val   = set(df["label"].unique()) - set(val_df["label"].unique())
missing_test  = set(df["label"].unique()) - set(test_df["label"].unique())

# ── Save ──────────────────────────────────────────────────────────────────────
train_df.to_csv(OUTDIR / "train.csv", index=False)
val_df.to_csv(OUTDIR / "val.csv",     index=False)
test_df.to_csv(OUTDIR / "test.csv",   index=False)

all_diseases = sorted(df["disease"].unique().tolist())
label_map = {
    disease: int(df.loc[df["disease"] == disease, "label"].iloc[0])
    for disease in all_diseases
}
with open(OUTDIR / "label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SIGNER-STRATIFIED SPLIT SUMMARY")
print("=" * 60)
print(f"train : {len(train_df):5d}  ({len(train_df)/total:.1%})")
print(f"val   : {len(val_df):5d}  ({len(val_df)/total:.1%})")
print(f"test  : {len(test_df):5d}  ({len(test_df)/total:.1%})")
print()
print(f"Classes in train : {train_df['label'].nunique()} / 48")
print(f"Classes in val   : {val_df['label'].nunique()} / 48")
print(f"Classes in test  : {test_df['label'].nunique()} / 48")
print()
print("Signer disjointness: OK (verified by assertion)")

if missing_train:
    print(f"\nWARNING - classes missing from train: {missing_train}")
if missing_val:
    print(f"WARNING - classes missing from val  : {missing_val}")
if missing_test:
    print(f"WARNING - classes missing from test : {missing_test}")

print(f"\nFiles written to: {OUTDIR}")
print("  train.csv / val.csv / test.csv / label_map.json")
print("\nRun train_finetune.py to start training with the new signer-disjoint split.")
