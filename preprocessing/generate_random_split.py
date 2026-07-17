
from __future__ import annotations

from pathlib import Path
import sys

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
except Exception as e:
    print("Missing dependency:", e)
    print("Please install pandas and scikit-learn: pip install pandas scikit-learn")
    sys.exit(1)

ROOT = Path(__file__).parents[1] if (Path(__file__).parent.name in ("tools", "preprocessing")) else Path(__file__).parent
INDEX = ROOT / "dataset_index.csv"
OUTDIR = ROOT / "data" / "processed"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

if not INDEX.exists():
    print("dataset_index.csv not found at", INDEX)
    sys.exit(1)

print("Loading index...")
df = pd.read_csv(INDEX)
if "label" not in df.columns:
    print("Index must contain a 'label' column. Exiting.")
    sys.exit(1)

print(f"Total samples: {len(df)}, total classes: {df['label'].nunique()}")
print(
    f"Target split: {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / "
    f"{TEST_RATIO:.0%} test, stratified by class, signers NOT kept disjoint.\n"
)

# First split: train vs temp (val+test), stratified by label.
temp_size = VAL_RATIO + TEST_RATIO
train_df, temp_df = train_test_split(
    df,
    test_size=temp_size,
    stratify=df["label"],
    random_state=RANDOM_SEED,
)

# Second split: temp -> val/test, stratified by label, preserving the
# relative val:test ratio within the temp portion.
val_share_of_temp = VAL_RATIO / temp_size
val_df, test_df = train_test_split(
    temp_df,
    test_size=(1 - val_share_of_temp),
    stratify=temp_df["label"],
    random_state=RANDOM_SEED,
)

train_file = OUTDIR / "train.csv"
val_file = OUTDIR / "val.csv"
test_file = OUTDIR / "test.csv"
train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)

print("=" * 60)
print("RANDOM SPLIT SUMMARY (stratified by class, signers mixed)")
print("=" * 60)
total = len(df)
print(
    f"train={len(train_df)} ({len(train_df)/total:.1%}), "
    f"val={len(val_df)} ({len(val_df)/total:.1%}), "
    f"test={len(test_df)} ({len(test_df)/total:.1%})"
)

all_labels = sorted(df["label"].unique())
print(f"\nAll classes present in train: {train_df['label'].nunique() == len(all_labels)}")
print(f"All classes present in val:   {val_df['label'].nunique() == len(all_labels)}")
print(f"All classes present in test:  {test_df['label'].nunique() == len(all_labels)}")

if "signer" in df.columns:
    train_signers = set(train_df["signer"])
    val_signers = set(val_df["signer"])
    test_signers = set(test_df["signer"])
    print(f"\nSigner overlap (EXPECTED to be > 0 - signers are mixed by design):")
    print(f"  train-val: {len(train_signers & val_signers)}")
    print(f"  train-test: {len(train_signers & test_signers)}")
    print(f"  val-test: {len(val_signers & test_signers)}")

print("\nRegeneration complete.")
print(
    "\nNOTE: this split does NOT guarantee signer independence. If you need "
    "a leakage-free split, use regenerate_splits_per_class.py instead."
)