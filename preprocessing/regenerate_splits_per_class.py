"""
Per-class stratified signer split for MedicalSignFormer.

Fixes a structural bug in the previous global-shuffle approach: with 48
classes and highly uneven signer group sizes (some signers contributing
150+ samples to a single class), a global random signer shuffle could land
an ENTIRE class 100% in one split.

Algorithm:
- Each signer is assigned to exactly ONE split globally (never split across
  splits) - this preserves strict signer-independence, so no person's
  recordings ever appear in more than one split.
- Classes are processed in order of FEWEST unique signers first (most
  constrained first).
- For each class, if train/val/test coverage is missing, the smallest
  remaining unassigned signer is assigned to the missing split first.
- Remaining unassigned signers are assigned to whichever split is furthest
  below its target share (70/10/20) for that class.
- Classes with fewer than 3 unique signers CANNOT have full 3-way coverage
  without violating signer-independence - reported as a known limitation.

Split ratios: TRAIN=0.70, VAL=0.10, TEST=0.20
  - train.csv  : used for fine-tuning
  - val.csv    : used for early stopping only
  - test.csv   : held-out, touched ONCE at final evaluation via evaluate.py

Usage:
    python regenerate_splits_per_class.py
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import sys

# Add project root to Python path BEFORE importing config
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO

try:
    import pandas as pd
except Exception as e:
    print("Missing dependency:", e)
    print("Please install pandas: pip install pandas")
    sys.exit(1)

ROOT = Path(__file__).parents[1] if (Path(__file__).parent.name in ('tools', 'preprocessing')) else Path(__file__).parent
INDEX = ROOT / 'dataset_index.csv'
OUTDIR = ROOT / 'data' / 'processed'
OUTDIR.mkdir(parents=True, exist_ok=True)



TRAIN_TARGET = TRAIN_RATIO
VAL_TARGET = VAL_RATIO
TEST_TARGET = TEST_RATIO
SPLITS = ['train', 'val', 'test']  # 70/10/20: train for learning, val for early stopping, test held-out

if not INDEX.exists():
    print('dataset_index.csv not found at', INDEX)
    sys.exit(1)

print('Loading index...')
df = pd.read_csv(INDEX)
if 'label' not in df.columns or 'signer' not in df.columns:
    print('Index must contain "label" and "signer" columns. Exiting.')
    sys.exit(1)

all_labels = sorted(df['label'].unique())
print(f'Total samples: {len(df)}, total classes: {len(all_labels)}')

# --- Build lookup structures ---
# signer_class_counts[signer][label] = count of samples for that signer/class
signer_class_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
for _, row in df.iterrows():
    signer_class_counts[row['signer']][row['label']] += 1

# class_signers[label] = list of (signer, count) for that class
class_signers: dict[int, list[tuple[str, int]]] = defaultdict(list)
for signer, class_counts in signer_class_counts.items():
    for label, count in class_counts.items():
        class_signers[label].append((signer, count))

# Process classes with the FEWEST unique signers first (most constrained)
classes_by_constraint = sorted(all_labels, key=lambda lbl: len(class_signers[lbl]))

global_signer_split: dict[str, str] = {}  # signer -> assigned split (locked once set)
per_class_split_totals: dict[int, dict[str, int]] = {lbl: {s: 0 for s in SPLITS} for lbl in all_labels}
incomplete_classes: list[tuple[int, list[str]]] = []  # (label, missing_splits)

print('\nAssigning signers to splits, class by class (most constrained first)...\n')

for label in classes_by_constraint:
    signers_here = class_signers[label]
    n_signers = len(signers_here)

    # Which splits are already covered for this class via previously-locked signers?
    covered_splits = set()
    unassigned_here = []
    for signer, count in signers_here:
        if signer in global_signer_split:
            covered_splits.add(global_signer_split[signer])
        else:
            unassigned_here.append((signer, count))

    missing_splits = [s for s in SPLITS if s not in covered_splits]

    # Sort unassigned signers smallest-first, so we spend the smallest ones
    # on filling coverage gaps (minimizes distortion to overall train share).
    unassigned_here.sort(key=lambda sc: sc[1])

    # Step A: fill missing splits first, if any unassigned signers are available.
    for missing_split in missing_splits:
        if not unassigned_here:
            break
        signer, count = unassigned_here.pop(0)
        global_signer_split[signer] = missing_split
        per_class_split_totals[label][missing_split] += count

    # Recompute what's still missing after Step A.
    still_missing = [
        s for s in SPLITS
        if per_class_split_totals[label][s] == 0
        and not any(
            global_signer_split.get(sig) == s for sig, _ in signers_here
        )
    ]
    if still_missing:
        incomplete_classes.append((label, still_missing))

    # Step B: assign remaining unassigned signers to whichever split is
    # currently furthest below its target share for this class. Compute
    # "current totals so far" by summing over THIS class's signers using the
    # real global_signer_split lock state (includes contributions from
    # signers that were already locked while processing an earlier class
    # that happens to share this signer) - not the incremental dict, which
    # only reflects assignments made while processing this class itself.
    for signer, count in unassigned_here:
        current_totals = {s: 0 for s in SPLITS}
        for sig, cnt in signers_here:
            assigned_split = global_signer_split.get(sig)
            if assigned_split is not None:
                current_totals[assigned_split] += cnt
        current_total_sum = sum(current_totals.values())
        targets = {'train': TRAIN_TARGET, 'val': VAL_TARGET, 'test': TEST_TARGET}
        deficits = {
            s: targets[s] - (current_totals[s] / current_total_sum if current_total_sum > 0 else 0)
            for s in SPLITS
        }
        best_split = max(deficits, key=deficits.get)
        global_signer_split[signer] = best_split
        per_class_split_totals[label][best_split] += count

# Any signer that appears in the dataset but somehow never got assigned
# (shouldn't happen, but guard against it) defaults to train.
for signer in signer_class_counts:
    if signer not in global_signer_split:
        global_signer_split[signer] = 'train'

# --- Repair pass: fix classes that have enough signers but missed coverage
# purely due to processing order (a signer that could have covered them was
# claimed first by another class). This does NOT attempt to fix classes with
# a genuine hard limitation (e.g. only 2 unique signers total for 3 splits).
print('\nRunning repair pass for classes with recoverable coverage gaps...\n')

def class_missing_splits(label: int) -> list[str]:
    """Return which splits currently have zero coverage for this class."""
    covered = set()
    for signer, _count in class_signers[label]:
        covered.add(global_signer_split.get(signer))
    return [s for s in SPLITS if s not in covered]


def signer_is_sole_coverage(signer: str, split: str) -> list[int]:
    """Return list of class labels for which `signer` is the ONLY signer
    currently assigned to `split` - i.e. classes that would lose coverage of
    `split` if `signer` were moved away from it."""
    affected = []
    for label in signer_class_counts[signer]:
        signers_in_split = [
            sig for sig, _cnt in class_signers[label]
            if global_signer_split.get(sig) == split
        ]
        if signers_in_split == [signer]:
            affected.append(label)
    return affected


repaired = []
still_unrepairable = []

for label in all_labels:
    missing = class_missing_splits(label)
    if not missing:
        continue

    n_signers = len(class_signers[label])
    if n_signers < 3:
        # Hard mathematical limitation - cannot cover 3 splits with fewer
        # than 3 signers without violating signer-independence.
        still_unrepairable.append((label, missing, f'{n_signers} signers total'))
        continue

    for missing_split in missing:
        # Find a split currently covering this class with MORE THAN ONE
        # signer - a donor could come from there without leaving that split
        # uncovered for this class.
        split_signer_counts: dict[str, list[str]] = defaultdict(list)
        for signer, _count in class_signers[label]:
            assigned = global_signer_split.get(signer)
            if assigned is not None:
                split_signer_counts[assigned].append(signer)

        donor_found = False
        for donor_split, signers_in_split in split_signer_counts.items():
            if len(signers_in_split) < 2:
                continue  # would leave donor_split uncovered for this class
            for candidate_signer in signers_in_split:
                # Check moving this signer doesn't break sole-coverage for
                # ANY class (including this one's own donor_split, already
                # guarded above, but also any other class relying on it).
                broken = signer_is_sole_coverage(candidate_signer, donor_split)
                if broken:
                    continue  # would break coverage elsewhere - try next candidate
                # Safe to move.
                global_signer_split[candidate_signer] = missing_split
                repaired.append((label, missing_split, candidate_signer, donor_split))
                donor_found = True
                break
            if donor_found:
                break

        if not donor_found:
            still_unrepairable.append(
                (label, [missing_split], 'no safe donor signer found')
            )

if repaired:
    print(f'Repaired {len(repaired)} coverage gap(s):')
    for label, split, signer, donor_split in repaired:
        print(f'  label {label}: moved signer "{signer}" from {donor_split} '
              f'-> {split} to fill the gap')
else:
    print('No repairable gaps found (remaining gaps are hard limitations).')


df['split'] = df['signer'].map(global_signer_split)

train_df = df[df['split'] == 'train'].drop(columns=['split']).copy()
val_df   = df[df['split'] == 'val'].drop(columns=['split']).copy()
test_df  = df[df['split'] == 'test'].drop(columns=['split']).copy()

train_file = OUTDIR / 'train.csv'
val_file   = OUTDIR / 'val.csv'
test_file  = OUTDIR / 'test.csv'
train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)

# --- Report ---
print('=' * 60)
print('PER-CLASS STRATIFIED SPLIT SUMMARY')
print('=' * 60)

total = len(df)
print(f'\nOverall split: train={len(train_df)} ({len(train_df)/total:.1%}), '
      f'val={len(val_df)} ({len(val_df)/total:.1%}), '
      f'test={len(test_df)} ({len(test_df)/total:.1%})')

train_signers = set(train_df['signer'])
val_signers   = set(val_df['signer'])
test_signers  = set(test_df['signer'])
print(f'\nSigner overlap check (should all be 0):')
print(f'  train-val  : {len(train_signers & val_signers)}')
print(f'  train-test : {len(train_signers & test_signers)}')
print(f'  val-test   : {len(val_signers & test_signers)}')


final_per_class_totals: dict[int, dict[str, int]] = {
    lbl: {'train': 0, 'val': 0, 'test': 0} for lbl in all_labels
}
for label, count in train_df['label'].value_counts().items():
    final_per_class_totals[label]['train'] = int(count)
for label, count in val_df['label'].value_counts().items():
    final_per_class_totals[label]['val'] = int(count)
for label, count in test_df['label'].value_counts().items():
    final_per_class_totals[label]['test'] = int(count)

print(f'\nPer-class coverage (recomputed from final saved splits):')
final_incomplete_classes: list[tuple[int, list[str]]] = []
for label in all_labels:
    t = final_per_class_totals[label]
    zero_splits = [s for s in SPLITS if t[s] == 0]
    flag = f'  <-- MISSING: {zero_splits}' if zero_splits else ''
    print(f'  label {label}: train={t["train"]}, val={t["val"]}, test={t["test"]}{flag}')
    if zero_splits:
        final_incomplete_classes.append((label, zero_splits))

if final_incomplete_classes:
    print(f'\n{"=" * 60}')
    print(f'REMAINING GAPS: {len(final_incomplete_classes)} class(es) still missing full')
    print('coverage after the repair pass.')
    print('=' * 60)
    for label, missing in final_incomplete_classes:
        n_signers = len(class_signers[label])
        reason = next(
            (r for lbl, m, r in still_unrepairable if lbl == label),
            'unknown'
        )
        print(f'  label {label}: {n_signers} unique signer(s) total, missing {missing} ({reason})')
else:
    print('\nAll classes achieved full train/val/test coverage.')

print('\nRegeneration complete.')