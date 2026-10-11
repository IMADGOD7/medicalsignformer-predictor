from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).parent


def load_v1_predictions(v1_csv_path: Path, label_map_path: Path) -> pd.DataFrame:
    df = pd.read_csv(v1_csv_path)
    required_cols = {"filepath", "true_label", "pred_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"v1 predictions CSV is missing required columns: {missing}")

    with label_map_path.open("r", encoding="utf-8") as f:
        name_to_idx = json.load(f)
    idx_to_name = {idx: name for name, idx in name_to_idx.items()}

    df["v1_true_name"] = df["true_label"].map(idx_to_name)
    df["v1_pred_name"] = df["pred_label"].map(idx_to_name)

    if df["v1_true_name"].isna().any() or df["v1_pred_name"].isna().any():
        n_bad = df["v1_true_name"].isna().sum() + df["v1_pred_name"].isna().sum()
        raise ValueError(
            f"{n_bad} v1 label indices could not be mapped to a class name via "
            f"{label_map_path} - v1 and the current label_map.json may disagree "
            f"on class indexing."
        )

    df["v1_correct"] = df["v1_true_name"] == df["v1_pred_name"]
    return df[["filepath", "v1_true_name", "v1_pred_name", "v1_correct"]]


def load_v2_predictions(v2_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(v2_csv_path)
    required_cols = {"filepath", "true_label", "predicted_label", "correct"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"v2 predictions CSV is missing required columns: {missing}. "
            f"Make sure evaluate.py was run with test_csv_path set."
        )
    return df.rename(columns={
        "true_label": "v2_true_name", "predicted_label": "v2_pred_name", "correct": "v2_correct",
    })[["filepath", "v2_true_name", "v2_pred_name", "v2_correct"]]


def pair_predictions(v1_df: pd.DataFrame, v2_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Inner-join on filepath using the intersection. v1 and v2 may have
    different test-set sizes — we compare only shared samples.

    Returns (merged_df, info_dict) where info_dict reports how many were
    excluded from each side.
    """
    merged = v1_df.merge(v2_df, on="filepath", how="inner")

    if len(merged) == 0:
        raise ValueError(
            "Filepath join produced 0 matched samples. "
            "v1 and v2 appear to have no overlap in their test sets."
        )

    info = {
        "n_shared": len(merged),
        "n_v1_total": len(v1_df),
        "n_v2_total": len(v2_df),
        "n_v1_dropped": len(v1_df) - len(merged),
        "n_v2_dropped": len(v2_df) - len(merged),
    }
    if info["n_v1_dropped"] > 0 or info["n_v2_dropped"] > 0:
        print(
            f"  Using {info['n_shared']} shared samples "
            f"(v1={info['n_v1_total']}, v2={info['n_v2_total']}; "
            f"dropped {info['n_v1_dropped']} v1-only, {info['n_v2_dropped']} v2-only)."
        )

    label_mismatch = merged["v1_true_name"] != merged["v2_true_name"]
    if label_mismatch.any():
        raise ValueError(
            f"{label_mismatch.sum()} samples have DIFFERENT true labels between "
            f"v1 and v2 for the same filepath — label_map.json inconsistency. "
            f"Fix the mapping before comparing."
        )

    return merged, info


def run_mcnemar_test(merged: pd.DataFrame) -> dict:
    both_correct = (merged["v1_correct"] & merged["v2_correct"]).sum()
    v1_only = (merged["v1_correct"] & ~merged["v2_correct"]).sum()
    v2_only = (~merged["v1_correct"] & merged["v2_correct"]).sum()
    both_wrong = (~merged["v1_correct"] & ~merged["v2_correct"]).sum()

    contingency = [[int(both_correct), int(v1_only)], [int(v2_only), int(both_wrong)]]
    result = mcnemar(contingency, exact=True)

    return {
        "contingency_table": contingency,
        "both_correct": int(both_correct),
        "v1_only_correct": int(v1_only),
        "v2_only_correct": int(v2_only),
        "both_wrong": int(both_wrong),
        "statistic": result.statistic,
        "p_value": result.pvalue,
    }


def run_paired_f1_ttest(merged: pd.DataFrame) -> dict:
    classes = sorted(set(merged["v1_true_name"]))

    v1_f1_per_class = f1_score(
        merged["v1_true_name"], merged["v1_pred_name"], labels=classes, average=None, zero_division=0,
    )
    v2_f1_per_class = f1_score(
        merged["v2_true_name"], merged["v2_pred_name"], labels=classes, average=None, zero_division=0,
    )

    t_statistic, p_value = stats.ttest_rel(v2_f1_per_class, v1_f1_per_class)

    per_class_df = pd.DataFrame({
        "class": classes, "v1_f1": v1_f1_per_class, "v2_f1": v2_f1_per_class,
        "difference_v2_minus_v1": v2_f1_per_class - v1_f1_per_class,
    }).sort_values("difference_v2_minus_v1")

    return {
        "t_statistic": t_statistic,
        "p_value": p_value,
        "mean_v1_f1": float(np.mean(v1_f1_per_class)),
        "mean_v2_f1": float(np.mean(v2_f1_per_class)),
        "per_class_df": per_class_df,
    }


def main() -> None:
    v1_csv_path = ROOT / "v1_predictions.csv"
    v2_csv_path = ROOT / "evaluation" / "mc_dropout_predictions.csv"
    label_map_path = ROOT / "data" / "processed" / "label_map.json"
    output_dir = ROOT / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions...")
    v1_df = load_v1_predictions(v1_csv_path, label_map_path)
    v2_df = load_v2_predictions(v2_csv_path)
    merged, pair_info = pair_predictions(v1_df, v2_df)

    alpha = 0.05

    print("=" * 60)
    print("OVERALL ACCURACY (on shared samples)")
    print("=" * 60)
    v1_acc = merged["v1_correct"].mean()
    v2_acc = merged["v2_correct"].mean()
    print(f"v1 accuracy: {v1_acc:.4f}  ({pair_info['n_v1_total']} total, {pair_info['n_v1_dropped']} excluded)")
    print(f"v2 accuracy: {v2_acc:.4f}  ({pair_info['n_v2_total']} total, {pair_info['n_v2_dropped']} excluded)")
    print(f"Difference (v2 - v1): {v2_acc - v1_acc:+.4f}\n")

    print("=" * 60)
    print("MCNEMAR'S TEST (paired, sample-level)")
    print("=" * 60)
    mcnemar_result = run_mcnemar_test(merged)
    print(f"Both correct       : {mcnemar_result['both_correct']}")
    print(f"v1 correct, v2 wrong: {mcnemar_result['v1_only_correct']}")
    print(f"v2 correct, v1 wrong: {mcnemar_result['v2_only_correct']}")
    print(f"Both wrong         : {mcnemar_result['both_wrong']}")
    print(f"\nMcNemar statistic  : {mcnemar_result['statistic']:.4f}")
    print(f"p-value            : {mcnemar_result['p_value']:.6f}")
    if mcnemar_result["p_value"] < alpha:
        favored = "v2" if mcnemar_result["v2_only_correct"] > mcnemar_result["v1_only_correct"] else "v1"
        print(f"-> Statistically significant (p < {alpha}); favors {favored}.")
    else:
        print(f"-> NOT statistically significant (p >= {alpha}).")

    print("\n" + "=" * 60)
    print("PAIRED T-TEST ON PER-CLASS F1 SCORES")
    print("=" * 60)
    ttest_result = run_paired_f1_ttest(merged)
    print(f"Mean v1 per-class F1: {ttest_result['mean_v1_f1']:.4f}")
    print(f"Mean v2 per-class F1: {ttest_result['mean_v2_f1']:.4f}")
    print(f"t-statistic: {ttest_result['t_statistic']:.4f}")
    print(f"p-value    : {ttest_result['p_value']:.6f}")
    if ttest_result["p_value"] < alpha:
        print(f"-> Statistically significant (p < {alpha}).")
    else:
        print(f"-> NOT statistically significant (p >= {alpha}).")

    per_class_path = output_dir / "v1_vs_v2_per_class_f1.csv"
    ttest_result["per_class_df"].to_csv(per_class_path, index=False)
    print(f"\nPer-class F1 comparison saved to: {per_class_path}")

    print("\nClasses with the largest v2 improvement over v1:")
    print(ttest_result["per_class_df"].tail(5).to_string(index=False))
    print("\nClasses with the largest v2 regression vs v1:")
    print(ttest_result["per_class_df"].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
