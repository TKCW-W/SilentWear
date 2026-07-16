# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Append the paper's counterpart numbers (full-model Adam FT) to our on-device SUBJECT_MEAN table.

Reads the paper's ft_summary.csv (all folds x batches), aggregates per batch across folds
(mean/std of zero_shot_balanced_acc = paper FT, and balanced_acc_no_ft = paper base), and
appends them as `paper_*` columns to ft_summary_ondevice_<subj>_<cond>_SUBJECT_MEAN.csv.
"""
import argparse
import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S01")
    ap.add_argument("--condition", default="vocalized")
    ap.add_argument("--ddof", type=int, default=1)
    ap.add_argument("--artifacts-root", default="/app/SilentWear/SilentWear/artifacts/models")
    args = ap.parse_args()

    # locate the paper ft_summary.csv (ft_config_* dir)
    pat = (f"{args.artifacts_root}/inter_session_ft/{args.subject}/{args.condition}/"
           f"speechnet/w1400ms/ft_config_*/ft_summary.csv")
    paper_files = sorted(glob.glob(pat))
    if not paper_files:
        raise SystemExit(f"paper ft_summary.csv not found: {pat}")
    paper = pd.read_csv(paper_files[0])
    print(f"paper: {paper_files[0]}  ({paper['base_model'].nunique()} folds)")

    # aggregate paper per batch across folds (values are in [0,1] -> x100)
    g = paper.groupby("zero_shot_test_batch")
    pg = pd.DataFrame({
        "batch": sorted(paper["zero_shot_test_batch"].unique()),
        "paper_no_ft_mean": (g["balanced_acc_no_ft"].mean() * 100).values,
        "paper_no_ft_std": (g["balanced_acc_no_ft"].std(ddof=args.ddof) * 100).values,
        "paper_ft_mean": (g["zero_shot_balanced_acc"].mean() * 100).values,
        "paper_ft_std": (g["zero_shot_balanced_acc"].std(ddof=args.ddof) * 100).values,
    })
    pg["paper_ft_gain_mean"] = pg["paper_ft_mean"] - pg["paper_no_ft_mean"]

    # merge into our SUBJECT_MEAN
    ours_path = os.path.join(HERE, "results",
                             f"ft_summary_ondevice_{args.subject}_{args.condition}_SUBJECT_MEAN.csv")
    ours = pd.read_csv(ours_path)
    merged = ours.merge(pg, on="batch", how="left")
    for c in merged.columns:
        if c not in ("batch", "n_folds"):
            merged[c] = merged[c].round(2)
    merged.to_csv(ours_path, index=False)

    # readable comparison view
    view = merged[["batch", "no_ft_mean", "ft_mean", "ft_std",
                   "paper_ft_mean", "paper_ft_std"]].copy()
    view.columns = ["batch", "no_ft(base)", "ours_ft", "ours_std", "paper_ft", "paper_std"]
    view["ours-paper"] = (merged["ft_mean"] - merged["paper_ft_mean"]).round(2)
    print(view.to_string(index=False))
    print(f"\nsaved (appended paper_* cols) -> {ours_path}")
    # sanity: our no_ft must equal the paper's no_ft (same base model + eval)
    d = (merged["no_ft_mean"] - merged["paper_no_ft_mean"]).abs().max()
    print(f"[check] max|our no_ft_mean - paper no_ft_mean| = {d:.2f} pp (should be ~0)")


if __name__ == "__main__":
    main()
