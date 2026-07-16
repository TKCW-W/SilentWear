# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Aggregate the per-fold ft_summary_ondevice CSVs of one subject/condition into a per-batch
mean ± std across folds (for both the no-FT base accuracy and the fine-tuned accuracy).

Output: results/ft_summary_ondevice_<subject>_<condition>_SUBJECT_MEAN.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S01")
    ap.add_argument("--condition", default="vocalized")
    ap.add_argument("--ddof", type=int, default=1, help="std ddof (1 = sample std across folds)")
    args = ap.parse_args()

    pat = os.path.join(HERE, "results",
                       f"ft_summary_ondevice_{args.subject}_{args.condition}_fold*.csv")
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"no fold CSVs found: {pat}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    folds = sorted(df["base_model"].unique())
    print(f"{args.subject}/{args.condition}: {len(folds)} folds -> {folds}\n")

    g = df.groupby("zero_shot_test_batch")
    out = pd.DataFrame({
        "batch": sorted(df["zero_shot_test_batch"].unique()),
        "n_folds": g.size().values,
        "no_ft_mean": g["balanced_acc_no_ft"].mean().values,
        "no_ft_std": g["balanced_acc_no_ft"].std(ddof=args.ddof).values,
        "ft_mean": g["zero_shot_balanced_acc"].mean().values,
        "ft_std": g["zero_shot_balanced_acc"].std(ddof=args.ddof).values,
    })
    out["ft_gain_mean"] = out["ft_mean"] - out["no_ft_mean"]
    for c in out.columns:
        if c not in ("batch", "n_folds"):
            out[c] = out[c].round(2)

    outpath = os.path.join(HERE, "results",
                           f"ft_summary_ondevice_{args.subject}_{args.condition}_SUBJECT_MEAN.csv")
    out.to_csv(outpath, index=False)
    print(out.to_string(index=False))
    print(f"\nsaved -> {outpath}")

    # overall (mean over batches) for the fine-tuned accuracy
    print(f"\nSubject {args.subject} — fine-tuned accuracy, mean over batches 1-5: "
          f"{out['ft_mean'].mean():.2f}%  |  no-FT base: {out['no_ft_mean'].mean():.2f}%  |  "
          f"mean gain: {out['ft_gain_mean'].mean():+.2f} pp")


if __name__ == "__main__":
    main()
