# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 1b — inject outliers into the BN-stat collection set and measure the effect (+ a robust fix).

For each batch we take a COPY of its windows as the "collection set", corrupt a fraction `f` of them
with an amplitude spike (×8 — simulates electrode pop / motion artifact; since EMG is unnormalized,
this inflates that window's variance contribution ~×64), collect BN stats on it, then evaluate on the
CLEAN batch. So only the stat-collection data is contaminated (realistic: the unlabeled incoming data
used for AdaBN may contain artifacts), isolating "corrupted stats -> worse accuracy".

Two collection methods:
  standard : collect on all windows (including corrupted)  -> stats get polluted
  robust   : reject outlier windows first (per-window max-abs amplitude > median + 3·MAD),
             then collect on survivors -> a simple deployment-side artifact reject

Recollect-only (no FT), base weights. S01 + S02, f in {0,2,5,10,20}%, several seeds.
CLI:  python3 outlier_injection.py S01 --out results/outlier_S01.csv
"""
import argparse, os, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from adabn_full_training import make_model, collect_bn_stats   # noqa: E402
from windowing import load_windows                             # noqa: E402
from ondevice_ft import balanced_accuracy                      # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
FOLDS = [1, 2, 3]
BATCHES = [2, 3, 4, 5]
FRACS = [0.0, 0.02, 0.05, 0.10, 0.20]
SPIKE = 8.0
N_SEEDS = 6


def corrupt(X, frac, seed):
    """Return a copy with `frac` of windows amplitude-scaled by SPIKE."""
    Xc = X.copy()
    n = len(Xc); k = int(round(frac * n))
    if k > 0:
        idx = np.random.RandomState(seed).choice(n, k, replace=False)
        Xc[idx] = Xc[idx] * SPIKE
    return Xc


def robust_reject(X):
    """Drop windows whose max-abs amplitude exceeds median + 3·MAD (per-window scalar)."""
    amp = np.abs(X.reshape(len(X), -1)).max(axis=1)
    med = np.median(amp); mad = np.median(np.abs(amp - med)) + 1e-9
    keep = amp <= med + 3.0 * 1.4826 * mad
    return X[keep] if keep.sum() >= 2 else X


def run(subject, cond):
    rows = []
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{subject}/{cond}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        for b in BATCHES:
            Xe, ye = load_windows(DATA, subject, fold, b, cond, downsample_rest=True)
            for f in FRACS:
                for s in range(N_SEEDS):
                    Xc = corrupt(Xe, f, seed=2000 + s)
                    for method in ("standard", "robust"):
                        Xcol = robust_reject(Xc) if method == "robust" else Xc
                        m = make_model(ckpt)
                        collect_bn_stats(m, Xcol)
                        acc = balanced_accuracy(m, Xe, ye)     # eval on CLEAN batch
                        rows.append((fold, b, f, s, method, acc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--cond", default="vocalized")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    df = pd.DataFrame(run(a.subject, a.cond),
                      columns=["fold", "batch", "frac", "seed", "method", "acc"])
    df["subject"] = a.subject
    df.to_csv(os.path.join(HERE, a.out), index=False)
    print(f"=== {a.subject} outlier injection (recollect-only, eval on clean batch) ===")
    print(f"{'frac':>5} {'standard':>16} {'robust':>16}")
    for f in FRACS:
        st = df[(df.frac == f) & (df.method == "standard")].acc
        ro = df[(df.frac == f) & (df.method == "robust")].acc
        print(f"{int(f*100):>4}% {st.mean():>7.2f}±{st.std(ddof=1):<5.2f}   "
              f"{ro.mean():>7.2f}±{ro.std(ddof=1):<5.2f}", flush=True)


if __name__ == "__main__":
    main()
