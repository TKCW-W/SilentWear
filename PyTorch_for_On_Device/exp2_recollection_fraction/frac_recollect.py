# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 2 — how many windows are enough to recollect the BN running stats?

Recollect-only probe (no fine-tuning): for a batch, collect BN stats on K randomly-drawn windows
(at the base pretrained weights), then evaluate balanced accuracy on the whole batch. Sweeping K
tells us the smallest collection size whose stats are as good as the full-batch (K=180) stats, and
— via the spread over random draws — how outlier-sensitive small K is.

K=180 (full) reproduces the existing recollect-only number. S01 (clean) and S02 (noisy) so the
robustness question is answered directly. Config: AdaBN best is n8/3e-4, but recollection quality is
weight-independent, so this probe uses the base model (no FT) — the cleanest isolation of stat quality.

CLI:  python3 frac_recollect.py <subject> <K> --seeds 8 --out results/frac_<subject>_K<K>.csv
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


def run(subject, cond, K, n_seeds):
    per = []   # (fold, batch, seed, acc)
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{subject}/{cond}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        for b in BATCHES:
            Xe, ye = load_windows(DATA, subject, fold, b, cond, downsample_rest=True)
            for s in range(n_seeds):
                m = make_model(ckpt)
                if K < len(Xe):
                    idx = np.random.RandomState(1000 + s).choice(len(Xe), K, replace=False)
                    Xcol = Xe[idx]
                else:
                    Xcol = Xe
                collect_bn_stats(m, Xcol)
                per.append((fold, b, s, balanced_accuracy(m, Xe, ye)))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("K", type=int)
    ap.add_argument("--cond", default="vocalized")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    per = run(a.subject, a.cond, a.K, a.seeds)
    df = pd.DataFrame(per, columns=["fold", "batch", "seed", "acc"])
    df["subject"] = a.subject; df["K"] = a.K
    df.to_csv(os.path.join(HERE, a.out), index=False)
    # summary: pool all (fold,batch,seed) -> mean, std, min (worst draw = outlier sensitivity)
    a_all = df["acc"].values
    # per-batch 3-fold-mean averaged over seeds, then mean over b2-5 (matches other tables)
    b25 = np.mean([df[(df.batch == b)].groupby("seed")["acc"].apply(
        lambda x: x.mean()).mean() for b in BATCHES])
    print(f"{a.subject} K={a.K:>3}: mean_b2_5={b25:.2f} | pooled {a_all.mean():.2f}±{a_all.std(ddof=1):.2f} "
          f"min-draw {a_all.min():.1f}", flush=True)


if __name__ == "__main__":
    main()
