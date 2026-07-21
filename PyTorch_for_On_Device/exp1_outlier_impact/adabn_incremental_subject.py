# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 1a — AdaBN full incremental for a chosen subject (default S02 = noisiest), vs paper reference.

Same decided AdaBN flow as run_inter_session_ft_adabn.py (per batch: collect BN stats on the batch,
eval, then full-train on 30%), best config n_accum=8 lr=3e-4, but subject-parameterized. Tells us how
AdaBN holds up on a noisy subject where the recollected stats are most at risk from outliers.

CLI:  python3 adabn_incremental_subject.py S02 --out results/adabn_S02_vocalized.csv
"""
import argparse, os, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from adabn_full_training import make_model, collect_bn_stats, full_train   # noqa: E402
from windowing import load_windows, stratified_draw                       # noqa: E402
from ondevice_ft import balanced_accuracy                                  # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
FOLDS = [1, 2, 3]
N_ACCUM, LR, EPOCHS = 8, 3e-4, 40


def run_fold(subject, cond, fold):
    ckpt = (f"{ART}/inter_session_ft/{subject}/{cond}/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{fold}.pt")
    base = make_model(ckpt); m = make_model(ckpt)
    rows = []
    for b in range(1, 6):
        Xe, ye = load_windows(DATA, subject, fold, b, cond, downsample_rest=True)
        no_ft = balanced_accuracy(base, Xe, ye)
        collect_bn_stats(m, Xe)
        zs = balanced_accuracy(m, Xe, ye)
        rows.append((b, no_ft, zs))
        if b != 5:
            Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
            full_train(m, Xtr, ytr, LR, N_ACCUM, epochs=EPOCHS)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--cond", default="vocalized")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    byb = {b: {"nf": [], "zs": []} for b in range(1, 6)}
    for fold in FOLDS:
        for b, nf, zs in run_fold(a.subject, a.cond, fold):
            byb[b]["nf"].append(nf); byb[b]["zs"].append(zs)
        print(f"[{a.subject}] fold {fold} done", flush=True)
    rows = []
    for b in range(1, 6):
        nf = np.array(byb[b]["nf"]); zs = np.array(byb[b]["zs"])
        rows.append(dict(batch=b, no_ft_mean=round(nf.mean(), 2), no_ft_std=round(nf.std(ddof=1), 2),
                         adabn_mean=round(zs.mean(), 2), adabn_std=round(zs.std(ddof=1), 2)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, a.out), index=False)
    m25 = np.mean([r["adabn_mean"] for r in rows if r["batch"] >= 2])
    print(df.to_string(index=False))
    print(f"[{a.subject}] AdaBN mean b2-5 = {m25:.2f}", flush=True)


if __name__ == "__main__":
    main()
