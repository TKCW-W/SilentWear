# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Optimizer-matched naive baseline for the AdaBN study: the SAME on-device full-training recipe
(SGD no-momentum, batch-1 + n_accum SUM, fixed lr, 40 ep, 30% data, full model) but with
LIVE BatchNorm (train-mode batch stats at batch-1) instead of collected-frozen stats.

This is the exact "naive" counterpart to adabn_full_training.py — same everything except the BN
handling — so (adabn - livebn) is the clean AdaBN gain. b1->b2, S01 vocalized, 3 folds,
same (n_accum, lr) grid.  Usage: python3 livebn_full_training.py 8:0.0003 ... --out X.csv
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from adabn_full_training import make_model
from windowing import load_windows, stratified_draw
from ondevice_ft import balanced_accuracy

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND, FOLDS = "S01", "vocalized", [1, 2, 3]
EPOCHS = 40


def full_train_livebn(model, X, y, lr, n_accum, epochs=EPOCHS, seed=42):
    """Full training with LIVE BN: model.train() so BN uses batch-1 stats and updates running stats.
    Everything else identical to adabn_full_training.full_train (SGD, batch-1, n_accum SUM)."""
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()                                    # LIVE BN: batch-1 stats + running EMA update
        perm = rng.permutation(N); opt.zero_grad(); c = 0
        for j in perm:
            crit(model(Xt[j:j + 1]), yt[j:j + 1]).backward(); c += 1
            if c % n_accum == 0:
                opt.step(); opt.zero_grad()
        if c % n_accum:
            opt.step(); opt.zero_grad()
    model.eval(); return model                           # eval uses the EMA running stats it accumulated


def run_config(n_accum, lr):
    accs = []
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        Xb1, yb1 = load_windows(DATA, SUBJECT, fold, 1, COND, downsample_rest=True)
        Xtr, ytr = stratified_draw(Xb1, yb1, 6, seed=42)
        Xb2, yb2 = load_windows(DATA, SUBJECT, fold, 2, COND, downsample_rest=True)
        m = make_model(ckpt)
        full_train_livebn(m, Xtr, ytr, lr, n_accum)
        accs.append(balanced_accuracy(m, Xb2, yb2))       # eval-mode: uses accumulated EMA stats
    return np.mean(accs), np.std(accs, ddof=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("combos", nargs="+")
    ap.add_argument("--out", default="livebn_sweep.csv")
    args = ap.parse_args()
    import pandas as pd
    csv = os.path.join(HERE, "results", args.out)
    for combo in args.combos:
        na, lr = combo.split(":"); na = int(na); lr = float(lr)
        mean, std = run_config(na, lr)
        row = dict(n_accum=na, lr=lr, eff_lr=round(na * lr, 5),
                   livebn_mean=round(mean, 2), livebn_std=round(std, 2))
        if os.path.exists(csv):
            df = pd.read_csv(csv); df = df[~((df.n_accum == na) & (df.lr == lr))]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(csv, index=False)
        print(f"n_accum={na:2d} lr={lr:<7g} eff_lr={na*lr:<7g} | livebn={mean:5.1f}±{std:4.1f}", flush=True)


if __name__ == "__main__":
    main()
