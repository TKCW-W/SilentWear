# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
AdaBN + full-model on-device FT — b1->b2 sweep over (n_accum, lr).  See AdaBN_Full_training.md.

Per config, per fold:
  collect population BN stats over ALL of batch-1 (forward-only, label-free) -> running buffers;
  freeze them; FULL-train (conv + BN gamma,beta + fc) on 30% of batch-1 with the on-device recipe
  (SGD no-momentum, fixed lr, 40 epochs, batch-1 + n_accum SUM accumulation); eval on batch 2.

Two eval variants: ft_frozen (stats from before training) and ft_recollect (stats re-collected on
the FT data at the final weights).  Usage: python3 adabn_full_training.py 1:0.01 4:0.001 ... --out X.csv
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from idea2_alt_norm import SpeechNetNorm
from windowing import load_windows, stratified_draw
from ondevice_ft import balanced_accuracy

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND, FOLDS = "S01", "vocalized", [1, 2, 3]
EPOCHS = 40


def make_model(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model_state_dict", sd)
    m = SpeechNetNorm(norm="bn", p_dropout=0.0); m.load_state_dict(sd); m.eval()  # no dropout (on-device)
    return m


def collect_bn_stats(model, X):
    """Forward-only, label-free: population per-channel mean/var over all windows -> running buffers.
    (Single population pass == the on-device one-at-a-time accumulation of mu, sigma^2.)"""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = None       # cumulative -> exact population stats
    model.train()
    with torch.no_grad():
        model(torch.from_numpy(X.astype(np.float32)))        # all windows at once == accumulate over all
    model.eval()                                             # freeze: buffers now hold collected stats


def full_train(model, X, y, lr, n_accum, epochs=EPOCHS, seed=42):
    """Full training with FROZEN BN stats: BN eval-mode (uses collected running stats, not updated);
    conv + BN gamma,beta + fc all trainable. SGD, batch-1, n_accum SUM accumulation."""
    model.eval()                                             # BN uses frozen collected stats
    for p in model.parameters():
        p.requires_grad = True                               # full model trains (incl. BN affine)
    opt = torch.optim.SGD(model.parameters(), lr=lr)         # no momentum / no wd
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(N); opt.zero_grad(); c = 0
        for j in perm:
            crit(model(Xt[j:j + 1]), yt[j:j + 1]).backward(); c += 1   # SUM accumulation
            if c % n_accum == 0:
                opt.step(); opt.zero_grad()
        if c % n_accum:
            opt.step(); opt.zero_grad()
    model.eval(); return model


def run_config(n_accum, lr):
    froz, recol = [], []
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        Xb1, yb1 = load_windows(DATA, SUBJECT, fold, 1, COND, downsample_rest=True)
        Xtr, ytr = stratified_draw(Xb1, yb1, 6, seed=42)     # 30% of batch1 (labeled, for training)
        Xb2, yb2 = load_windows(DATA, SUBJECT, fold, 2, COND, downsample_rest=True)
        m = make_model(ckpt)
        collect_bn_stats(m, Xb1)                             # (1) collect on ALL of batch1
        full_train(m, Xtr, ytr, lr, n_accum)                 # (2) freeze + full-train on 30%
        froz.append(balanced_accuracy(m, Xb2, yb2))          # eval with pre-training collected stats
        collect_bn_stats(m, Xb1)                             # re-collect on FT data at final weights
        recol.append(balanced_accuracy(m, Xb2, yb2))
    return (np.mean(froz), np.std(froz, ddof=1), np.mean(recol), np.std(recol, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("combos", nargs="+", help="n_accum:lr pairs, e.g. 4:0.001")
    ap.add_argument("--out", default="adabn_sweep.csv")
    args = ap.parse_args()
    import pandas as pd
    csv = os.path.join(HERE, "results", args.out)
    for combo in args.combos:
        na, lr = combo.split(":"); na = int(na); lr = float(lr)
        fm, fs, rm, rs = run_config(na, lr)
        row = dict(n_accum=na, lr=lr, eff_lr=round(na * lr, 5),
                   ft_frozen_mean=round(fm, 2), ft_frozen_std=round(fs, 2),
                   ft_recollect_mean=round(rm, 2), ft_recollect_std=round(rs, 2))
        if os.path.exists(csv):
            df = pd.read_csv(csv); df = df[~((df.n_accum == na) & (df.lr == lr))]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(csv, index=False)
        print(f"n_accum={na:2d} lr={lr:<7g} eff_lr={na*lr:<7g} | frozen={fm:5.1f}±{fs:4.1f} "
              f"recollect={rm:5.1f}±{rs:4.1f}", flush=True)


if __name__ == "__main__":
    main()
