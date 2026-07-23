# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 13 — progressive unfreezing: is there a middle ground between head-only and full training?

Freeze the whole body, then unfreeze the LAST K blocks (+ fc) and fine-tune. K=0 = head-only (fc only),
K=5 = full model. Frozen pretrained BN stats (setting-3 regime, no recollection), streaming incremental
b1->b5, S01 vocalized, 3 folds, n_accum 4, 40 epochs, 30% data, fixed order. Sweep an lr grid per K
(fewer unfrozen blocks tolerate a higher lr), report the best lr per K.

Question: does any K (at its best lr) beat head-only (84.68) / approach the paper, or does partial
fine-tuning just land between head-only and full (85.65) as capacity/overfitting theory predicts?

CLI:  python3 run_progressive_unfreeze.py
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model                # noqa: E402
from windowing import load_windows, stratified_draw       # noqa: E402
from ondevice_ft import balanced_accuracy                 # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND, FOLDS = "S01", "vocalized", [1, 2, 3]
N_ACCUM, EPOCHS = 4, 40
K_GRID = [0, 1, 2, 3, 4, 5]
LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2]


def set_trainable(model, k):
    """Freeze all; unfreeze fc + the LAST k blocks (conv + BN affine)."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    nb = len(model.blocks)
    for blk in list(model.blocks)[nb - k:] if k > 0 else []:
        for p in blk.parameters():
            p.requires_grad = True


def train_lastk(model, X, y, lr, k, seed=42):
    model.eval()                                           # frozen pretrained BN stats
    set_trainable(model, k)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr)                   # no momentum/wd
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y)
    for _ in range(EPOCHS):
        opt.zero_grad(); c = 0
        for j in range(N):                                 # fixed order (head-only convention)
            crit(model(Xt[j:j + 1]), yt[j:j + 1]).backward(); c += 1
            if c % N_ACCUM == 0:
                opt.step(); opt.zero_grad()
        if c % N_ACCUM:
            opt.step(); opt.zero_grad()
    model.eval(); return model


def stream_mean_b25(k, lr):
    byb = {b: [] for b in range(2, 6)}
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        m = make_model(ckpt)
        for b in range(1, 6):
            Xe, ye = load_windows(DATA, SUBJECT, fold, b, COND, downsample_rest=True)
            if b >= 2:
                byb[b].append(balanced_accuracy(m, Xe, ye))
            if b != 5:
                Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
                train_lastk(m, Xtr, ytr, lr, k)
    return float(np.mean([np.mean(byb[b]) for b in range(2, 6)]))


def main():
    import pandas as pd
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows = []
    print(f"{'K':>2} {'trainable':>10} " + " ".join(f"lr={lr:<7g}" for lr in LR_GRID) + "  best", flush=True)
    for k in K_GRID:
        accs = {}
        for lr in LR_GRID:
            accs[lr] = stream_mean_b25(k, lr)
            rows.append(dict(K=k, lr=lr, mean_b25=round(accs[lr], 2)))
        best_lr = max(accs, key=accs.get)
        tag = "fc only" if k == 0 else f"last {k} blk+fc"
        print(f"{k:>2} {tag:>10} " + " ".join(f"{accs[lr]:>7.2f} " for lr in LR_GRID) +
              f"  best {accs[best_lr]:.2f} @lr{best_lr:g}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results/progressive_unfreeze_S01.csv"), index=False)
    best = df.loc[df.groupby("K")["mean_b25"].idxmax()][["K", "lr", "mean_b25"]]
    print("\nbest per K:\n" + best.to_string(index=False), flush=True)
    print("\nreference: head-only 84.68 ; full frozen-stat 85.65 ; paper 88.24", flush=True)


if __name__ == "__main__":
    main()
