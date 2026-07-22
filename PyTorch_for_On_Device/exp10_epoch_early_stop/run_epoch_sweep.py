# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 10 — is streaming AdaBN overfitting by epoch 40, and would fewer epochs do better?

Two parts, answering the two forms of the question:

  A. DIAGNOSTIC learning curve (oracle / peeks at eval batch, upper bound — NOT a recipe).
     One round: collect BN stats on ALL of batch1, FULL-train on 30% of batch1 (frozen stats), and
     evaluate the HELD-OUT batch2 at a series of epoch checkpoints. Shows the *shape*: does held-out
     accuracy peak below 40 and decline (overfitting) or keep rising/flat? Averaged over 3 folds.

  B. DEPLOYABLE fixed-epoch sweep (legitimate — a global constant, chosen like we chose 40).
     Run the FULL streaming AdaBN protocol (exp5) end-to-end with EPOCHS fixed to each value in a grid,
     report mean b2-5. Tells us whether a lower FIXED epoch count actually beats 40 in the realistic
     (no test-peeking) setting.

Config carried from exp5: AdaBN n_accum=8, lr=3e-4, 30% data, S01 vocalized, 3 folds.
CLI:  python3 run_epoch_sweep.py --subject S01
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model, collect_bn_stats                  # noqa: E402
from windowing import load_windows, stratified_draw                          # noqa: E402
from ondevice_ft import balanced_accuracy                                     # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
FOLDS = [1, 2, 3]
N_ACCUM, LR = 8, 3e-4
CKPTS = [2, 5, 10, 15, 20, 30, 40, 60, 80]     # diagnostic learning-curve checkpoints
E_GRID = [5, 10, 20, 30, 40, 60]               # deployable fixed-epoch grid


def ckpt_path(subject, cond, fold):
    return (f"{ART}/inter_session_ft/{subject}/{cond}/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{fold}.pt")


def full_train_capped(model, X, y, lr, n_accum, epochs, seed=42):
    """Identical to adabn_full_training.full_train but with an explicit epoch cap (used per E)."""
    model.eval()                                              # frozen BN (collected stats)
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(N); opt.zero_grad(); c = 0
        for j in perm:
            crit(model(Xt[j:j + 1]), yt[j:j + 1]).backward(); c += 1
            if c % n_accum == 0:
                opt.step(); opt.zero_grad()
        if c % n_accum:
            opt.step(); opt.zero_grad()
    model.eval(); return model


def train_with_snapshots(model, X, y, lr, n_accum, ckpts, Xe, ye, seed=42):
    """Train to max(ckpts); at each checkpoint epoch, eval held-out (Xe,ye). Returns {epoch: acc}."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y); rng = np.random.default_rng(seed)
    ckset = set(ckpts); accs = {}
    for ep in range(1, max(ckpts) + 1):
        perm = rng.permutation(N); opt.zero_grad(); c = 0
        for j in perm:
            crit(model(Xt[j:j + 1]), yt[j:j + 1]).backward(); c += 1
            if c % n_accum == 0:
                opt.step(); opt.zero_grad()
        if c % n_accum:
            opt.step(); opt.zero_grad()
        if ep in ckset:
            accs[ep] = balanced_accuracy(model, Xe, ye)       # frozen BN preserved (never call .train())
    return accs


# ---------- Part A: diagnostic learning curve on the b1 -> b2 round ----------
def part_a(subject, cond):
    per_fold = {e: [] for e in CKPTS}
    for fold in FOLDS:
        Xb1, yb1 = load_windows(DATA, subject, fold, 1, cond, downsample_rest=True)
        Xb2, yb2 = load_windows(DATA, subject, fold, 2, cond, downsample_rest=True)
        m = make_model(ckpt_path(subject, cond, fold))
        collect_bn_stats(m, Xb1)                              # adapt stats to b1 (as in streaming)
        Xtr, ytr = stratified_draw(Xb1, yb1, 6, seed=42)      # 30% of b1
        accs = train_with_snapshots(m, Xtr, ytr, LR, N_ACCUM, CKPTS, Xb2, yb2)
        for e in CKPTS:
            per_fold[e].append(accs[e])
    return [(e, float(np.mean(per_fold[e])), float(np.std(per_fold[e], ddof=1))) for e in CKPTS]


# ---------- Part B: deployable fixed-epoch streaming sweep ----------
def streaming_mean_b25(subject, cond, epochs):
    byb = {b: [] for b in range(2, 6)}
    for fold in FOLDS:
        base_ckpt = ckpt_path(subject, cond, fold)
        m = make_model(base_ckpt)                             # carries (W, S) forward
        for b in range(1, 6):
            Xe, ye = load_windows(DATA, subject, fold, b, cond, downsample_rest=True)
            if b >= 2:
                byb[b].append(balanced_accuracy(m, Xe, ye))   # classify b with previous round's (W,S)
            if b != 5:
                collect_bn_stats(m, Xe)                        # adapt stats to b (after eval)
                Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
                full_train_capped(m, Xtr, ytr, LR, N_ACCUM, epochs)
    return float(np.mean([np.mean(byb[b]) for b in range(2, 6)]))


def part_b(subject, cond):
    return [(e, streaming_mean_b25(subject, cond, e)) for e in E_GRID]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S01"); ap.add_argument("--cond", default="vocalized")
    a = ap.parse_args()
    import pandas as pd
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    # Part A (oracle diagnostic) dropped: it peeks at the eval batch and is not deployable on-device.
    # Only Part B — the fixed global epoch count, chosen like we chose 40 — is a real on-device recipe.
    print("=== Part B: deployable fixed-epoch streaming sweep (mean b2-5) ===", flush=True)
    b_rows = part_b(a.subject, a.cond)
    dfb = pd.DataFrame([dict(epochs=e, streaming_mean_b25=round(v, 2)) for e, v in b_rows])
    dfb.to_csv(os.path.join(HERE, f"results/fixed_epoch_sweep_{a.subject}.csv"), index=False)
    print(dfb.to_string(index=False), flush=True)
    best = max(b_rows, key=lambda r: r[1])
    print(f"best fixed epochs = {best[0]} -> {best[1]:.2f} ; epoch 40 -> "
          f"{dict(b_rows)[40]:.2f}", flush=True)


if __name__ == "__main__":
    main()
