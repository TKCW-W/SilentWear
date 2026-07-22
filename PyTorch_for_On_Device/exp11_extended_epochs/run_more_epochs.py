# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 11 — extend the fixed-epoch sweep FAR past 40 to test under-training.

Rationale: we use plain SGD (no momentum, no weight decay), batch-1 + sum-accum, small lr 3e-4 — much
slower to converge than the paper's Adam. So 40 epochs may be UNDER-trained rather than converged/
overfitted. The exp10 sweep was flat 20->60 (no overfitting), but that doesn't prove the loss has
converged. Here we push EPOCHS to {40,80,120,160,200} on the full streaming AdaBN protocol and, for
each, report BOTH held-out mean b2-5 AND the mean final-epoch TRAINING loss across all FT rounds — so
we can distinguish "loss converged, held-out plateaued (ceiling)" from "loss still dropping, held-out
still climbing (was under-trained)".

n_accum=8, lr=3e-4, 30% data, S01 vocalized, 3 folds.
CLI:  python3 run_more_epochs.py --subject S01
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
E_GRID = [40, 80, 120, 160, 200]


def ckpt_path(subject, cond, fold):
    return (f"{ART}/inter_session_ft/{subject}/{cond}/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{fold}.pt")


def full_train_loss(model, X, y, lr, n_accum, epochs, seed=42):
    """Full-model FT (frozen BN); returns the mean per-window training loss of the FINAL epoch."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y); rng = np.random.default_rng(seed)
    last_losses = []
    for ep in range(epochs):
        perm = rng.permutation(N); opt.zero_grad(); c = 0
        ep_losses = []
        for j in perm:
            loss = crit(model(Xt[j:j + 1]), yt[j:j + 1]); loss.backward(); c += 1
            ep_losses.append(float(loss.detach()))
            if c % n_accum == 0:
                opt.step(); opt.zero_grad()
        if c % n_accum:
            opt.step(); opt.zero_grad()
        if ep == epochs - 1:
            last_losses = ep_losses
    model.eval()
    return float(np.mean(last_losses))


def streaming_run(subject, cond, epochs):
    byb = {b: [] for b in range(2, 6)}; final_losses = []
    for fold in FOLDS:
        m = make_model(ckpt_path(subject, cond, fold))
        for b in range(1, 6):
            Xe, ye = load_windows(DATA, subject, fold, b, cond, downsample_rest=True)
            if b >= 2:
                byb[b].append(balanced_accuracy(m, Xe, ye))
            if b != 5:
                collect_bn_stats(m, Xe)
                Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
                final_losses.append(full_train_loss(m, Xtr, ytr, LR, N_ACCUM, epochs))
    acc = float(np.mean([np.mean(byb[b]) for b in range(2, 6)]))
    return acc, float(np.mean(final_losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S01"); ap.add_argument("--cond", default="vocalized")
    a = ap.parse_args()
    import pandas as pd
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    print("=== Exp10b: extended fixed-epoch sweep (held-out b2-5 + final train loss) ===", flush=True)
    rows = []
    for e in E_GRID:
        acc, loss = streaming_run(a.subject, a.cond, e)
        rows.append(dict(epochs=e, streaming_mean_b25=round(acc, 2), final_train_loss=round(loss, 4)))
        print(f"epochs={e:4d}  held-out b2-5={acc:5.2f}  final_train_loss={loss:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, f"results/more_epochs_{a.subject}.csv"), index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
