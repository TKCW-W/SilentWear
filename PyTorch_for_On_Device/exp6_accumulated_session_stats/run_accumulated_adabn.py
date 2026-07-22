# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 6 — ACCUMULATED-session AdaBN (streaming): don't reset BN stats per batch; pool across session 3.

Instead of recollecting BN stats fresh on each batch (exp5 streaming) or on the current batch (our
transductive), we accumulate a per-channel pooled estimate over ALL session-3 batches seen so far:
maintain running Σx, Σx², N (proper pooled μ, σ² — not an EMA of per-batch variances), updated online
at each batch's current weights. Each batch is classified with the accumulated stats from the previous
batches (streaming). By the end of the session the stats reflect the whole session (~900 windows), so
they are more robust (outlier-diluted) and lower-variance than any single batch's.

Protocol (weights W and pooled accumulator ACC both carry forward within the session):
    W <- base ;  ACC <- empty ;  S <- pretrained
    for b = 1..5:
        zs(b) = classify batch b using (W, S)                    # S = pooled stats over b1..b-1
        if b < 5:
            ACC += (Σx, Σx², N) of batch b at current W          # accumulate (online, at current weights)
            S <- ACC.mean / ACC.var  -> set into BN buffers      # session-so-far pooled stats
            W <- fine-tune W on 30% of batch b (BN frozen at S)
    (zs(1) = base + pretrained = no-FT cold start.)

Comparison target: exp5 streaming (reset per batch) S01 = 85.42; transductive = 87.78. Config n8/3e-4.
CLI:  python3 run_accumulated_adabn.py S01 --out results/accum_incr_S01.csv
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model, full_train   # noqa: E402
from windowing import load_windows, stratified_draw      # noqa: E402
from ondevice_ft import balanced_accuracy                # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
FOLDS = [1, 2, 3]
N_ACCUM, LR, EPOCHS = 8, 3e-4, 40


def accumulate_batch(model, X, first):
    """Accumulate batch X into the BN running stats across the session, via PyTorch's native cumulative
    running-stat update (momentum=None). One whole-batch TRAIN-mode forward so each BN layer normalizes
    by its own fresh stats (correct upstream normalization); NOT reset between batches, so stats pool
    over all session batches seen so far. `first`=True zeroes the accumulator at the session start.
    (NB: a per-window hook accumulation in eval mode is WRONG — deep layers would be normalized by stale
    upstream stats; this native path avoids that.)"""
    if first:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats(); m.momentum = None   # cumulative average across batches
    model.train()
    with torch.no_grad():
        model(torch.from_numpy(X.astype(np.float32)))        # whole batch -> cumulative running stats
    model.eval()


def run_fold(subject, cond, fold):
    ckpt = (f"{ART}/inter_session_ft/{subject}/{cond}/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{fold}.pt")
    base = make_model(ckpt); m = make_model(ckpt)
    rows = []; started = False
    for b in range(1, 6):
        Xe, ye = load_windows(DATA, subject, fold, b, cond, downsample_rest=True)
        no_ft = balanced_accuracy(base, Xe, ye)
        zs = balanced_accuracy(m, Xe, ye)          # classify with pooled stats from previous batches
        rows.append((b, no_ft, zs))
        if b != 5:
            accumulate_batch(m, Xe, first=not started); started = True   # pool batch b into session stats
            Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
            full_train(m, Xtr, ytr, LR, N_ACCUM, epochs=EPOCHS)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject"); ap.add_argument("--cond", default="vocalized"); ap.add_argument("--out", required=True)
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
        rows.append(dict(batch=b, no_ft_mean=round(nf.mean(), 2),
                         accum_adabn_mean=round(zs.mean(), 2), accum_adabn_std=round(zs.std(ddof=1), 2)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, a.out), index=False)
    m25 = np.mean([r["accum_adabn_mean"] for r in rows if r["batch"] >= 2])
    print(df.to_string(index=False))
    print(f"[{a.subject}] accumulated-AdaBN mean b2-5 = {m25:.2f}  (vs exp5 streaming-reset & transductive)", flush=True)


if __name__ == "__main__":
    main()
